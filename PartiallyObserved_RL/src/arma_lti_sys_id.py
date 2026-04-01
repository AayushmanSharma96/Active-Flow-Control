import numpy as np
from arma_ltv_sys_id import ARMA_LTV_SysID


class ARMA_LTI_SysID(ARMA_LTV_SysID):
    """
    LTI System Identification for LQR regulator design.
    
    Runs multiple short rollouts from a suppressed steady state,
    collects augmented state transitions, and fits a SINGLE (A, B) pair
    using SVD with energy thresholding.
    
    Inherits rollout infrastructure from ARMA_LTV_SysID.
    
    Usage:
        lti_id = ARMA_LTI_SysID(model, n_x, n_u, n_z, q, q_u,
                                  rollout_horizon=10, n_rollouts=12, pert_sigma=0.3)
        A, B = lti_id.identify(init_solution)
    """

    def __init__(self, MODEL, n_x, n_u, n_z, q, q_u,
                 rollout_horizon=10, n_rollouts=5, pert_sigma=3.0,
                 svd_energy_threshold=1.00):
        """
        Args:
            MODEL: SimulatePyFR instance
            n_x: state dimension (same as n_z for observation-based)
            n_u: control dimension
            n_z: observation dimension (2 for [CD, CL])
            q: observation history length
            q_u: control history length
            rollout_horizon: steps per rollout (short! system is unstable)
            n_rollouts: number of independent rollouts
            pert_sigma: control perturbation std dev (keep small for linear regime)
            svd_energy_threshold: energy threshold for SVD truncation
        """
        # Initialize parent with rollout_horizon as N, n_rollouts as n_samples
        super().__init__(MODEL, n_x, n_u, n_z, q, q_u, 
                         N=rollout_horizon, n_samples=n_rollouts, pert_sigma=pert_sigma)
        
        self.rollout_horizon = rollout_horizon
        self.n_rollouts = n_rollouts
        self.svd_energy_threshold = svd_energy_threshold
        self.aug_dim = n_z * q + n_u * max(0, q_u - 1)
        self.X_eq = None

    def identify(self, init_solution):
        """
        Main entry point: run rollouts and fit LTI model.
        
        Args:
            init_solution: path to suppressed-state .pyfrs file
            
        Returns:
            A: (n_aug, n_aug) LTI state transition matrix
            B: (n_aug, n_u) LTI input matrix
        """
        print("=" * 70)
        print("LTI SYSTEM IDENTIFICATION FOR LQR REGULATOR")
        print("=" * 70)
        print(f"  n_z={self.n_z}, q={self.q}, q_u={self.q_u}, n_aug={self.aug_dim}")
        print(f"  Rollouts: {self.n_rollouts} x {self.rollout_horizon} steps")
        print(f"  Perturbation sigma: {self.sigma}")
        print(f"  SVD energy threshold: {self.svd_energy_threshold}")
        print()

        # Run rollouts from suppressed state
        delta_Z, delta_U = self._run_rollouts(init_solution)

        # Fit single (A, B) pair
        A, B = self._fit_lti(delta_Z, delta_U)

        return A, B

    def _run_rollouts(self, init_solution):
        """
        Run multiple short rollouts from the suppressed state.
        
        Returns:
            delta_Z: (n_rollouts, n_z, rollout_horizon+1) observation deviations
            delta_U: (n_rollouts, n_u, rollout_horizon) control perturbations
        """
        self.model.set_mode('rollout')

        n_z, n_u = self.n_z, self.n_u
        H = self.rollout_horizon

        # Generate small control perturbations
        U_pertb = self.sigma * np.random.normal(0, 1, (H, n_u, self.n_rollouts))

        # Storage
        X_all = np.zeros((H + 1, n_z, self.n_rollouts))

        # Run a baseline (zero control) rollout to get nominal trajectory
        X_all[:, :, 0] = self.model.simulate_trajectory(
            y_init=init_solution,
            u=U_pertb[:, :, 0].flatten(),
            horizon=H
        )
        X_eq = 0*X_all[0, :, 0]
        X_eq[1] = 0.0 # Target CL=0
        self.X_eq = X_eq

        # Run perturbed rollouts
        print(f"\n  Running {self.n_rollouts} perturbed rollouts...")
        

        for j in range(1, self.n_rollouts):
            X_all[:, :, j] = self.model.simulate_trajectory(
                y_init=init_solution,
                u=U_pertb[:, :, j].flatten(),
                horizon=H
            )

        # Compute deviations from baseline
        delta_Z = np.zeros((self.n_rollouts, n_z, H + 1))
        delta_U = np.zeros((self.n_rollouts, n_u, H))

        for j in range(self.n_rollouts):
            for t in range(H + 1):
                delta_Z[j, :, t] = X_all[t, :, j] - X_eq
            for t in range(H):
                delta_U[j, :, t] = U_pertb[t, :, j]


        print(f"\n  Delta_Z stats:")
        for t in range(min(5, H)):
            var = np.var(delta_Z[:, :, t], axis=0)
            print(f"    t={t}: var(CD)={var[0]:.6f}, var(CL)={var[1]:.6f}")

        return delta_Z, delta_U

    def _fit_lti(self, delta_Z, delta_U):
        """
        Fit a single LTI (A, B) from all collected transitions.
        
        Uses ARMA companion structure:
            z_aug[t+1] = A @ z_aug[t] + B @ u[t]
        
        Only the top n_z rows are learned; rest is companion shifting.
        
        Args:
            delta_Z: (n_rollouts, n_z, H+1)
            delta_U: (n_rollouts, n_u, H)
            
        Returns:
            A: (n_aug, n_aug)
            B: (n_aug, n_u)
        """
        n_z, n_u, q, q_u = self.n_z, self.n_u, self.q, self.q_u
        aug_dim = self.aug_dim
        H = self.rollout_horizon
        n_rollouts = self.n_rollouts

        start_idx = max(q, q_u)
        regressor_dim = n_z * q + n_u * q_u  # [z_history | u_history | u_current]

        # Collect ALL transitions across ALL rollouts
        M_list = []
        target_list = []

        for j in range(n_rollouts):
            for t in range(start_idx, H):
                # Build regressor: [δz[t], δz[t-1], ..., δz[t-q+1], δu[t-1], ..., δu[t-q_u]]
                row = np.zeros(regressor_dim)

                # Observation history (newest first)
                for i in range(q):
                    row[i * n_z:(i + 1) * n_z] = delta_Z[j, :, t - i]

                # Control history (newest first): u[t-1], u[t-2], ..., then u[t] at end
                # Match ARMA_LTV convention: [u[t-1], ..., u[t-q_u]]
                for i in range(q_u):
                    row[q * n_z + i * n_u:q * n_z + (i + 1) * n_u] = delta_U[j, :, t - i - 1]

                # Target: δz[t+1]
                target = delta_Z[j, :, t + 1]

                M_list.append(row)
                target_list.append(target)

        M = np.array(M_list)       # (n_transitions, regressor_dim)
        target = np.array(target_list)  # (n_transitions, n_z)

        n_transitions = M.shape[0]
        print(f"\n  LTI Regression:")
        print(f"    Transitions collected: {n_transitions}")
        print(f"    Regressor dimension: {regressor_dim}")
        print(f"    Parameters to fit: {n_z} x {regressor_dim} = {n_z * regressor_dim}")
        print(f"    Overdetermination ratio: {n_transitions / regressor_dim:.1f}x")

        # ========================================
        # SVD with energy thresholding
        # ========================================
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)

        energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        n_keep = np.searchsorted(energy, self.svd_energy_threshold) + 1
        n_keep = min(n_keep, len(s))

        print(f"\n  SVD Analysis:")
        print(f"    Total singular values: {len(s)}")
        print(f"    Keeping: {n_keep} ({self.svd_energy_threshold * 100:.2f}% energy)")
        print(f"    Condition (full): {s[0] / s[-1]:.2e}")
        print(f"    Condition (truncated): {s[0] / s[n_keep - 1]:.2e}")
        print(f"    Singular values: {s[:min(10, len(s))]}")

        # Truncated pseudoinverse
        s_inv = np.zeros_like(s)
        s_inv[:n_keep] = 1.0 / s[:n_keep]

        fitcoef = ((Vt.T * s_inv) @ (U_svd.T @ target)).T  # (n_z, regressor_dim)

        # Reconstruction error
        pred = M @ fitcoef.T
        rmse = np.sqrt(np.mean((pred - target) ** 2, axis=0))
        r2 = 1.0 - np.sum((target - pred) ** 2, axis=0) / np.sum((target - target.mean(axis=0)) ** 2, axis=0)
        print(f"\n  Fit quality:")
        print(f"    RMSE: CD={rmse[0]:.6f}, CL={rmse[1]:.6f}")
        print(f"    R²:   CD={r2[0]:.4f}, CL={r2[1]:.4f}")

        # ========================================
        # Build companion (A, B) matrices
        # ========================================
        A = np.zeros((aug_dim, aug_dim))
        B = np.zeros((aug_dim, n_u))

        # Top n_z rows from regression
        # fitcoef layout: [alpha_0, alpha_1, ..., alpha_{q-1} | beta_0, beta_1, ..., beta_{q_u-1}]
        # alpha_i corresponds to z[t-i], beta_i corresponds to u[t-i-1]
        
        # A top rows: observation coefficients + control history coefficients
        A[:n_z, :n_z * q] = fitcoef[:, :n_z * q]  # alpha coefficients
        
        if q_u > 1:
            # beta_1, ..., beta_{q_u-1} go into A (control history part)
            A[:n_z, n_z * q:] = fitcoef[:, n_z * q + n_u:]  # skip beta_0 (goes to B)

        # B top rows: beta_0 (current control coefficient)
        B[:n_z, :] = fitcoef[:, n_z * q:n_z * q + n_u]

        # Observation shift block: z[t-i] → z[t-i-1]
        if q > 1:
            A[n_z:n_z * q, :n_z * (q - 1)] = np.eye(n_z * (q - 1))

        # Control history: u[t] enters, shifts down
        if q_u > 1:
            ctrl_start = n_z * q
            B[ctrl_start:ctrl_start + n_u, :] = np.eye(n_u)

            if q_u > 2:
                A[ctrl_start + n_u:ctrl_start + n_u * (q_u - 1),
                  ctrl_start:ctrl_start + n_u * (q_u - 2)] = np.eye(n_u * (q_u - 2))

        # ========================================
        # Validation
        # ========================================
        eigs = np.linalg.eigvals(A)
        eig_mags = np.sort(np.abs(eigs))[::-1]

        print(f"\n  System properties:")
        print(f"    A shape: {A.shape}")
        print(f"    B shape: {B.shape}")
        print(f"    Eigenvalue magnitudes: {eig_mags[:5]}")
        print(f"    Max |eig|: {eig_mags[0]:.6f}")

        if eig_mags[0] > 1.0:
            print(f"    ⚠️  Open-loop unstable (expected — LQR will stabilize)")
        else:
            print(f"    ✓  Open-loop stable")

        # Check controllability (rank of [B, AB, A²B, ...])
        C_mat = B.copy()
        Ak = np.eye(aug_dim)
        for i in range(aug_dim - 1):
            Ak = Ak @ A
            C_mat = np.hstack([C_mat, Ak @ B])
        ctrl_rank = np.linalg.matrix_rank(C_mat)
        print(f"    Controllability rank: {ctrl_rank}/{aug_dim}", end="")
        if ctrl_rank == aug_dim:
            print(" ✓ Fully controllable")
        else:
            print(f" ⚠️  Not fully controllable")

        print("=" * 70)

        return A, B


def design_lqr(A, B, Q, R):
    """
    Design infinite-horizon LQR regulator.
    
    Args:
        A: (n_aug, n_aug) state transition
        B: (n_aug, n_u) input matrix
        Q: (n_aug, n_aug) state penalty
        R: (n_u, n_u) control penalty
        
    Returns:
        K_lqr: (n_u, n_aug) feedback gain (u = -K_lqr @ z_aug)
        P: (n_aug, n_aug) solution to DARE
    """
    from scipy.linalg import solve_discrete_are

    print("\nDesigning LQR regulator...")
    print(f"  Q diag (first 6): {np.diag(Q)[:6]}")
    print(f"  R: {R}")

    P = solve_discrete_are(A, B, Q, R)
    K_lqr = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    # Closed-loop stability check
    A_cl = A - B @ K_lqr
    cl_eigs = np.linalg.eigvals(A_cl)
    cl_mags = np.sort(np.abs(cl_eigs))[::-1]

    print(f"\n  LQR gain norm: {np.linalg.norm(K_lqr):.4f}")
    print(f"  Closed-loop eigenvalue magnitudes: {cl_mags[:5]}")
    print(f"  Max |eig| closed-loop: {cl_mags[0]:.6f}")

    if cl_mags[0] < 1.0:
        print(f"  ✓ Closed-loop STABLE (margin: {1.0 - cl_mags[0]:.4f})")
    else:
        print(f"  ✗ Closed-loop UNSTABLE — increase R or check model")

    return K_lqr, P