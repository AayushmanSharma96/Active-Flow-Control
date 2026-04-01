import numpy as np
from ltv_sys_id import LTV_SysID
class ARMA_LTV_SysID(LTV_SysID):

    def __init__(self, MODEL, n_x, n_u, n_z, q, q_u, N, n_samples=500, pert_sigma = 1e-3):
        """
        ARMA LTV System Identification
        Args:
            MODEL: dynamics model
            n_x: full state dimension
            n_u: control dimension  
            n_z: observation dimension (e.g., 1 for pendulum, 2 for cartpole)
            q: state history length
            q_u: control history length 
            n_samples: number of perturbed trajectories
            pert_sigma: perturbation standard deviation for actions
        """
        # super().__init__(MODEL, n_x, n_u, N, n_samples, pert_sigma)
        LTV_SysID.__init__(self, MODEL, n_x, n_u, N, n_samples = n_samples, pert_sigma = pert_sigma)
        self.n_z = n_z
        self.q = q
        self.q_u = q_u

    def traj_sys_id(self, x_nom, u_nom, roll_start = 0):
        '''
            System identification for a given nominal state and control
            x_nom = (N+1, n_x, 1)
            u_nom = (N, n_u, 1)
            returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################
        n_z, N = self.n_z, self.N
		##########################################################################################
        # Generating perturbations
        X_pertb, U_pertb = self.generate_rollouts(x_nom, u_nom, roll_start)
        
        # X_pertb = np.load('checkpoints/rollouts/rollouts_Re100_works/z_rollouts_it_4.npy')
        # U_pertb = np.load('checkpoints/rollouts/rollouts_Re100_works/u_rollouts_it_4.npy')
        # X_pertb = np.load('X_pertb_rollouts.npy')
        # U_pertb = np.load('U_pertb_rollouts.npy')
        
        # Generating delta_z for all rollouts
        delta_Z = np.zeros((N+1, n_z, self.n_samples))
        for i in range(N+1):
            delta_Z[i, :, :] = X_pertb[i, :, :] - x_nom[i, :, 0:1]
        
        delta_Z = delta_Z.transpose(2, 1, 0)  # (n_samples, n_z, N+1)
        U_pertb = U_pertb.transpose(2, 1, 0)  # (n_samples, n_u, N)

        print("Delta_Z variance across samples:")
        for t in range(self.N):
            var = np.var(delta_Z[:, :, t], axis=0)
            print(f"t={t}: {np.mean(var)}")
        
        return self.arma_fit(delta_Z, U_pertb)
    

    def arma_fit(self, delta_Z, delta_U):
        """
        ARMA LTV fitting with correct indexing (matches paper formulation)
        delta_Z : (n_samples, n_z, N+1)
        delta_U : (n_samples, n_u, N)
        returns : AB_aug : (N, aug_dim, aug_dim + n_u)
        """
        n_z, n_u, q, q_u, N, n_samples = self.n_z, self.n_u, self.q, self.q_u, self.N, self.n_samples
        
        # Augmented state dimension
        aug_dim = n_z*q + n_u*max(0, q_u-1)
        
        # Output arrays
        A_aug = np.zeros((N, aug_dim, aug_dim))
        B_aug = np.zeros((N, aug_dim, n_u))
        
        # Pre-allocate regressor arrays
        M = np.zeros((n_samples, n_z*q + n_u*q_u))
        target = np.zeros((n_samples, n_z))
        
        # [Keep all the pre-computed blocks - same as before]
        state_shift_block = None
        if q > 1:
            state_eye = np.eye(n_z*(q-1))
            state_zeros = np.zeros((n_z*(q-1), n_z + n_u*max(0, q_u-1)))
            state_shift_block = np.hstack([state_eye, state_zeros])
        
        ctrl_eye = None
        if q_u > 2:
            ctrl_eye = np.eye(n_u*(q_u-2))
        
        b_state_zeros = np.zeros((n_z*max(0, q-1), n_u))
        
        if q_u > 1:
            b_ctrl_eye = np.eye(n_u)
            if q_u > 2:
                b_ctrl_zeros = np.zeros((n_u*(q_u-2), n_u))
                b_ctrl_block = np.vstack([b_ctrl_eye, b_ctrl_zeros])
            else:
                b_ctrl_block = b_ctrl_eye
        else:
            b_ctrl_block = np.zeros((0, n_u))
        
        
        start_idx = max(q, q_u)
        
        for t in range(start_idx, N):
            
            # Regressor: [δz[t], δz[t-1], ..., δz[t-q+1], δu[t-1], δu[t-2], ..., δu[t-q_u]]
            for i in range(q):
                M[:, i*n_z:(i+1)*n_z] = delta_Z[:, :, t - i]  # CHANGED: t-i instead of t-q+i
            
            # Control history starts at u[t-1]
            for i in range(q_u):
                M[:, q*n_z + i*n_u : q*n_z + (i+1)*n_u] = delta_U[:, :, t - i - 1]  # CHANGED: t-i-1 instead of t-q_u+i
            
            target[:, :] = delta_Z[:, :, t + 1]
            
            

            # Compute SVD
            U, s, Vt = np.linalg.svd(M, full_matrices=False)
            
            # Compute cumulative energy
            s_squared = s**2
            total_energy = np.sum(s_squared)
            cumulative_energy = np.cumsum(s_squared)
            relative_energy = cumulative_energy / total_energy
            energy_threshold = 0.99#99
            
            # Find number of singular values to keep
            n_keep = np.searchsorted(relative_energy, energy_threshold) + 1
            n_keep = min(n_keep, len(s))  # Don't exceed available singular values
            
            # Optional: Print diagnostics for first few timesteps
            if t < start_idx + 3:
                print(f"\n[ARMA SVD t={t}]")
                print(f"  Total singular values: {len(s)}")
                print(f"  Keeping {n_keep} singular values ({n_keep/len(s)*100:.1f}%)")
                print(f"  Energy captured: {relative_energy[n_keep-1]*100:.3f}%")
                print(f"  Condition number (before): {s[0]/s[-1]:.2e}")
                print(f"  Condition number (after):  {s[0]/s[n_keep-1]:.2e}")
                print(f"  Largest singular value:  {s[0]:.2e}")
                print(f"  Smallest kept s.v.:      {s[n_keep-1]:.2e}")
                if n_keep < len(s):
                    print(f"  Largest discarded s.v.:  {s[n_keep]:.2e}")
            
            # Truncated pseudo-inverse
            s_inv = np.zeros_like(s)
            s_inv[:n_keep] = 1.0 / s[:n_keep]
            
            # Solve: fitcoef = M_pinv @ target
            M_pinv = Vt.T @ np.diag(s_inv) @ U.T
            fitcoef = (M_pinv @ target).T # (n_z, n_z*q + n_u*q_u)
            
            out_idx = t  

            # === A_aug construction ================
            # Top row: [α_t, β_{t,2}, ..., β_{t,q_u}]
            if q_u > 1:
                A_aug[out_idx, :n_z, :n_z*q] = fitcoef[:, :n_z*q]
                A_aug[out_idx, :n_z, n_z*q:] = fitcoef[:, n_z*q+n_u:]
            else:
                A_aug[out_idx, :n_z, :n_z*q] = fitcoef[:, :n_z*q]
            
            # State shifting block
            if q > 1 and state_shift_block is not None:
                A_aug[out_idx, n_z:n_z*q, :] = state_shift_block
            
            # Control shifting block
            if q_u > 2 and ctrl_eye is not None:
                row_start = n_z*q + n_u
                col_start = n_z*q + n_u
                row_end = row_start + n_u*(q_u-2)
                col_end = col_start + n_u*(q_u-2)
                
                if row_end <= aug_dim and col_end <= aug_dim:
                    A_aug[out_idx, row_start:row_end, col_start:col_end] = ctrl_eye
            
            # === B_aug construction =======
            # Top part: [β_{t,1}; 0; ...; 0]
            if q > 1:
                B_aug[out_idx, :n_z*q, :] = np.vstack([
                    fitcoef[:, n_z*q:n_z*q+n_u],
                    b_state_zeros
                ])
            else:
                B_aug[out_idx, :n_z, :] = fitcoef[:, n_z*q:n_z*q+n_u]
            
            # Bottom part: control history storage
            if q_u > 1 and n_z*q < aug_dim:
                rows_to_fill = min(aug_dim - n_z*q, b_ctrl_block.shape[0])
                B_aug[out_idx, n_z*q:n_z*q+rows_to_fill, :] = b_ctrl_block[:rows_to_fill, :]
        
        # Fill early timesteps (t < start_idx) with identity
        for t in range(start_idx):
            A_aug[t, :, :] = np.eye(aug_dim)
            B_aug[t, :, :] = 0.0
        
        if N > start_idx:
            A_aug[N-1, :, :] = A_aug[N-2, :, :]  # Copy last valid timestep
            B_aug[N-1, :, :] = B_aug[N-2, :, :]
        
        # Concatenate A_aug and B_aug
        AB_aug = np.concatenate((A_aug, B_aug), axis=2)
        
        return AB_aug