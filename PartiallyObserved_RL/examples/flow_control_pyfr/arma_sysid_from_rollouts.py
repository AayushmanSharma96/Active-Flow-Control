"""
ARMA System Identification from Rollout Data
=============================================

Takes pre-computed rollout deviations and produces:
1. Alpha and beta coefficients (dynamics parameters)
2. A and B matrices in augmented state-space form

Compatible with any q, q_u values.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def arma_system_id(X_pertb, U_pertb, Z_nom, U_nom, q, q_u, verbose=True):
    """
    Perform ARMA system identification from rollout data.
    
    Args:
        X_pertb: (N+1, n_z, n_samples) - perturbed state trajectories
        U_pertb: (N, n_u, n_samples) - perturbed control sequences
        Z_nom: (N+1, n_z, 1) - nominal state trajectory
        U_nom: (N, n_u, 1) - nominal control sequence
        q: observation history length
        q_u: control history length
        verbose: print progress
    
    Returns:
        alphas: (N, n_z, n_z*q) - alpha coefficients at each timestep
        betas: (N, n_z, n_u*q_u) - beta coefficients at each timestep
        A_matrices: (N, n_aug, n_aug) - A matrices in augmented form
        B_matrices: (N, n_aug, n_u) - B matrices in augmented form
        AB_combined: (N, n_aug, n_aug+n_u) - Combined [A|B] for convenience
    """
    
    # Extract dimensions
    N_plus_1, n_z, n_samples = X_pertb.shape
    N = N_plus_1 - 1
    n_u = U_pertb.shape[1]
    
    # Augmented state dimension
    aug_dim = n_z * q + n_u * max(0, q_u - 1)
    
    if verbose:
        print("="*80)
        print("ARMA SYSTEM IDENTIFICATION")
        print("="*80)
        print(f"\nParameters:")
        print(f"  Horizon (N): {N}")
        print(f"  Observations (n_z): {n_z}")
        print(f"  Controls (n_u): {n_u}")
        print(f"  Samples: {n_samples}")
        print(f"  History lengths (q, q_u): ({q}, {q_u})")
        print(f"  Augmented state dim: {aug_dim}")
        print(f"  Overdetermination ratio: {n_samples/aug_dim:.2f}x")
    
    # Compute deviations
    delta_Z = X_pertb - Z_nom  # (N+1, n_z, n_samples)
    delta_U = U_pertb  # (N, n_u, n_samples)
    
    # Transpose for easier indexing: (n_samples, n_z, N+1)
    delta_Z_T = delta_Z.transpose(2, 1, 0)
    delta_U_T = delta_U.transpose(2, 1, 0)
    
    if verbose:
        print(f"\nDeviation statistics:")
        print(f"  delta_Z range: [{delta_Z.min():.4f}, {delta_Z.max():.4f}]")
        print(f"  delta_Z std: {delta_Z.std():.4f}")
        print(f"  delta_U range: [{delta_U.min():.4f}, {delta_U.max():.4f}]")
        print(f"  delta_U std: {delta_U.std():.4f}")
    
    # Allocate output arrays
    alphas = np.zeros((N, n_z, n_z * q))
    betas = np.zeros((N, n_z, n_u * q_u))
    A_matrices = np.zeros((N, aug_dim, aug_dim))
    B_matrices = np.zeros((N, aug_dim, n_u))
    
    # Pre-compute constant blocks for A matrix
    # State shifting block: z[t] -> z[t-1]
    state_shift_block = None
    if q > 1:
        state_eye = np.eye(n_z * (q - 1))
        state_zeros = np.zeros((n_z * (q - 1), n_z + n_u * max(0, q_u - 1)))
        state_shift_block = np.hstack([state_eye, state_zeros])
    
    # Control shifting block: u[t-1] -> u[t-2]
    ctrl_shift_block = None
    if q_u > 2:
        ctrl_eye = np.eye(n_u * (q_u - 2))
        ctrl_shift_block = ctrl_eye
    
    # Pre-allocate regressor and target
    M = np.zeros((n_samples, n_z * q + n_u * q_u))
    target = np.zeros((n_samples, n_z))
    
    # Main loop - system ID at each timestep
    start_idx = max(q, q_u)
    
    if verbose:
        print(f"\nPerforming system ID from t={start_idx} to t={N}...")
    
    for t in range(start_idx, N):
        if verbose and t % 50 == 0:
            print(f"  Progress: {t}/{N}")
        
        # Build regressor matrix M for this timestep
        # Regressor: [z[t], z[t-1], ..., z[t-q+1], u[t-1], u[t-2], ..., u[t-q_u]]
        
        # State history: [z[t], z[t-1], ..., z[t-q+1]]
        for i in range(q):
            M[:, i*n_z:(i+1)*n_z] = delta_Z_T[:, :, t - i]
        
        # Control history: [u[t-1], u[t-2], ..., u[t-q_u]]
        for i in range(q_u):
            M[:, q*n_z + i*n_u : q*n_z + (i+1)*n_u] = delta_U_T[:, :, t - i - 1]
        
        # Target: δz[t+1]
        target[:, :] = delta_Z_T[:, :, t + 1]
        
        # Least squares solution with ridge regularization
        ridge_lambda = 0
        M_reg = np.vstack([M, np.sqrt(ridge_lambda) * np.eye(M.shape[1])])
        target_reg = np.vstack([target, np.zeros((M.shape[1], n_z))])
        fitcoef, _, _, _ = np.linalg.lstsq(M_reg, target_reg, rcond=None)
        fitcoef = fitcoef.T  # (n_z, n_z*q + n_u*q_u)
        
        # Extract alpha and beta coefficients
        alphas[t, :, :] = fitcoef[:, :n_z*q]
        betas[t, :, :] = fitcoef[:, n_z*q:]
        
        # ===================================================================
        # Build A matrix (augmented state dynamics)
        # ===================================================================
        A = np.zeros((aug_dim, aug_dim))
        
        # Alpha coefficients (for observation history)
        A[:n_z, :n_z*q] = fitcoef[:, :n_z*q]
        
        # Beta coefficients (for control history, excluding current control)
        if q_u > 1:
            A[:n_z, n_z*q:n_z*q + n_u*(q_u-1)] = fitcoef[:, n_z*q + n_u : n_z*q + n_u*q_u]
        
        # State shifting block
        if q > 1 and state_shift_block is not None:
            A[n_z:n_z*q, :] = state_shift_block
        
        # Control shifting block
        if q_u > 2 and ctrl_shift_block is not None:
            row_start = n_z * q + n_u
            col_start = n_z * q + n_u
            row_end = row_start + n_u * (q_u - 2)
            col_end = col_start + n_u * (q_u - 2)
            
            if row_end <= aug_dim and col_end <= aug_dim:
                A[row_start:row_end, col_start:col_end] = ctrl_shift_block
        
        A_matrices[t, :, :] = A
        
        # ===================================================================
        # Build B matrix (control input to augmented state)
        # ===================================================================
        B = np.zeros((aug_dim, n_u))
        
        # Top n_z rows: β_0 (immediate effect of u[t])
        B[:n_z, :] = fitcoef[:, n_z*q : n_z*q + n_u]
        
        # For control history: u[t] enters as new u[t-1]
        if q_u > 1:
            B[n_z*q : n_z*q + n_u, :] = np.eye(n_u)
        
        B_matrices[t, :, :] = B
    
    # Fill in early timesteps (t < start_idx) with identity/zero
    for t in range(start_idx):
        A_matrices[t, :, :] = np.eye(aug_dim)
        B_matrices[t, :, :] = 0.0
    
    # Combine A and B for convenience
    AB_combined = np.zeros((N, aug_dim, aug_dim + n_u))
    AB_combined[:, :, :aug_dim] = A_matrices
    AB_combined[:, :, aug_dim:] = B_matrices
    
    if verbose:
        print(f"\n✓ System ID complete!")
        print(f"\nOutput shapes:")
        print(f"  alphas: {alphas.shape}")
        print(f"  betas: {betas.shape}")
        print(f"  A_matrices: {A_matrices.shape}")
        print(f"  B_matrices: {B_matrices.shape}")
        print(f"  AB_combined: {AB_combined.shape}")

    # print('alphas = ', alphas)
    # print('betas = ', betas)
    
    return alphas, betas, A_matrices, B_matrices, AB_combined


def analyze_system_id_quality(A_matrices, B_matrices, start_idx=0):
    """
    Analyze quality of identified system matrices.
    
    Returns:
        dict with conditioning, spectral radius, stability stats
    """
    N, aug_dim, _ = A_matrices.shape
    
    cond_list = []
    spec_rad_list = []
    norm_B_list = []
    
    for t in range(start_idx, N):
        A_t = A_matrices[t, :, :]
        B_t = B_matrices[t, :, :]
        
        # Conditioning
        cond_A = np.linalg.cond(A_t)
        if np.isfinite(cond_A):
            cond_list.append(cond_A)
        
        # Spectral radius
        try:
            eigs = np.linalg.eigvals(A_t)
            spec_rad = np.max(np.abs(eigs))
            if np.isfinite(spec_rad):
                spec_rad_list.append(spec_rad)
        except:
            pass
        
        # Control influence
        norm_B_list.append(np.linalg.norm(B_t))
    
    cond_arr = np.array(cond_list)
    spec_rad_arr = np.array(spec_rad_list)
    norm_B_arr = np.array(norm_B_list)
    
    # Handle empty arrays
    if len(cond_arr) == 0:
        cond_arr = np.array([np.inf])
    if len(spec_rad_arr) == 0:
        spec_rad_arr = np.array([np.inf])
    
    # Compute statistics
    stats = {
        'mean_cond': np.mean(cond_arr),
        'median_cond': np.median(cond_arr),
        'max_cond': np.max(cond_arr),
        'min_cond': np.min(cond_arr),
        
        'mean_spec_rad': np.mean(spec_rad_arr),
        'median_spec_rad': np.median(spec_rad_arr),
        'max_spec_rad': np.max(spec_rad_arr),
        'min_spec_rad': np.min(spec_rad_arr),
        
        'mean_norm_B': np.mean(norm_B_arr),
        'max_norm_B': np.max(norm_B_arr),
        
        'n_stable': np.sum(spec_rad_arr < 1.0),
        'n_total': len(spec_rad_arr),
        'frac_stable': np.sum(spec_rad_arr < 1.0) / len(spec_rad_arr),
        
        'n_wellcond': np.sum(cond_arr < 1000),
        'frac_wellcond': np.sum(cond_arr < 1000) / len(cond_arr),
    }
    
    return stats


def print_system_id_analysis(stats, q, q_u):
    """Pretty print analysis results."""
    
    print(f"\n{'='*80}")
    print(f"SYSTEM ID QUALITY ANALYSIS (q={q}, q_u={q_u})")
    print("="*80)
    
    print(f"\nA Matrix Conditioning:")
    print(f"  Mean:   {stats['mean_cond']:.2e}")
    print(f"  Median: {stats['median_cond']:.2e}")
    print(f"  Min:    {stats['min_cond']:.2e}")
    print(f"  Max:    {stats['max_cond']:.2e}")
    print(f"  Well-conditioned (<1000): {stats['frac_wellcond']*100:.1f}%")
    
    print(f"\nSpectral Radius (Stability):")
    print(f"  Mean:   {stats['mean_spec_rad']:.4f}")
    print(f"  Median: {stats['median_spec_rad']:.4f}")
    print(f"  Min:    {stats['min_spec_rad']:.4f}")
    print(f"  Max:    {stats['max_spec_rad']:.4f}")
    print(f"  Stable (ρ<1.0): {stats['frac_stable']*100:.1f}% ({stats['n_stable']}/{stats['n_total']})")
    
    print(f"\nControl Influence:")
    print(f"  Mean ||B||: {stats['mean_norm_B']:.6f}")
    print(f"  Max ||B||:  {stats['max_norm_B']:.6f}")
    
    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print("="*80)
    
    if stats['frac_stable'] > 0.95 and stats['frac_wellcond'] > 0.95:
        print(f"✅ EXCELLENT - Ready for iLQR with R=5-10")
    elif stats['frac_stable'] > 0.8 and stats['frac_wellcond'] > 0.8:
        print(f"✓ GOOD - Usable for iLQR with R=10-20")
    elif stats['frac_stable'] > 0.5 and stats['frac_wellcond'] > 0.5:
        print(f"⚠️ MARGINAL - Try iLQR with high R=50-100 or reduce q")
    else:
        print(f"❌ POOR - Reduce q or switch to simpler model")


def plot_system_id_quality(A_matrices, B_matrices, alphas, betas, 
                            start_idx, q, q_u, save_path=None):
    """Create comprehensive visualization of system ID quality."""
    
    N = A_matrices.shape[0]
    t_range = np.arange(start_idx, N)
    
    # Compute metrics
    cond_list = []
    spec_rad_list = []
    norm_B_list = []
    alpha_norms = []
    beta_norms = []
    
    for t in range(start_idx, N):
        A_t = A_matrices[t, :, :]
        B_t = B_matrices[t, :, :]
        
        cond_list.append(np.linalg.cond(A_t))
        
        try:
            eigs = np.linalg.eigvals(A_t)
            spec_rad_list.append(np.max(np.abs(eigs)))
        except:
            spec_rad_list.append(np.nan)
        
        norm_B_list.append(np.linalg.norm(B_t))
        alpha_norms.append(np.linalg.norm(alphas[t, :, :]))
        beta_norms.append(np.linalg.norm(betas[t, :, :]))
    
    cond_arr = np.array(cond_list)
    spec_rad_arr = np.array(spec_rad_list)
    norm_B_arr = np.array(norm_B_list)
    alpha_norms = np.array(alpha_norms)
    beta_norms = np.array(beta_norms)
    
    # Create plots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Conditioning over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogy(t_range, cond_arr, 'b-', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=100, color='orange', linestyle='--', linewidth=2, label='Good')
    ax1.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Bad')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Condition Number')
    ax1.set_title(f'A Matrix Conditioning (q={q}, q_u={q_u})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Spectral radius
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(t_range, spec_rad_arr, 'g-', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Stable threshold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Spectral Radius')
    ax2.set_title('Stability Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, min(3, np.nanmax(spec_rad_arr) * 1.1)])
    
    # Plot 3: Control influence
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(t_range, norm_B_arr, 'purple', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('||B||')
    ax3.set_title('Control Influence Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Alpha/Beta norms
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.plot(t_range, alpha_norms, 'b-', linewidth=1, alpha=0.7, label='||α||')
    ax4.plot(t_range, beta_norms, 'r-', linewidth=1, alpha=0.7, label='||β||')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Coefficient Norm')
    ax4.set_title('Dynamics Coefficients')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Spectral radius histogram
    ax5 = fig.add_subplot(gs[1, 2])
    finite_spec = spec_rad_arr[np.isfinite(spec_rad_arr)]
    ax5.hist(finite_spec, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Stable')
    ax5.set_xlabel('Spectral Radius')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Stability Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Conditioning histogram
    ax6 = fig.add_subplot(gs[2, 2])
    finite_cond = cond_arr[np.isfinite(cond_arr)]
    log_cond = np.log10(finite_cond[finite_cond > 0])
    ax6.hist(log_cond, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax6.axvline(x=2, color='orange', linestyle='--', linewidth=2, label='Good (10²)')
    ax6.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Bad (10³)')
    ax6.set_xlabel('log₁₀(Condition Number)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Conditioning Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'System ID Quality: q={q}, q_u={q_u}', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    return fig


def validate_prediction(X_pertb, U_pertb, Z_nom, U_nom, 
                       alphas, betas, A_matrices, B_matrices, 
                       q, q_u, rollout_idx=None, verbose=True):
    """
    Validate system ID by predicting a rollout trajectory and comparing to ground truth.
    
    Computes BOTH recursive (multi-step) and one-step predictions.
    
    Args:
        X_pertb, U_pertb, Z_nom, U_nom: rollout data
        alphas, betas, A_matrices, B_matrices: identified system matrices
        q, q_u: history lengths
        rollout_idx: which rollout to test (None = random)
        verbose: print detailed output
    
    Returns:
        dict with prediction errors and statistics
    """
    
    N_plus_1, n_z, n_samples = X_pertb.shape
    N = N_plus_1 - 1
    n_u = U_pertb.shape[1]
    aug_dim = n_z * q + n_u * max(0, q_u - 1)
    
    # Select random rollout if not specified
    if rollout_idx is None:
        rollout_idx = np.random.randint(0, n_samples)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"VALIDATION: Predicting Rollout {rollout_idx}")
        print("="*80)
        print(f"\nUsing q={q}, q_u={q_u}")
    
    # Get ground truth for this rollout
    z_actual = X_pertb[:, :, rollout_idx]  # (N+1, n_z)
    u_actual = U_pertb[:, :, rollout_idx]  # (N, n_u)
    
    # Compute deviations
    delta_z_actual = z_actual - Z_nom[:, :, 0]
    delta_u_actual = u_actual
    
    if verbose:
        print(f"\nGround truth ranges:")
        print(f"  δz: [{delta_z_actual.min():.4f}, {delta_z_actual.max():.4f}]")
        print(f"  δu: [{delta_u_actual.min():.4f}, {delta_u_actual.max():.4f}]")
    
    # Initialize prediction starting point
    start_idx = max(q, q_u)
    
    # ========================================================================
    # RECURSIVE (Multi-Step) Prediction
    # ========================================================================
    if verbose:
        print(f"\n{'='*40}")
        print("RECURSIVE (Multi-Step) Prediction")
        print("="*40)
        print(f"Predicting from t={start_idx} to t={N}...")
    
    # Storage for predicted observations
    z_pred_recursive = np.zeros((N+1, n_z))
    
    # Initialize history buffers with actual initial conditions
    z_history = []
    u_history = []
    
    for i in range(q):
        t_hist = start_idx - i
        if t_hist >= 0:
            z_history.append(delta_z_actual[t_hist, :])
    
    for i in range(q_u):
        t_hist = start_idx - i - 1
        if t_hist >= 0:
            u_history.append(delta_u_actual[t_hist, :])
    
    # Store initial predictions
    for i in range(start_idx + 1):
        z_pred_recursive[i, :] = delta_z_actual[i, :]
    
    # Recursively predict forward
    for t in range(start_idx, N):
        # Build regressor from PREDICTED history
        z_regressor = np.concatenate(z_history[:q])
        u_regressor = np.concatenate(u_history[:q_u])
        
        # Get dynamics matrices
        alpha_t = alphas[t, :, :]
        beta_t = betas[t, :, :]
        
        # Predict: z[t+1] = α @ [z history] + β @ [u history]
        z_pred_recursive[t+1, :] = alpha_t @ z_regressor + (beta_t @ u_regressor).flatten()
        
        # Update histories for next iteration
        z_history = [z_pred_recursive[t+1, :]] + z_history[:q-1]
        u_history = [delta_u_actual[t, :]] + u_history[:q_u-1]
    
    # ========================================================================
    # ONE-STEP Prediction (Using Actual History)
    # ========================================================================
    if verbose:
        print(f"\n{'='*40}")
        print("ONE-STEP Prediction (Actual History)")
        print("="*40)
    
    z_pred_onestep = np.zeros((N+1, n_z))
    
    # Store initial
    for i in range(start_idx + 1):
        z_pred_onestep[i, :] = delta_z_actual[i, :]
    
    # One-step predictions using actual history
    for t in range(start_idx, N):
        # Build regressor from ACTUAL history
        z_regressor = []
        for i in range(q):
            z_regressor.append(delta_z_actual[t - i, :])
        z_regressor = np.concatenate(z_regressor)
        
        u_regressor = []
        for i in range(q_u):
            u_regressor.append(delta_u_actual[t - i - 1, :])
        u_regressor = np.concatenate(u_regressor)
        
        # One-step prediction
        alpha_t = alphas[t, :, :]
        beta_t = betas[t, :, :]
        
        z_pred_onestep[t+1, :] = alpha_t @ z_regressor + (beta_t @ u_regressor).flatten()
    
    # ========================================================================
    # Compute Errors
    # ========================================================================
    
    # Recursive errors
    pred_error_recursive = np.linalg.norm(z_pred_recursive[start_idx:, :] - delta_z_actual[start_idx:, :], axis=1)
    relative_error_recursive = pred_error_recursive / (np.linalg.norm(delta_z_actual[start_idx:, :], axis=1) + 1e-10)
    
    # One-step errors
    pred_error_onestep = np.linalg.norm(z_pred_onestep[start_idx+1:, :] - delta_z_actual[start_idx+1:, :], axis=1)
    relative_error_onestep = pred_error_onestep / (np.linalg.norm(delta_z_actual[start_idx+1:, :], axis=1) + 1e-10)
    
    # Statistics
    stats = {
        'rollout_idx': rollout_idx,
        'start_idx': start_idx,
        
        # Recursive (multi-step) prediction
        'mean_pred_error_recursive': np.mean(pred_error_recursive),
        'max_pred_error_recursive': np.max(pred_error_recursive),
        'final_pred_error_recursive': pred_error_recursive[-1],
        'mean_relative_error_recursive': np.mean(relative_error_recursive),
        
        # One-step prediction
        'mean_pred_error_onestep': np.mean(pred_error_onestep),
        'max_pred_error_onestep': np.max(pred_error_onestep),
        'mean_relative_error_onestep': np.mean(relative_error_onestep),
        
        # Raw data for plotting
        'pred_error_recursive': pred_error_recursive,
        'pred_error_onestep': pred_error_onestep,
        'relative_error_recursive': relative_error_recursive,
        'relative_error_onestep': relative_error_onestep,
        'z_pred_recursive': z_pred_recursive,
        'z_pred_onestep': z_pred_onestep,
        'z_actual': delta_z_actual,
        't_range': np.arange(start_idx, N+1),
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("PREDICTION QUALITY COMPARISON")
        print("="*80)
        
        print(f"\nRECURSIVE (Multi-Step) Prediction:")
        print(f"  Mean error: {stats['mean_pred_error_recursive']:.6f}")
        print(f"  Max error: {stats['max_pred_error_recursive']:.6f}")
        print(f"  Final error: {stats['final_pred_error_recursive']:.6f}")
        print(f"  Mean relative error: {stats['mean_relative_error_recursive']*100:.2f}%")
        
        print(f"\nONE-STEP Prediction (Actual History):")
        print(f"  Mean error: {stats['mean_pred_error_onestep']:.6f}")
        print(f"  Max error: {stats['max_pred_error_onestep']:.6f}")
        print(f"  Mean relative error: {stats['mean_relative_error_onestep']*100:.2f}%")
        
        print(f"\nIMPROVEMENT (Recursive → One-Step):")
        print(f"  Mean error: {stats['mean_pred_error_recursive'] / stats['mean_pred_error_onestep']:.2f}×")
        print(f"  Max error: {stats['max_pred_error_recursive'] / stats['max_pred_error_onestep']:.2f}×")
    
    return stats


def plot_validation(stats, q, q_u, save_path=None):
    """Plot validation results comparing recursive and one-step predictions."""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    t_range = stats['t_range']
    z_pred_recursive = stats['z_pred_recursive'][stats['start_idx']:, :]
    z_pred_onestep = stats['z_pred_onestep'][stats['start_idx']:, :]
    z_actual = stats['z_actual'][stats['start_idx']:, :]
    
    n_z = z_actual.shape[1]
    
    # Plot 1: Prediction error comparison
    ax = axes[0, 0]
    ax.semilogy(t_range, stats['pred_error_recursive'], 'b-', linewidth=2, label='Recursive', alpha=0.7)
    ax.semilogy(t_range[:-1], stats['pred_error_onestep'], 'g-', linewidth=2, label='One-step', alpha=0.7)
    ax.axhline(y=0.1, color='orange', linestyle='--', label='Acceptable')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error: Recursive vs One-Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Relative error comparison
    ax = axes[0, 1]
    ax.plot(t_range, stats['relative_error_recursive'] * 100, 'b-', linewidth=2, label='Recursive', alpha=0.7)
    ax.plot(t_range[:-1], stats['relative_error_onestep'] * 100, 'g-', linewidth=2, label='One-step', alpha=0.7)
    ax.axhline(y=10, color='orange', linestyle='--', label='10%')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Relative Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: State predictions - Recursive
    ax = axes[1, 0]
    for i in range(min(2, n_z)):
        ax.plot(t_range, z_actual[:, i], '-', linewidth=2, label=f'Actual δz[{i}]')
        ax.plot(t_range, z_pred_recursive[:, i], '--', linewidth=1.5, alpha=0.7, label=f'Recursive δz[{i}]')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Deviation')
    ax.set_title('Recursive Prediction vs Actual')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: State predictions - One-Step
    ax = axes[1, 1]
    for i in range(min(2, n_z)):
        ax.plot(t_range, z_actual[:, i], '-', linewidth=2, label=f'Actual δz[{i}]')
        ax.plot(t_range, z_pred_onestep[:, i], '--', linewidth=1.5, alpha=0.7, label=f'One-step δz[{i}]')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Deviation')
    ax.set_title('One-Step Prediction vs Actual')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Scatter - prediction quality
    ax = axes[2, 0]
    ax.scatter(z_actual.flatten(), z_pred_recursive.flatten(), alpha=0.3, s=1, c='blue', label='Recursive')
    ax.scatter(z_actual.flatten(), z_pred_onestep.flatten(), alpha=0.3, s=1, c='green', label='One-step')
    lim = max(abs(z_actual.min()), abs(z_actual.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('Actual δz')
    ax.set_ylabel('Predicted δz')
    ax.set_title('Prediction Quality Scatter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 6: Summary statistics
    ax = axes[2, 1]
    ax.axis('off')
    
    summary_text = f"""
    VALIDATION SUMMARY
    Rollout: {stats['rollout_idx']}
    q={q}, q_u={q_u}
    
    Recursive (Multi-Step):
      Mean error: {stats['mean_pred_error_recursive']:.4f}
      Max error: {stats['max_pred_error_recursive']:.4f}
      Final error: {stats['final_pred_error_recursive']:.4f}
      Mean relative: {stats['mean_relative_error_recursive']*100:.1f}%
    
    One-Step (Actual History):
      Mean error: {stats['mean_pred_error_onestep']:.4f}
      Max error: {stats['max_pred_error_onestep']:.4f}
      Mean relative: {stats['mean_relative_error_onestep']*100:.1f}%
    
    Improvement: {stats['mean_pred_error_recursive']/stats['mean_pred_error_onestep']:.1f}×
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    plt.suptitle(f'System ID Validation: q={q}, q_u={q_u}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved validation plot to {save_path}")
    
    return fig


def main():
    """
    Main function: Load data, run system ID, analyze and visualize results.
    """
    
    # Load rollout data
    print("Loading rollout data...")
    X_pertb = np.load('X_pertb_rollouts.npy')[1:,:,:]
    U_pertb = np.load('U_pertb_rollouts.npy')[:,:,:]
    Z_nom = np.load('Z_reference.npy').reshape((X_pertb.shape[0],X_pertb.shape[1], 1))
    U_nom = np.load('U_temp.npy')


    # plt.plot(X_pertb[:,0,:])
    plt.plot(X_pertb[:,1,:])
    plt.show()
    # k = 10
    # p1 = 0.5*(X_pertb[:,0,k]+X_pertb[:,1,k])
    # p2 = X_pertb[:,1,k]-p1
    # plt.plot(p2-p1)
    # plt.plot(0.5*(p2+p1))
    # plt.figure()
    # plt.plot(p1)
    # plt.plot(p2)
    # plt.show()
    # sys.exit()
    
    print(f"✓ Loaded data:")
    print(f"  X_pertb: {X_pertb.shape}")
    print(f"  U_pertb: {U_pertb.shape}")
    print(f"  Z_nom: {Z_nom.shape}")
    print(f"  U_nom: {U_nom.shape}")
    
    # Test multiple q values
    results = {}
    
    for q in [3]:
        q_u = q
        
        print(f"\n{'='*80}")
        print(f"TESTING q={q}, q_u={q_u}")
        print("="*80)
        
        # Run system ID
        alphas, betas, A_matrices, B_matrices, AB_combined = arma_system_id(
            X_pertb, U_pertb, Z_nom, U_nom, q, q_u, verbose=True
        )
        
        # Analyze quality
        start_idx = max(q, q_u)
        stats = analyze_system_id_quality(A_matrices, B_matrices, start_idx)
        print_system_id_analysis(stats, q, q_u)
        
        # Save results
        results[q] = {
            'alphas': alphas,
            'betas': betas,
            'A_matrices': A_matrices,
            'B_matrices': B_matrices,
            'AB_combined': AB_combined,
            'stats': stats
        }
        
        # Save matrices to file
        output_dir = Path('.')
        np.save(output_dir / f'AB_combined_q{q}.npy', AB_combined)
        np.save(output_dir / f'alphas_q{q}.npy', alphas)
        np.save(output_dir / f'betas_q{q}.npy', betas)
        print(f"\n✓ Saved matrices to outputs/AB_combined_q{q}.npy")
        
        # Create visualization
        fig = plot_system_id_quality(
            A_matrices, B_matrices, alphas, betas,
            start_idx, q, q_u,
            save_path=output_dir / f'sysid_quality_q{q}.png'
        )
        plt.close(fig)
        
        # VALIDATION: Test prediction quality on specific rollout
        print(f"\n{'='*80}")
        print(f"VALIDATION: Testing prediction quality")
        print("="*80)
        X_pertb = np.load('X_pertb_rollouts.npy')[1:,:,:]
        U_pertb = np.load('U_pertb_rollouts.npy')
        val_stats = validate_prediction(
            X_pertb, U_pertb, Z_nom, U_nom,
            alphas, betas, A_matrices, B_matrices,
            q, q_u, rollout_idx=18, verbose=True
        )
        
        # Plot validation
        val_fig = plot_validation(
            val_stats, q, q_u,
            save_path=output_dir / f'validation_q{q}.png'
        )
        plt.close(val_fig)

        plt.figure()
        plt.plot(X_pertb[:,0,28])
        plt.plot(X_pertb[:,1,28])
        plt.show()
        
        # Store validation stats
        results[q]['validation'] = val_stats
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\n{'q':>3} | {'Stable %':>10} | {'Wellcond %':>12} | {'Med ρ':>8} | {'Med cond':>12} | {'Verdict':>15}")
    print("-" * 80)
    
    for q in sorted(results.keys()):
        stats = results[q]['stats']
        verdict = "✅" if stats['frac_stable'] > 0.95 else "⚠️" if stats['frac_stable'] > 0.5 else "❌"
        print(f"{q:3d} | {stats['frac_stable']*100:>9.1f}% | {stats['frac_wellcond']*100:>11.1f}% | "
              f"{stats['median_spec_rad']:>8.4f} | {stats['median_cond']:>12.2e} | {verdict:>15}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print("="*80)
    
    # Find best q
    best_q = None
    best_score = -1
    
    for q, res in results.items():
        stats = res['stats']
        # Score based on stability and conditioning
        score = stats['frac_stable'] * stats['frac_wellcond']
        if score > best_score:
            best_score = score
            best_q = q
    
    print(f"\nBest q value: {best_q}")
    print(f"  - {results[best_q]['stats']['frac_stable']*100:.1f}% stable")
    print(f"  - {results[best_q]['stats']['frac_wellcond']*100:.1f}% well-conditioned")
    print(f"  - Median spectral radius: {results[best_q]['stats']['median_spec_rad']:.4f}")
    print(f"\nUse: AB_combined_q{best_q}.npy for your iLQR implementation")


if __name__ == "__main__":
    main()