"""
System ID Quality Test
======================

Test if the linearized model (A, B matrices) can accurately predict
the nominal trajectory when given the same initial condition and controls.

Process:
1. Take nominal trajectory (Z_nom, U_nom)
2. Do system ID to get A, B matrices
3. Use A, B to predict forward from Z_nom[0] with U_nom
4. Compare predicted trajectory vs actual nominal trajectory
5. Check prediction error at each timestep
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project paths
project_root = Path.home() / "PartiallyObserved_RL"
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "examples" / "flow_control_pyfr"))

from arma_ltv_sys_id import ARMA_LTV_SysID
from sim_flow import SimulatePyFR

def test_system_id_quality():
    """
    Test how well system ID can reconstruct the nominal trajectory.
    """
    
    print("="*80)
    print("SYSTEM ID QUALITY TEST")
    print("="*80)
    
    # Parameters (should match your iLQR config)
    n_z = 2  # observation dimension
    n_u = 1  # control dimension
    q = 7    # ARMA history for observations
    q_u = 7  # ARMA history for controls
    dt = 0.5
    horizon = 300
    n_rollouts = 80  # Reduced for testing - increase to 80 for real run
    
    n_aug = n_z * q + n_u * (q_u - 1)  # Augmented state dimension
    
    print(f"\nParameters:")
    print(f"  State dim (n_z): {n_z}")
    print(f"  Control dim (n_u): {n_u}")
    print(f"  ARMA order (q, q_u): ({q}, {q_u})")
    print(f"  Augmented state dim: {n_aug}")
    print(f"  Horizon: {horizon}")
    print(f"  dt: {dt}")
    print(f"  Rollouts for sys ID: {n_rollouts}")
    
    # Setup paths
    pyfr_dir = Path.home() / "PyFR-Flow-Control" / "2d-inc-cylinder-base"
    config_path = pyfr_dir / 'inc-cylinder.ini'
    mesh_path = pyfr_dir / 'inc-cylinder.pyfrm'
    
    # Find initial solution
    pyfrs_files = sorted(pyfr_dir.glob('inc-cylinder-re60*.pyfrs'))
    if not pyfrs_files:
        print("ERROR: No initial solution found!")
        return
    
    init_solution = pyfrs_files[-1]
    print(f"\nInitial solution: {init_solution.name}")
    
    # Create simulator
    print("\n" + "="*80)
    print("STEP 1: Generate Nominal Trajectory")
    print("="*80)
    
    sim = SimulatePyFR(
        nx=n_z,
        nu=n_u,
        dt=dt,
        config_path=str(config_path),
        mesh_path=str(mesh_path),
        backend='cuda',
        mode='rollout'
    )
    
    # Generate nominal control (zero or small random)
    # NOTE: PyFR uses antisymmetric jets, so nu=1 but actual control is 2D
    U_nom = 5*np.random.normal(0, 1.0, (horizon, n_u))
    print(f"\nNominal control U_nom:")
    print(f"  Shape: {U_nom.shape}")
    print(f"  Range: [{U_nom.min():.3f}, {U_nom.max():.3f}]")
    print(f"  Mean: {U_nom.mean():.3f}")
    print(f"  Note: Will be converted to antisymmetric [u, -u] inside simulator")
    
    # Run nominal trajectory
    print("\nRunning nominal trajectory...")
    Z_nom = sim.simulate_trajectory(
        y_init=init_solution,
        u=U_nom,
        horizon=horizon
    )
    
    print(f"\nNominal trajectory Z_nom:")
    print(f"  Shape: {Z_nom.shape}")
    print(f"  Range: [{Z_nom.min():.6f}, {Z_nom.max():.6f}]")
    print(f"  Mean: {Z_nom.mean():.6f}")
    
    # Build augmented state from nominal trajectory
    # This matches the implementation in forward_pass_simulate
    Z_aug_nom = np.zeros((horizon, n_aug, 1))
    Z_nom_3d = Z_nom[:-1].reshape(horizon, n_z, 1)  # Convert to 3D
    
    for t in range(horizon):
        # Build observation history: [z[t-q+1], ..., z[t-1], z[t]]
        # If not enough history, pad with first observation
        if t < q - 1:
            # Pad with Z[0]
            z_pad = np.tile(Z_nom_3d[0], (q - t - 1, 1, 1))
            z_actual = Z_nom_3d[0:t+1]
            z_history = np.vstack([z_pad, z_actual])
        else:
            # Full history available
            z_history = Z_nom_3d[t - q + 1:t + 1]
        
        z_flat = z_history.reshape(q * n_z, 1)
        
        # Build control history: [u[t-q_u+1], ..., u[t-1]]
        # Note: does NOT include u[t]
        if q_u > 1:
            if t < q_u - 1:
                if t == 0:
                    u_history = np.zeros((q_u - 1, n_u, 1))
                else:
                    u_pad = np.zeros((q_u - 1 - t, n_u, 1))
                    u_actual = U_nom[0:t].reshape(t, n_u, 1)
                    u_history = np.vstack([u_pad, u_actual])
            else:
                u_history = U_nom[t - q_u + 1:t].reshape(q_u - 1, n_u, 1)
            
            u_flat = u_history.reshape(n_u * (q_u - 1), 1)
            Z_aug_nom[t, :, 0] = np.vstack([z_flat, u_flat]).flatten()
        else:
            Z_aug_nom[t, :, 0] = z_flat.flatten()
    
    print(f"\nAugmented nominal trajectory Z_aug_nom:")
    print(f"  Shape: {Z_aug_nom.shape}")
    print(f"  Range: [{Z_aug_nom.min():.6f}, {Z_aug_nom.max():.6f}]")
    
    # Setup system ID
    print("\n" + "="*80)
    print("STEP 2: System Identification")
    print("="*80)
    
    sys_id = ARMA_LTV_SysID(
        MODEL=sim,
        n_x=n_aug,
        n_u=n_u,
        n_z=n_z,
        q=q,
        q_u=q_u,
        N=horizon,
        n_samples=n_rollouts,
        pert_sigma=0.08,  # 8% perturbation
    )
    
    # Do system ID around nominal trajectory
    print(f"\nRunning {n_rollouts} perturbed rollouts for system ID...")
    print("(This will take a while...)")
    
    # IMPORTANT: traj_sys_id expects OBSERVATIONS (n_z), not augmented states (n_aug)!
    # Prepend initial observation
    Z_nom_with_init = np.concatenate([
        Z_nom[0:1, :].reshape(1, n_z, 1),  # Initial observation
        Z_nom[:-1, :].reshape(horizon, n_z, 1)  # All observations (excluding final)
    ], axis=0)
    
    # Reshape U_nom to correct format
    U_nom_3d = U_nom.reshape(horizon, n_u, 1)
    
    print(f"\nSystem ID input shapes:")
    print(f"  Z_nom_with_init: {Z_nom_with_init.shape} (should be ({horizon+1}, {n_z}, 1))")
    print(f"  U_nom_3d: {U_nom_3d.shape} (should be ({horizon}, {n_u}, 1))")
    np.save("Z_nom_with_init.npy", Z_nom_with_init)
    np.save("U_nom_3d.npy", U_nom_3d)
    
    Fx_Fu = sys_id.traj_sys_id(
        Z_nom_with_init,  # Shape: (N+1, n_z, 1)
        U_nom_3d,         # Shape: (N, n_u, 1)
        roll_start=init_solution
    )
    
    
    print(f"\nSystem ID complete!")
    print(f"  Fx_Fu shape: {Fx_Fu.shape}")
    print(f"  Expected: ({horizon}, {n_aug + n_u})")
    
    # Extract A and B matrices
    print("\n" + "="*80)
    print("STEP 3: Analyze System ID Quality")
    print("="*80)
    
    print("\nA and B matrix diagnostics:")
    print(f"{'Time':>6} {'cond(A)':>12} {'||B||':>12} {'max|B|':>12}")
    print("-" * 48)
    
    for t in range(min(10, horizon)):
        F_x = Fx_Fu[t][:, :n_aug]
        F_u = Fx_Fu[t][:, n_aug:]
        
        cond_A = np.linalg.cond(F_x)
        norm_B = np.linalg.norm(F_u)
        max_B = np.max(np.abs(F_u))
        
        print(f"{t:6d} {cond_A:12.2e} {norm_B:12.4f} {max_B:12.4f}")
    
    # Check for ill-conditioning
    A_conds = []
    for t in range(horizon):
        F_x = Fx_Fu[t][:, :n_aug]
        A_conds.append(np.linalg.cond(F_x))
    
    print(f"\nA matrix conditioning:")
    print(f"  Min cond(A): {np.min(A_conds):.2e}")
    print(f"  Max cond(A): {np.max(A_conds):.2e}")
    print(f"  Mean cond(A): {np.mean(A_conds):.2e}")
    
    if np.max(A_conds) > 1000:
        print("  ⚠️  WARNING: A matrices are ILL-CONDITIONED!")
        print("  This will cause unstable feedback gains!")
    else:
        print("  ✓ A matrices are well-conditioned")
    
    # Predict trajectory using linearized model
    print("\n" + "="*80)
    print("STEP 4: Predict Trajectory Using Linearized Deviation Model")
    print("="*80)
    
    print("\nLTV Model: δz[t+1] = A[t] @ δz[t] + B[t] @ δu[t]")
    print("Where δz = z - z_nom, δu = u - u_nom")
    
    # Create small control perturbation
    delta_u = 0.3 * np.random.randn(horizon, n_u)  # Small random perturbation
    u_perturbed = U_nom + delta_u
    
    print(f"\nControl perturbation δu:")
    print(f"  Range: [{delta_u.min():.6f}, {delta_u.max():.6f}]")
    print(f"  Std: {delta_u.std():.6f}")
    
    # Predict deviation trajectory using LTV model
    print("\nPredicting deviation trajectory using A, B matrices...")
    delta_Z_pred = np.zeros((horizon + 1, n_aug))
    delta_Z_pred[0, :] = 0.0  # Start from nominal (zero deviation)
    
    for t in range(horizon):
        F_x = Fx_Fu[t][:, :n_aug]
        F_u = Fx_Fu[t][:, n_aug:]
        
        # LTV prediction: δz[t+1] = A @ δz[t] + B @ δu[t]
        delta_Z_pred[t + 1, :] = F_x @ delta_Z_pred[t, :] + (F_u @ delta_u[t, :]).flatten()
    
    # Compute predicted absolute trajectory
    Z_pred_ltv = np.zeros((horizon + 1, n_aug))
    Z_pred_ltv[0, :] = Z_aug_nom[0, :, 0]
    for t in range(horizon):
        Z_pred_ltv[t + 1, :] = Z_aug_nom[t, :, 0] + delta_Z_pred[t + 1, :]
    
    print("LTV prediction complete!")
    
    # Now simulate actual perturbed trajectory
    print("\n" + "="*80)
    print("STEP 5: Simulate Actual Perturbed Trajectory")
    print("="*80)
    
    print("\nRunning simulator with perturbed control u_nom + δu...")
    Z_actual_perturbed = sim.simulate_trajectory(
        y_init=init_solution,
        u=u_perturbed,
        horizon=horizon
    )
    
    print(f"Actual perturbed trajectory shape: {Z_actual_perturbed.shape}")
    
    # Build augmented states for actual perturbed trajectory
    Z_aug_actual = np.zeros((horizon, n_aug, 1))
    Z_actual_3d = Z_actual_perturbed[:-1].reshape(horizon, n_z, 1)
    
    for t in range(horizon):
        # Build observation history
        if t < q - 1:
            z_pad = np.tile(Z_actual_3d[0], (q - t - 1, 1, 1))
            z_actual = Z_actual_3d[0:t+1]
            z_history = np.vstack([z_pad, z_actual])
        else:
            z_history = Z_actual_3d[t - q + 1:t + 1]
        
        z_flat = z_history.reshape(q * n_z, 1)
        
        # Build control history (using perturbed control)
        if q_u > 1:
            if t < q_u - 1:
                if t == 0:
                    u_history = np.zeros((q_u - 1, n_u, 1))
                else:
                    u_pad = np.zeros((q_u - 1 - t, n_u, 1))
                    u_actual = u_perturbed[0:t].reshape(t, n_u, 1)
                    u_history = np.vstack([u_pad, u_actual])
            else:
                u_history = u_perturbed[t - q_u + 1:t].reshape(q_u - 1, n_u, 1)
            
            u_flat = u_history.reshape(n_u * (q_u - 1), 1)
            Z_aug_actual[t, :, 0] = np.vstack([z_flat, u_flat]).flatten()
        else:
            Z_aug_actual[t, :, 0] = z_flat.flatten()
    
    print("Actual augmented trajectory built!")
    
    # Compare predicted vs actual
    print("\n" + "="*80)
    print("STEP 6: Compare LTV Prediction vs Actual Perturbed Trajectory")
    print("="*80)
    
    # Compare: Z_pred_ltv[1:] vs Z_aug_actual[:, :, 0]
    Z_actual_compare = Z_aug_actual[:, :, 0]  # Shape: (horizon, n_aug)
    Z_pred_compare = Z_pred_ltv[1:, :]  # Shape: (horizon, n_aug)
    
    # Compute errors
    errors = np.linalg.norm(Z_pred_compare - Z_actual_compare, axis=1)
    relative_errors = errors / (np.linalg.norm(Z_actual_compare, axis=1) + 1e-10)
    
    # Also compute deviation errors (how well we predict δz)
    delta_Z_actual = Z_actual_compare - Z_aug_nom[:, :, 0]
    delta_Z_pred_compare = delta_Z_pred[1:, :]
    deviation_errors = np.linalg.norm(delta_Z_pred_compare - delta_Z_actual, axis=1)
    
    print("\nPrediction Error Statistics (Absolute Trajectory):")
    print(f"  First timestep error: {errors[0]:.6f}")
    print(f"  Max error: {np.max(errors):.6f}")
    print(f"  Mean error: {np.mean(errors):.6f}")
    print(f"  Final error: {errors[-1]:.6f}")
    print(f"\n  Mean relative error: {np.mean(relative_errors):.4%}")
    print(f"  Max relative error: {np.max(relative_errors):.4%}")
    
    print("\nPrediction Error Statistics (Deviations δz):")
    print(f"  First timestep error: {deviation_errors[0]:.6f}")
    print(f"  Max deviation error: {np.max(deviation_errors):.6f}")
    print(f"  Mean deviation error: {np.mean(deviation_errors):.6f}")
    print(f"  Final deviation error: {deviation_errors[-1]:.6f}")
    
    # Verdict based on deviation errors (this is what matters for LTV!)
    mean_dev_error = np.mean(deviation_errors)
    max_dev_error = np.max(deviation_errors)
    delta_u_scale = np.std(delta_u)
    
    # Check if deviation errors are proportional to control perturbation
    error_ratio = mean_dev_error / (delta_u_scale + 1e-10)
    
    print(f"\nDeviation error / control perturbation: {error_ratio:.4f}")
    
    if mean_dev_error < 0.01 * np.mean(np.abs(delta_Z_actual)):
        print("\n  ✓ EXCELLENT PREDICTION: δz error < 1% of deviation magnitude")
        print("  LTV linearization is very accurate!")
    elif mean_dev_error < 0.1 * np.mean(np.abs(delta_Z_actual)):
        print("\n  ⚠️  MODERATE PREDICTION: δz error 1-10% of deviation magnitude")
        print("  LTV linearization is acceptable but could be better")
    else:
        print("\n  ❌ POOR PREDICTION: δz error > 10% of deviation magnitude")
        print("  LTV linearization is not accurate!")
    
    # Plot results
    print("\n" + "="*80)
    print("STEP 7: Visualize Results")
    print("="*80)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Plot 1: Absolute trajectory prediction errors
    ax = axes[0, 0]
    ax.plot(errors, 'b-', linewidth=2, label='Absolute traj error')
    ax.plot(deviation_errors, 'r-', linewidth=2, label='Deviation (δz) error')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Prediction Error')
    ax.set_title('LTV Prediction Error vs Time')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()
    
    # Plot 2: Deviations comparison
    ax = axes[0, 1]
    # Show first few deviation components
    for i in range(min(3, n_z)):
        ax.plot(delta_Z_actual[:, i], '-', label=f'Actual δz[{i}]', linewidth=2)
        ax.plot(delta_Z_pred_compare[:, i], '--', label=f'Pred δz[{i}]', linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Deviation Value')
    ax.set_title('Deviations: Actual vs Predicted')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: First few absolute states comparison
    ax = axes[1, 0]
    for i in range(min(3, n_z)):
        ax.plot(Z_actual_compare[:, i], '-', label=f'Actual z[{i}]', linewidth=2)
        ax.plot(Z_pred_compare[:, i], '--', label=f'Pred z[{i}]', linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State Value')
    ax.set_title('Absolute States: Actual vs Predicted')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: A matrix condition numbers
    ax = axes[1, 1]
    ax.plot(A_conds, 'g-', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Condition Number')
    ax.set_title('A Matrix Conditioning')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.axhline(y=100, color='orange', linestyle='--', label='Good threshold')
    ax.axhline(y=1000, color='red', linestyle='--', label='Bad threshold')
    ax.legend()
    
    # Plot 5: Deviation scatter plot
    ax = axes[2, 0]
    ax.scatter(delta_Z_actual.flatten(), delta_Z_pred_compare.flatten(), alpha=0.3, s=1)
    lim = max(abs(delta_Z_actual.min()), abs(delta_Z_actual.max()), 
              abs(delta_Z_pred_compare.min()), abs(delta_Z_pred_compare.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Deviation δz')
    ax.set_ylabel('Predicted Deviation δz')
    ax.set_title('Deviation Prediction Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 6: Control perturbation and its effect
    ax = axes[2, 1]
    ax.plot(delta_u, 'k-', linewidth=1, alpha=0.7, label='Control pert δu')
    ax.plot(deviation_errors, 'r-', linewidth=2, label='Deviation error')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Control Perturbation vs Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path.home() / "PartiallyObserved_RL" / "results"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "sys_id_quality_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    # Cleanup
    sim.cleanup()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'mean_relative_error': np.mean(relative_errors),
        'mean_deviation_error': np.mean(deviation_errors),
        'max_deviation_error': np.max(deviation_errors),
        'max_cond_A': np.max(A_conds),
        'mean_cond_A': np.mean(A_conds),
        'control_pert_std': delta_u_scale
    }


if __name__ == "__main__":
    results = test_system_id_quality()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mean absolute traj error: {results['mean_error']:.6f}")
    print(f"Mean deviation (δz) error: {results['mean_deviation_error']:.6f}")
    print(f"Control perturbation std: {results['control_pert_std']:.6f}")
    print(f"Error/Pert ratio: {results['mean_deviation_error']/results['control_pert_std']:.4f}")
    print(f"Max cond(A): {results['max_cond_A']:.2e}")
    print(f"Mean cond(A): {results['mean_cond_A']:.2e}")
    
    # Verdict based on deviation prediction quality (what matters for LTV!)
    error_ratio = results['mean_deviation_error'] / results['control_pert_std']
    
    if error_ratio < 0.5 and results['max_cond_A'] < 100:
        print("\n✅ VERDICT: System ID is HIGH QUALITY")
        print("   LTV linearization accurately predicts deviations!")
        print("   Your iLQR should work well!")
    elif error_ratio < 2.0 and results['max_cond_A'] < 1000:
        print("\n⚠️  VERDICT: System ID is MODERATE QUALITY")
        print("   LTV captures trends but with some error")
        print("   iLQR might work with regularization (R=10+)")
    else:
        print("\n❌ VERDICT: System ID is POOR QUALITY")
        print("   LTV linearization is inaccurate!")
        print("\n   Possible fixes:")
        print("   1. Reduce augmented state dimension (smaller q, q_u)")
        print("   2. Increase number of rollouts (try 50-100)")
        print("   3. Tune perturbation std (currently using 5% of u_max)")
        print("   4. Check if system is too nonlinear for LTV approximation")