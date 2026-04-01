import numpy as np
import sys
import os
from pathlib import Path
import shutil
import re
from scipy.linalg import solve_discrete_are


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sim_flow import SimulatePyFR
from flow_params import *
from main_ilqr import iLQR
from main_pod_ilqr import POD_iLQR
from ltv_sys_id import LTV_SysID
from arma_ltv_sys_id import ARMA_LTV_SysID
import sys

class RunFlowControl(SimulatePyFR):
    """
    Flow control wrapper with NO RESET between steps
    """
    def __init__(self, state_dimension, control_dimension, dt, config_path, mesh_path, backend='cuda', init_solution=None, output_dir=None):
        SimulatePyFR.__init__(self, state_dimension, control_dimension, dt, 
                             config_path, mesh_path, backend, output_dir=output_dir, use_force_approximation=True)
        self.C = np.eye(state_dimension)
        
        # Store initial solution path (OUTSIDE work_dir)
        if init_solution is not None:
            self.init_solution_path = Path(init_solution).resolve()
        else:
            self.init_solution_path = None
        
        # Track current solution for chaining
        self.next_restart_solution = self.init_solution_path
        self._step_count = 0
        self._initialized = False
    
    def simulate_step(self, x_prev, u, restart_from=None):
        """
        Simulate one step - track actual simulation times
        """
        u = np.atleast_1d(u).flatten()
        
        # Determine restart solution
        if restart_from is not None:
            y_init = Path(restart_from)
        else:
            y_init = self.next_restart_solution
        
        if y_init is None:
            raise RuntimeError("No solution file available for restart")
        
        # Extract ACTUAL start time from the solution file
        start_time = self._extract_time_from_solution(y_init)
        
        print(f"\n  [Step {self._step_count}] Restarting from: {y_init.name} (t={start_time:.2f})")
        
        # Run simulation
        trajectory = self.simulate_trajectory(
            y_init=y_init,
            u=u.reshape(1, -1),
            horizon=1
        )
        
        self._initialized = True
        
        # Calculate expected END time
        # After 1 step of dt=0.5, we should be at start_time + 0.5
        expected_end_time = start_time + self.dt
        
        print(f"    Simulated: t={start_time:.2f} → {expected_end_time:.2f}")
        
        # Find solution file at the END time
        solution_files = [
            f for f in self.work_dir.glob('inc-cylinder*.pyfrs')
            if 'tavg' not in f.name and 'src' not in f.name
        ]
        
        if solution_files:
            best_file = None
            min_diff = float('inf')
            
            for f in solution_files:
                # Extract time from filename
                match = re.search(r'(\d+\.\d+)\.pyfrs$', f.name)
                if match:
                    file_time = float(match.group(1))
                    diff = abs(file_time - expected_end_time)
                    if diff < min_diff and diff < 0.1:  # Within 0.1s tolerance
                        min_diff = diff
                        best_file = f
            
            if best_file is not None:
                self.next_restart_solution = best_file
                print(f"    → Next: {self.next_restart_solution.name}")
            else:
                print(f"    ⚠ Warning: No file found at t={expected_end_time:.2f}")
                print(f"    Available files: {[f.name for f in solution_files]}")
                # Don't update next_restart_solution - this will cause an error and stop
                raise RuntimeError(f"No solution file found at expected time {expected_end_time}")
        
        self._step_count += 1
        
        return trajectory[-1]


if __name__=="__main__":

    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"examples/flow_control_pyfr"

    path_to_export = path_to_vdp/"Flow_Experiments/exp_re100"
    path_to_policy_file = path_to_export/"flow_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"

    # Path to PyFR simulation files
    pyfr_dir = Path.home() / "PyFR-Flow-Control" / "2d-inc-cylinder-base"
    config_path = pyfr_dir / 'inc-cylinder.ini'
    mesh_path = pyfr_dir / 'inc-cylinder.pyfrm'

    default_init_solution = pyfr_dir / 'inc-cylinder-re100-220.00.pyfrs'#'inc-cylinder-re150-260.00.pyfrs'#'inc-cylinder-re200-125.00.pyfrs'#'inc-cylinder-re60-200.00.pyfrs'#'inc-cylinder-re200-125.00.pyfrs'#'inc-cylinder-re200-75.00.pyfrs'##
    if default_init_solution.exists():
        init_solution = str(default_init_solution)
        print(f"Using initial solution: {default_init_solution.name}")
    else:
        pyfrs_files = sorted(pyfr_dir.glob('inc-cylinder*.pyfrs'))
        if pyfrs_files:
            init_solution = str(pyfrs_files[-1])
            print(f"Using fallback solution: {Path(init_solution).name}")
        else:
            init_solution = None
            print("No initial solution found")
    
    path_to_export.mkdir(parents=True, exist_ok=True)

    solutions_base = path_to_export / "solutions"
    solutions_base.mkdir(exist_ok=True)
    
    n_iterations = 7

    # Create model instance
    run_flow = RunFlowControl(
        state_dimension=state_dimension,
        control_dimension=control_dimension,
        dt=dt,
        config_path=str(config_path),
        mesh_path=str(mesh_path),
        backend='cuda',
        init_solution=init_solution,
        output_dir=str(solutions_base)
    )

    # Initialize states
    obs_init_state = np.zeros((state_dimension, 1))
    final_state = np.zeros((n_aug, 1))
    
    Q = Q_aug
    Q_final = Q_final_aug

    print('Target: suppress vortex shedding\n')

    # Create iLQR instance
    ilqr = POD_iLQR(
        run_flow, 
        state_dimension, 
        control_dimension, 
        alpha, 
        horizon, 
        obs_init_state, 
        final_state, 
        n_z, 
        q, 
        q_u, 
        Q, 
        Q_final, 
        R, 
        nominal_init_stddev, 
        n_sys_id_samples=20,  
        pert_sys_id_sigma=0.1, 
        arma_sys_id_flag=False
    )

   
    # Run iLQR
    try:
        ilqr.iterate_ilqr(n_iterations, resume_from_checkpoint=True)
        
        ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
        ilqr.save_policy(path_to_policy_file)
        ilqr.save_cost(path_to_cost_file)

        
        
        print(f'\n✓ Results saved to: {path_to_export}')
        

        # Regulator design
        from arma_lti_sys_id import ARMA_LTI_SysID, design_lqr
        suppressed_solution = solutions_base / 'inc-cylinder-ptb-245.00.pyfrs'#'inc-cylinder-re150-260.00.pyfrs'#'inc-cylinder-ptb-150.00.pyfrs'#'inc-cylinder-ptb-245.00.pyfrs'

        lti_id = ARMA_LTI_SysID(
            MODEL=run_flow,
            n_x=state_dimension,
            n_u=control_dimension,
            n_z=obs_dimension,
            q=3, q_u=3,
            rollout_horizon=50,
            n_rollouts=10,
            pert_sigma=1.0
        )


        A, B = lti_id.identify(suppressed_solution)
        # A = np.load('A_LTI.npy')
        # B = np.load('B_LTI.npy')
        K_lqr, P = design_lqr(A, B, Q_lqr, R_lqr)
        np.save('A_LTI.npy', A)
        np.save('B_LTI.npy', B)
        # np.save('K_LTI.npy', K_lqr)
        # np.save('P_LTI.npy', P)

       
        # P = np.load('P_LTI.npy')
        # K_lqr = np.load('K_LTI.npy')
        

        #np.save('K_lqr.npy', K_lqr)

        # u = -K_lqr @ z_aug  is equivalent to:
        # u = u_temp + alpha*k + K @ (z_aug - z_ref)
        # with u_temp=0, k=0, K=-K_lqr, z_ref=0, alpha=1
        
        X_eq = 0*ilqr.Z[-1]#lti_id.X_eq#
        
        
        horizon_lqr = 200  # run long enough to confirm stability
        z_ref = np.tile(np.array([X_eq[0][0], 0.0]), (horizon_lqr, 1))  # [CD_eq, 0.0]
        print('Reference CD, CL = ', z_ref)
        
        # K_lqr = np.load('feedback_K.npy')
        
        np.save('feedback_K.npy', np.tile(K_lqr, (horizon_lqr, 1, 1)))  # (H, 1, n_aug)
        np.save('feedback_k.npy', np.zeros((horizon_lqr, 1, 1)))
        np.save('U_temp.npy', np.zeros((horizon_lqr, 1, 1)))
        np.save('Z_reference.npy', z_ref)
        np.save('feedback_metadata.npy', {
            'dt': 0.5,
            't_start': 245.0,  # start time of suppressed solution
            'q': 3,
            'q_u': 3,
            'alpha': 1.0,
            'n_z': 2,
            'n_u': 1,
        })

        

        # Run
        run_flow.set_mode('feedback')
        Y = run_flow.simulate_trajectory(
            y_init=suppressed_solution,
            u=None,
            horizon=horizon_lqr
        )

        # np.save('test_del.npy', Y)
        # np.save('CDCL_LTI_re200.npy', Y)
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        run_flow.cleanup()
        print('Cleaned up')