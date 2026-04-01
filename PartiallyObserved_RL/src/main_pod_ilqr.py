import numpy as np
import math
import json
import subprocess
from main_ilqr import iLQR
from arma_ltv_sys_id import ARMA_LTV_SysID
from pathlib import Path
# np.random.seed(50)

class POD_iLQR(iLQR):

    # def __init__(self, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, n_z, q, q_u, 
    #              Q, Q_final, R, nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, arma_sys_id_flag = True):
    #     self.n_z = n_z
    #     self.q = q
    #     self.q_u = q_u
    #     self.n_aug = n_z*q+n_u*(q_u-1)
    #     self.Z_0 = init_state
    #     iLQR.__init__(self, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, 
    #                   Q, Q_final, R, nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, arma_sys_id_flag = arma_sys_id_flag)
        
    #     self.Z_aug_0 = np.zeros((self.n_aug,1))
    #     self.Z_aug_0[0:n_z,:] = self.Z_0
    #     self.Z_aug = np.zeros((self.N, self.n_aug, 1))

    #     self.Z = np.zeros((self.N, self.n_z, 1))
    #     self.Z_temp = np.zeros((self.N, self.n_z, 1))

    #     self.K = np.zeros((self.N, self.n_u, self.n_aug))
    #     self.k = np.zeros((self.N, self.n_u, 1))
		
    #     self.V_xx = np.zeros((self.N, self.n_aug, self.n_aug))
    #     self.V_x = np.zeros((self.N, self.n_aug, 1))
        
    #     self.ltv_sys_id = ARMA_LTV_SysID(self.model, self.n_x, n_u, n_z, q, q_u, self.N, n_samples=n_sys_id_samples, pert_sigma = pert_sys_id_sigma)
    def __init__(self, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, n_z, q, q_u, 
             Q, Q_final, R, nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, 
             arma_sys_id_flag=True, checkpoint_dir='checkpoints'):
        
        # ===================================================================
        # CRITICAL: Set checkpoint_dir FIRST before calling parent __init__
        # ===================================================================
        from pathlib import Path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Set POD-specific attributes
        self.n_z = n_z
        self.q = q
        self.q_u = q_u
        self.n_aug = n_z*q+n_u*(q_u-1)
        self.Z_0 = init_state
        
        # Now call parent __init__
        iLQR.__init__(self, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, 
                    Q, Q_final, R, nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, 
                    arma_sys_id_flag=arma_sys_id_flag)
        
        # Rest of initialization...
        self.Z_aug_0 = np.zeros((self.n_aug,1))
        self.Z_aug_0[0:n_z,:] = self.Z_0
        self.Z_aug = np.zeros((self.N, self.n_aug, 1))
        self.Z = np.zeros((self.N, self.n_z, 1))
        self.Z_temp = np.zeros((self.N, self.n_z, 1))
        self.Z_aug_temp = np.zeros((self.N, self.n_aug, 1))
        self.K = np.zeros((self.N, self.n_u, self.n_aug))
        self.k = np.zeros((self.N, self.n_u, 1))
        self.V_xx = np.zeros((self.N, self.n_aug, self.n_aug))
        self.V_x = np.zeros((self.N, self.n_aug, 1))
        
        self.ltv_sys_id = ARMA_LTV_SysID(self.model, self.n_x, n_u, n_z, q, q_u, self.N, 
                                        n_samples=n_sys_id_samples, pert_sigma=pert_sys_id_sigma)
        
        # Track current iteration
        self.current_iteration = 0

    def save_checkpoint(self, iteration):
        """
        Save checkpoint after successful iteration
        Saves: Z, K, k, U, U_temp, Z_temp, cost_history, alpha, iteration number
        """
        checkpoint_path = self.checkpoint_dir / f'checkpoint_iter_{iteration}.npz'

        checkpoint_data = {
            'iteration': iteration,
            'Z': self.Z,
            'K': self.K,
            'k': self.k,
            'U': self.U,
            'U_temp': self.U_temp,
            'Z_temp': self.Z_temp,
            'Z_aug': self.Z_aug,
            'cost_history': np.array(self.episodic_cost_history),
            'alpha': self.alpha,
            'V_x': self.V_x,
            'V_xx': self.V_xx,
        }
        
        np.savez_compressed(checkpoint_path, **checkpoint_data)
        
        # Also save as 'latest' for easy recovery
        latest_path = self.checkpoint_dir / 'checkpoint_latest.npz'
        np.savez_compressed(latest_path, **checkpoint_data)
        
        print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")
        
        # Clean up old checkpoints (keep last 3)
        self._cleanup_old_checkpoints(keep_last=10)
    
    def load_checkpoint(self, checkpoint_path=None):
        """
        Load checkpoint to resume training
        
        Args:
            checkpoint_path: Path to specific checkpoint, or None to load latest
        
        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        if checkpoint_path is None:
            emergency_files = sorted(
            self.checkpoint_dir.glob('checkpoint_emergency_iter_*.npz'),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
            if emergency_files:
                checkpoint_path = emergency_files[0]  # Most recent emergency
                print(f"Found emergency checkpoint: {checkpoint_path.name}")
            else:
                checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.npz'#'checkpoint_iter_1.npz'#'checkpoint_iter_-1.npz'#'checkpoint_iter_-1.npz'#
            
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return False
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        try:
            data = np.load(checkpoint_path, allow_pickle=True)
            
            self.current_iteration = int(data['iteration'])
            self.Z = data['Z']
            self.K = data['K']
            self.k = data['k']
            self.U = data['U']
            self.U_temp = data['U_temp']
            self.Z_temp = data['Z_temp']
            self.Z_aug = data['Z_aug']
            self.episodic_cost_history = list(data['cost_history'])
            self.alpha = float(data['alpha'])
            self.V_x = data['V_x']
            self.V_xx = data['V_xx']
            
            print(f"  ✓ Resumed from iteration {self.current_iteration}")
            print(f"  ✓ Cost history: {self.episodic_cost_history}")
            print(f"  ✓ Alpha: {self.alpha:.6f}")

            
            
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to load checkpoint: {e}")
            return False
    
    def _cleanup_old_checkpoints(self, keep_last=3):
        """Remove old checkpoint files, keeping only the most recent ones"""
        checkpoint_files = sorted(
            self.checkpoint_dir.glob('checkpoint_iter_*.npz'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        # Remove old checkpoints (keep last N)
        if len(checkpoint_files) > keep_last:
            for old_checkpoint in checkpoint_files[:-keep_last]:
                old_checkpoint.unlink()
                print(f"    Removed old checkpoint: {old_checkpoint.name}")
        
    
    
    def iterate_ilqr(self, n_iter, u_init=None, resume_from_checkpoint=False):
        """
        Main function that carries out the algorithm at higher level
        
        Args:
            n_iter: Total number of iLQR iterations to run
            u_init: Initial control guess (if not resuming)
            resume_from_checkpoint: If True, tries to load latest checkpoint
        
        Returns:
            True if completed successfully
        """
        
        # Try to resume from checkpoint
        start_iter = 0
        if resume_from_checkpoint:
            if self.load_checkpoint():
                start_iter = self.current_iteration + 1
                print(f"\n{'='*80}")
                print(f"RESUMING FROM ITERATION {start_iter}")
                print(f"{'='*80}\n")
            else:
                print("No checkpoint found, starting from scratch")
        
        # Initialize trajectory if starting fresh
        if start_iter == 0:
            self.initialize_traj(u_init=u_init)
            
            # Do one forward pass to populate Z_aug properly
            self.forward_pass_simulate()
            
            # Compute initial cost
            cost = self.calculate_total_cost(self.Z_aug_0, self.Z_aug, self.U, self.N)
            self.episodic_cost_history = [float(cost)]
            print(f"Initial cost: {float(cost):.4f}\n")
            
            # Save initial checkpoint (iteration -1)
            self.save_checkpoint(iteration=-1)
        
        # Start the iLQR iterations
        for i in self.pbar(range(start_iter, n_iter)):
            self.current_iteration = i
            
            print(f"\n{'='*80}")
            print(f"ITERATION {i}")
            print(f"{'='*80}")
            
            try:
                backward_pass_flag, del_J_alpha = self.backward_pass()

                if backward_pass_flag:
                    self.dec_reg_mu()
                    forward_pass_flag = self.forward_pass(del_J_alpha)

                    if not forward_pass_flag:
                        attempts = 0
                        max_attempts = 1000
                        while not forward_pass_flag and attempts < max_attempts:
                            # Simulated annealing
                            self.alpha = self.alpha * 0.9#9
                            forward_pass_flag = self.forward_pass(del_J_alpha)
                            attempts += 1
                        
                        if not forward_pass_flag:
                            print(f"  ✗ Forward pass failed after {max_attempts} attempts")
                            # Save checkpoint before potentially failing
                            self.save_checkpoint(iteration=i)
                            return False
                else:
                    self.inc_reg_mu()
                    print(f"  ✗ Backward pass failed at iteration {i}")
                    # Save checkpoint even on failure
                    self.save_checkpoint(iteration=i)
                    continue

                # Update alpha schedule
                if i < 5:
                    self.alpha = self.alpha * 0.999#self.alpha * 0.9
                else:
                    self.alpha = self.alpha * 0.999

                print(f"  Alpha: {self.alpha:.6f}")
                
                # Compute and store cost
                current_cost = float(self.calculate_total_cost(self.Z_aug_0, self.Z_aug, self.U, self.N))
                self.episodic_cost_history.append(current_cost)
                
                # Compute cost reduction
                if len(self.episodic_cost_history) > 1:
                    cost_reduction = self.episodic_cost_history[-2] - current_cost
                    reduction_pct = 100 * cost_reduction / self.episodic_cost_history[-2]
                    print(f"  Cost: {current_cost:.4f} (reduction: {cost_reduction:.4f}, {reduction_pct:.2f}%)")
                else:
                    print(f"  Cost: {current_cost:.4f}")
                
                print(f"  Cost history: {[f'{c:.2f}' for c in self.episodic_cost_history]}")
                
                # Save cost history to file
                np.save('cost_history.npy', np.array(self.episodic_cost_history))
                
                # ✅ SAVE CHECKPOINT AFTER SUCCESSFUL ITERATION
                self.save_checkpoint(iteration=i)
                
            except Exception as e:
                print(f"\n{'='*80}")
                print(f"ERROR AT ITERATION {i}")
                print(f"{'='*80}")
                print(f"Exception: {e}")
                import traceback
                traceback.print_exc()
                
                # Save emergency checkpoint
                emergency_path = self.checkpoint_dir / f'checkpoint_emergency_iter_{i}.npz'
                try:
                    checkpoint_data = {
                        'iteration': i,
                        'Z': self.Z,
                        'K': self.K,
                        'k': self.k,
                        'U': self.U,
                        'U_temp': self.U_temp,
                        'Z_temp': self.Z_temp,
                        'Z_aug': self.Z_aug,
                        'cost_history': np.array(self.episodic_cost_history),
                        'alpha': self.alpha,
                        'V_x': self.V_x,
                        'V_xx': self.V_xx,
                    }
                    np.savez_compressed(emergency_path, **checkpoint_data)
                    print(f"  Emergency checkpoint saved: {emergency_path}")
                except:
                    print(f"  Failed to save emergency checkpoint!")
                
                return False
        
        print(f"\n{'='*80}")
        print(f"iLQR COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Final cost: {self.episodic_cost_history[-1]:.4f}")
        print(f"Total reduction: {self.episodic_cost_history[0] - self.episodic_cost_history[-1]:.4f}")
        print(f"Reduction %: {100*(self.episodic_cost_history[0] - self.episodic_cost_history[-1])/self.episodic_cost_history[0]:.2f}%")
        
        return True


    def backward_pass(self):
        """
        Carry out the backward pass to compute the feedforward and feedback gains
        returns : backward_pass_flag : indicates if backward pass was successful
                  del_J_alpha : expected cost reduction
        """
        ################## defining local functions & variables for faster access ################
        k = np.copy(self.k)
        K = np.copy(self.K)
        V_x = np.copy(self.V_x)
        V_xx = np.copy(self.V_xx)
        ##########################################################################################

        V_x[self.N-1] = self.l_x_N(self.Z_aug[self.N-1])    
        V_xx[self.N-1] = 2*self.Q_final

        # Initialize before forward pass
        del_J_alpha = 0

        Fx_Fu = self.ltv_sys_id.traj_sys_id(np.concatenate((self.Z_0.reshape(1, self.n_z, 1), self.Z), axis=0), self.U, roll_start=self.model.init_solution_path)
        # Save rollouts
        X_pertb = np.load('X_pertb_rollouts.npy')
        U_pertb = np.load('U_pertb_rollouts.npy')
        np.save('checkpoints/rollouts/z_rollouts_it_'+str(self.current_iteration), X_pertb)
        np.save('checkpoints/rollouts/u_rollouts_it_'+str(self.current_iteration), U_pertb)

        for t in range(self.N-1, max(self.q, self.q_u)-2, -1):
            F_x = Fx_Fu[t][:,:self.n_aug]
            F_u = Fx_Fu[t][:,self.n_aug:]

            if t>0:
                Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.get_gradients(F_x,F_u,self.Z_aug[t-1],self.U[t],V_x[t], V_xx[t])
            else:
                Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.get_gradients(F_x,F_u,self.Z_aug_0,self.U[0],V_x[0], V_xx[0])
            
            try:
                np.linalg.cholesky(Q_uu)
            except np.linalg.LinAlgError:
                print("FAILED! Q_uu is not Positive definite at t=",t)
                backward_pass_flag = 0
                k = np.copy(self.k)
                K = np.copy(self.K)
                V_x = np.copy(self.V_x)
                V_xx = np.copy(self.V_xx)
                break
            else:
                backward_pass_flag = 1
                Q_uu_inv = np.linalg.inv(Q_uu)
                k[t] = -(Q_uu_inv @ Q_u)
                K[t] = -(Q_uu_inv @ Q_ux)

                del_J_alpha += -self.alpha*((k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((k[t].T) @ (Q_uu @ k[t]))
                
                if t>0:
                    V_x[t-1] = Q_x + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_ux.T) @ k[t])
                    V_xx[t-1] = Q_xx + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_ux) + ((Q_ux.T) @ K[t])

        ######################### Update the new gains ##############################################
        self.k = np.copy(k)
        self.K = np.copy(K)
        self.V_x = np.copy(V_x)
        self.V_xx = np.copy(V_xx)

        return backward_pass_flag, del_J_alpha
    
    

    def forward_pass(self, del_J_alpha):
        """
        Forward pass with line search and NaN handling
        del_J_alpha : expected cost reduction scaled with alpha
        returns : forward_pass_flag : 1 if forward pass is successful else 0
        """
        # Cost before forward pass
        J1 = self.calculate_total_cost(self.Z_aug_0, self.Z_aug, self.U, self.N)

        # Save current state as reference
        self.Z_temp = np.copy(self.Z)
        self.U_temp = np.copy(self.U)
        self.Z_aug_temp = np.copy(self.Z_aug)
        
        # ===================================================================
        # NaN HANDLING: Try forward pass with progressively smaller alpha
        # ===================================================================
        max_nan_attempts = 30
        nan_attempt = 0
        alpha_backup = self.alpha
        simulation_success = False
        
        while not simulation_success and nan_attempt < max_nan_attempts:
            try:
                # ============================================================
                # CRITICAL: Catch PyFR crashes HERE before Z is even created
                # ============================================================
                self.forward_pass_simulate()
                
                # If we got here, simulation completed without crashing
                simulation_success = True
                
                # Now check if the results are valid
                if np.any(np.isnan(self.Z)) or np.any(np.isinf(self.Z)):
                    print(f"  ⚠️  NaN/Inf in observations despite successful sim (attempt {nan_attempt + 1})")
                    simulation_success = False
                
                if np.any(np.isnan(self.U)) or np.any(np.isinf(self.U)):
                    print(f"  ⚠️  NaN/Inf in controls despite successful sim (attempt {nan_attempt + 1})")
                    simulation_success = False
                
                if simulation_success and nan_attempt > 0:
                    print(f"  ✓ Valid trajectory found at alpha={self.alpha:.6f} after {nan_attempt} recovery attempts")
                    
            except RuntimeError as e:
                # PyFR detected NaN and crashed!
                if "NaN" in str(e) or "nan" in str(e).lower():
                    print(f"  ⚠️  PyFR NaN crash at alpha={self.alpha:.6f} (attempt {nan_attempt + 1}/{max_nan_attempts})")
                    print(f"      Error: {str(e)[:100]}...")  # First 100 chars
                else:
                    print(f"  ⚠️  PyFR RuntimeError at alpha={self.alpha:.6f}: {str(e)[:100]}")
                
                simulation_success = False
                
            except subprocess.CalledProcessError as e:
                # PyFR subprocess crashed
                print(f"  ⚠️  PyFR subprocess crash at alpha={self.alpha:.6f} (attempt {nan_attempt + 1}/{max_nan_attempts})")
                print(f"      Return code: {e.returncode}")
                simulation_success = False
                
            except Exception as e:
                # Any other error during simulation
                print(f"  ⚠️  Simulation error at alpha={self.alpha:.6f} (attempt {nan_attempt + 1}/{max_nan_attempts})")
                print(f"      Error type: {type(e).__name__}")
                print(f"      Error: {str(e)[:100]}...")
                simulation_success = False
            
            # If simulation failed, reduce alpha and retry
            if not simulation_success:
                # Z_temp and U_temp are still intact (simulation never completed)
                # Just need to reduce alpha for next attempt
                
                self.alpha = self.alpha * 0.9  # Aggressive reduction
                nan_attempt += 1
                
                if self.alpha < 1e-6:
                    print(f"  ✗ Alpha too small ({self.alpha:.2e}), cannot find stable trajectory")
                    self.alpha = alpha_backup
                    # Ensure Z, U are restored to valid state
                    self.Z = np.copy(self.Z_temp)
                    self.U = np.copy(self.U_temp)
                    self.Z_aug = np.copy(self.Z_aug_temp)
                    return 0
                
                print(f"      Reducing alpha to {self.alpha:.6f} and retrying...")
                
                # Make sure we're not carrying over any partial results
                # (though simulation crash should have prevented any updates)
                self.Z = np.copy(self.Z_temp)
                self.U = np.copy(self.U_temp)
                self.Z_aug = np.copy(self.Z_aug_temp)
        
        if not simulation_success:
            print(f"  ✗ Failed to find valid trajectory after {max_nan_attempts} attempts")
            self.alpha = alpha_backup
            self.Z = np.copy(self.Z_temp)
            self.U = np.copy(self.U_temp)
            self.Z_aug = np.copy(self.Z_aug_temp)
            return 0
        
        # ===================================================================
        # Cost-based line search acceptance (now with valid trajectory)
        # ===================================================================
        try:
            J2 = self.calculate_total_cost(self.Z_aug_0, self.Z_aug, self.U, self.N)
            
            if np.isnan(J2) or np.isinf(J2):
                print(f"  ✗ Cost is NaN/Inf")
                self.Z = np.copy(self.Z_temp)
                self.U = np.copy(self.U_temp)
                self.Z_aug = np.copy(self.Z_aug_temp)
                self.alpha = alpha_backup
                return 0
                
        except Exception as e:
            print(f"  ✗ Cannot compute cost: {e}")
            self.Z = np.copy(self.Z_temp)
            self.U = np.copy(self.U_temp)
            self.Z_aug = np.copy(self.Z_aug_temp)
            self.alpha = alpha_backup
            return 0
        
        # Print cost comparison
        print(f"  [Line Search] J1={float(J1):.4f}, J2={float(J2):.4f}, ΔJ={float(J1-J2):.4f}, alpha={self.alpha:.6f}")
        print('del_J_alpha = ', del_J_alpha)
        print('Cost change = ', (J1-J2)/del_J_alpha)
        # FIXED ACCEPTANCE CRITERION (MATLAB-style)
        cost_reduction = J1 - J2
        
        if (J1-J2)/del_J_alpha>self.J_change_eps:#cost_reduction> 0:
            # Cost decreased - ACCEPT!
            # print(f"  ✓ Accepted: Cost reduced by {float(cost_reduction):.4f} ({100*float(cost_reduction)/J1:.2f}%)")
            print('New cost = ', J2)
            print('Cost reduced by (%) = ', 100*(J1-J2)/J2)
            forward_pass_flag = 1
        else:
            # Cost increased - REJECT!
            # print(f"  ✗ Rejected: Cost increased by {-float(cost_reduction):.4f} ({100*(-float(cost_reduction))/J1:.2f}%)")
            print('Cost increased')
            self.Z = np.copy(self.Z_temp)
            self.U = np.copy(self.U_temp)
            self.Z_aug = np.copy(self.Z_aug_temp)
            forward_pass_flag = 0

        return forward_pass_flag
    
    def forward_pass_simulate_original(self):
        """ 
        Simulate the system with updated controls 
        """
        ################## defining local functions & variables for faster access ################
        n_z, n_u, q, q_u = self.n_z, self.n_u, self.q, self.q_u
		##########################################################################################
        z = np.zeros((n_z*q,1))
        u = np.zeros((n_u*q_u,1))
        d_z = np.zeros((n_z*q,1))
        d_u = np.zeros((n_u*q_u,1))

        for t in range(self.N):
            if t < max(q, q_u)-1: #TODO: check 
                self.U[t] = self.U_temp[t] + self.alpha*self.k[t]
                d_u[n_u*(q_u-t-1):n_u*(q_u-t)] = self.U[t]-self.U_temp[t]
                u[n_u*(q_u-t-1):n_u*(q_u-t)] = self.U[t]
                if t != 0:
                    d_z[n_z*(q-t-1):n_z*(q-t)] = self.Z[t-1] - self.Z_temp[t-1]	
                    z[n_z*(q-t-1):n_z*(q-t)] = self.Z[t-1]

                # ✅ FIX: Build Z_aug even during collection phase!
                if t > 0:
                    # Pad with first observation for missing history
                    z_history = np.zeros((n_z*q, 1))
                    # Fill available history
                    for i in range(t):
                        z_history[n_z*(q-t+i):n_z*(q-t+i+1)] = self.Z[i]
                    # Pad beginning with Z[0]
                    for i in range(q-t):
                        z_history[n_z*i:n_z*(i+1)] = self.Z[0]
                    
                    # Build augmented state
                    if q_u > 1:
                        self.Z_aug[t] = np.vstack([z_history, u[n_u:]])
                    else:
                        self.Z_aug[t] = z_history
                else:
                    # t=0: Only have initial observation
                    z_history = np.tile(self.Z_0, (q, 1))
                    if q_u > 1:
                        self.Z_aug[t] = np.vstack([z_history, np.zeros((n_u*(q_u-1), 1))])
                    else:
                        self.Z_aug[t] = z_history
                # ✅ FIX END
            else:
                d_z_prev = d_z[:n_z*(q-1)]
                d_z[n_z:] = d_z_prev
                z_prev = z[:n_z*(q-1)]
                z[n_z:] = z_prev

                d_u_prev = d_u[:n_u*(q_u-1)]
                d_u[n_u:] = d_u_prev
                u_prev = u[:n_u*(q_u-1)]
                u[n_u:] = u_prev

                d_z[:n_z] = self.Z[t-1] - self.Z_temp[t-1]
                z[:n_z] = self.Z[t-1]

                self.U[t] = self.U_temp[t] + self.alpha*self.k[t] + (self.K[t] @ np.vstack([d_z, d_u[n_u:]]))
                d_u[:n_u] = self.U[t]-self.U_temp[t]
                u[:n_u] = self.U[t]
                self.Z_aug[t] = np.vstack([z, u[n_u:]])

            if t==0:
                # First step: use initial solution
                self.Z[t] = self.model.simulate_step(
                    None, 
                    self.U[t].flatten(), 
                    restart_from=self.model.init_solution_path
                ).reshape((self.n_z,1))
            else:
                # Subsequent steps: let simulator use current solution
                self.Z[t] = self.model.simulate_step(
                    self.Z[t-1], 
                    self.U[t].flatten(), 
                    restart_from=None  # Uses most recent solution in work_dir
                ).reshape((self.n_z,1))

    def forward_pass_simulate(self):
        """Forward pass using internal feedback in PyFR plugin"""
        
        # Switch to feedback mode
        self.model.set_mode('feedback')
        
        # Save feedback gains AND U_temp in current directory
        # print('Feedback Gain K = ', self.K)
        np.save('feedback_K.npy', self.K)
        np.save('feedback_k.npy', self.k)
        np.save('U_temp.npy', self.U_temp)
        np.save('Z_reference.npy', self.Z_temp.reshape(self.N, self.n_z))
        np.save('feedback_metadata.npy', {
            'dt': 0.5,
            't_start': 125.0,
            'q': self.q,
            'q_u': self.q_u,
            'alpha': self.alpha,
            'n_z': self.n_z,
            'n_u': self.n_u,
            'n_z_aug': self.n_aug
        })
        
        print(f"[forward_pass_simulate] Saved U_temp: range=[{self.U_temp.min():.3f}, {self.U_temp.max():.3f}]")
        
        # Run PyFR with internal feedback
        Z_result = self.model.simulate_trajectory(
            y_init=self.model.init_solution_path,
            u=None,
            horizon=self.N
        )
        
        # Extract observations ONLY (don't overwrite U yet)
        self.Z = Z_result[:-1].reshape(self.N, self.n_z, 1)
        
        # ================================================================
        # FIXED: Build augmented state with NEWEST-to-OLDEST ordering
        # ================================================================
        for t in range(self.N):
            # Build observation history NEWEST to OLDEST
            if t < self.q - 1:
                # Early timesteps: [z[t], z[t-1], ..., z[0], z[0], ...]
                z_actual = [self.Z[i] for i in range(t, -1, -1)]  # t, t-1, ..., 0
                z_pad = [self.Z[0]] * (self.q - t - 1)
                z_history = np.vstack(z_actual + z_pad)
            else:
                # Normal case: [z[t], z[t-1], ..., z[t-q+1]]
                z_history = np.vstack([self.Z[i] for i in range(t, t-self.q, -1)])
            
            z_flat = z_history.reshape(self.q * self.n_z, 1)
            
            # Build control history (also NEWEST to OLDEST for consistency)
            if self.q_u > 1:
                if t < self.q_u - 1:
                    if t == 0:
                        u_history = np.zeros((self.q_u - 1, self.n_u, 1))
                    else:
                        # [u[t-1], u[t-2], ..., u[0], 0, ...]
                        u_actual = [self.U[i] for i in range(t-1, -1, -1)]
                        u_pad = [np.zeros((self.n_u, 1))] * (self.q_u - 1 - t)
                        u_history = np.vstack(u_actual + u_pad)
                else:
                    # [u[t-1], u[t-2], ..., u[t-q_u+1]]
                    u_history = np.vstack([self.U[i] for i in range(t-1, t-self.q_u, -1)])
                
                u_flat = u_history.reshape(self.n_u * (self.q_u - 1), 1)
                z_aug_t = np.vstack([z_flat, u_flat])
            else:
                z_aug_t = z_flat
            
            # Store augmented state
            self.Z_aug[t] = z_aug_t
            
            # Build reference augmented state (NEWEST to OLDEST)
            if t < self.q - 1:
                z_ref_actual = [self.Z_temp[i] for i in range(t, -1, -1)]
                z_ref_pad = [self.Z_temp[0]] * (self.q - t - 1)
                z_ref_history = np.vstack(z_ref_actual + z_ref_pad)
            else:
                z_ref_history = np.vstack([self.Z_temp[i] for i in range(t, t-self.q, -1)])
            
            z_ref_flat = z_ref_history.reshape(self.q * self.n_z, 1)
            
            if self.q_u > 1:
                if t < self.q_u - 1:
                    if t == 0:
                        u_ref_history_array = np.zeros((self.q_u - 1, self.n_u, 1))
                    else:
                        u_ref_actual = [self.U_temp[i] for i in range(t-1, -1, -1)]
                        u_ref_pad = [np.zeros((self.n_u, 1))] * (self.q_u - 1 - t)
                        u_ref_history_array = np.vstack(u_ref_actual + u_ref_pad)
                else:
                    u_ref_history_array = np.vstack([self.U_temp[i] for i in range(t-1, t-self.q_u, -1)])
                
                u_ref_flat = u_ref_history_array.reshape(self.n_u * (self.q_u - 1), 1)
                z_ref_aug_t = np.vstack([z_ref_flat, u_ref_flat])
            else:
                z_ref_aug_t = z_ref_flat
            
            # Compute control that was applied: u = u_temp + alpha*k + K @ delta_z
            delta_z_aug = z_aug_t - z_ref_aug_t
            feedforward = self.alpha * self.k[t]
            feedback = self.K[t] @ delta_z_aug
            self.U[t] = self.U_temp[t] + feedforward + feedback
            # self.U[t] = np.clip(self.U[t], -50.0, 50.0)
        
        print(f"[forward_pass_simulate] Computed U: range=[{self.U.min():.3f}, {self.U.max():.3f}]")
        
    def initialize_traj(self,u_init):
        """
        Initialize the nominal trajectory with an initial guess for control
        u_init : (N, n_u, 1)
        """
        if u_init is None:
            # self.U = np.random.normal(0, self.nominal_init_stddev, (self.N, self.n_u, 1))
            self.U = 0.0316174943015858*np.random.uniform(-0.1, 0.1, (self.N, self.n_u, 1))
        else:
            self.U = u_init #TODO: check the shape of u_init

        self.U = np.load('init_guess.npy').reshape((self.N, self.n_u, 1))*0.0316174943015858
        self.U_temp = self.U
        self.forward_pass_simulate()
        self.Z_temp = self.Z

    def save_policy(self, path_to_file):
        """
        Save the learned policy to a json file
        Modified for POD_iLQR to save augmented state
        """
        Pi = {}
        Pi['U'] = {}
        Pi['K'] = {}
        Pi['Z_aug'] = {}
        Pi['Z'] = {}

        for t in range(self.N):
            Pi['Z_aug'][t] = self.Z_aug[t].tolist()
            Pi['Z'][t] = self.Z[t].tolist()
            Pi['U'][t] = self.U[t].tolist()
            Pi['K'][t] = self.K[t].tolist()

        with open(path_to_file, 'w') as outfile:
            json.dump(Pi, outfile, indent=2)
