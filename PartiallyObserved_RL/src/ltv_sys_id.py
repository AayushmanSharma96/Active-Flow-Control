import numpy as np
import math
import sys
# np.random.seed(50)

class LTV_SysID:

    def __init__(self, MODEL, n_x, n_u, N, n_samples=500, pert_sigma = 1e-7):
        self.model = MODEL
        self.n_x = n_x # State dimension
        self.n_u = n_u # Control dimension
        self.N = N # Horizon length
        self.n_samples = n_samples       
        self.sigma = pert_sigma # Standard deviation of the perturbation 

    def sys_id_state_pertb(self, x_t, u_t):
        '''
        System identification at a single time step using state perturbations
        x_t : (n_x, 1)
        u_t : (n_u, 1)
        returns : AB : (n_x, n_x+n_u)
		'''
		################## defining local functions & variables for faster access ################
        n_x = self.n_x
        n_u = self.n_u
		##########################################################################################
		
        dXU = np.random.normal(0.0, self.sigma, (self.n_samples, n_x+n_u))
        dX = dXU[:,:n_x]
        dU = dXU[:,n_x:]

        dX_next = self.batch_simulate(x_t.T+dX, u_t.T+dU) - self.batch_simulate(x_t.T, u_t.T)

        AB = np.linalg.solve((dXU.T @ dXU), (dXU.T@dX_next)).T

        return AB

    def batch_simulate(self, X, U):
        '''
        Need a simulate function from the actual example
        forward_simulate(self,x,u)
        X : (n_samples, n_x)
        U : (n_samples, n_u)
        returns : X_next : (n_samples, n_x)
        '''
        X_next = np.zeros((X.shape[0], self.n_x))
        for i in range(X.shape[0]):
            X_next[i,:] = self.model.simulate_step(X[i],U[i])

        return X_next

    def traj_sys_id_state_pertb(self,x_nom,u_nom):
        """
        System identification over the entire trajectory using state perturbations
        x_nom : (N+1, n_x, 1)
        u_nom : (N, n_u, 1)
        returns : traj_AB : (N, n_x, n_x+n_u)
        """
        traj_AB = []
        for i in range(u_nom.shape[0]):
            traj_AB.append(self.sys_id_state_pertb(x_nom[i],u_nom[i]))
        
        return np.array(traj_AB)
    
    def traj_sys_id(self, x_nom, u_nom, roll_start = 0, central_diff=0): #TODO rollout 
        """
        System identification over the entire trajectory using control perturbations
        x_nom : (N+1, n_x, 1)
        u_nom : (N, n_u, 1)
        returns : traj_AB : (N, n_x, n_x+n_u)
        """            
        delta_x = np.zeros(((self.N+1), self.n_x, self.n_samples))
        X_pertb, U_pertb = self.generate_rollouts(x_nom, u_nom, roll_start)
        traj_AB = []

        # Generating delta_x for all rollouts
        for i in range(self.N+1):
            delta_x[i] = X_pertb[i] - x_nom[i]
            if i>0:
                regressor = np.hstack([delta_x[i-1,:].T, (U_pertb[i-1].T).reshape(self.n_samples, self.n_u)])
                AB = np.linalg.lstsq(regressor, delta_x[i].T, rcond=None)[0].T
                traj_AB.append(AB)
                           
        return np.array(traj_AB)
    
    # def generate_rollouts(self, x_nom, u_nom):
    #     """
    #     Generate rollouts around nominal trajectory using control perturbations
    #     x_nom : (N+1, n_x, 1)
    #     u_nom : (N, n_u, 1)
    #     returns : X_pertb : (N+1, n_x, n_samples)
    #               U_pertb : (N, n_u, n_samples)
    #     """
    #     # Taking control perturbations as a function of max control
    #     u_max = np.max(abs(u_nom))
    #     U_pertb = self.sigma*u_max*np.random.normal(0, 1, (self.N, self.n_u, self.n_samples))
    #     X_pertb = np.zeros((self.N+1, self.n_x, self.n_samples))
    #     ctrl = np.zeros((self.n_u, 1))

    #     for j in range(self.n_samples): # parallelize this loop
    #         X_pertb[0, :, j] = x_nom[0].flatten()
    #         for i in range(self.N):
    #             ctrl[:] = u_nom[i] + U_pertb[i,:,j].reshape(np.shape(u_nom[i]))
    #             X_pertb[i+1, :, j] = self.model.simulate_step(X_pertb[i, :, j], ctrl.flatten()).flatten()

    #     return X_pertb, U_pertb

    def generate_rollouts(self, x_nom, u_nom, roll_start):
        """
        Generate rollouts around nominal trajectory using control perturbations
        
        FIXED: Use existing model instead of creating new simulators
        
        x_nom : (N+1, n_x, 1)
        u_nom : (N, n_u, 1)
        roll_start: Initial solution file path
        returns : X_pertb : (N+1, n_x, n_samples)
                U_pertb : (N, n_u, n_samples)
        """
        # For flow control, set model to rollout mode
        self.model.set_mode('rollout')

        # Taking control perturbations as a function of max control
        u_max = np.max(abs(u_nom))
        if u_max < 4.0 * 0.0316174943015858:   # 0.1265
            u_max = 4.0 * 0.0316174943015858 / self.sigma  # Avoid division by zero
        
        U_pertb = self.sigma * u_max * np.random.normal(0, 1, (self.N, self.n_u, self.n_samples))
        

        ctrl = u_nom + U_pertb
        X_pertb = np.zeros((self.N+1, x_nom.shape[1], self.n_samples))
        
        print(f"\n  Running {self.n_samples} system ID rollouts...")
        
        # Run rollouts sequentially using the existing model
        # Each rollout will use its own work_dir internally
        for j in range(self.n_samples):
            print(f"    Rollout {j+1}/{self.n_samples}:", end=" ", flush=True)
            
            # Run trajectory with perturbed control
            X_pertb[:, :, j] = self.model.simulate_trajectory(
                y_init=roll_start,
                u=ctrl[:, :, j].flatten(),#squeeze(),  # Shape: (N,) for scalar control
                horizon=self.N
            )
            
            print(f"✓")
        
        print(f"  System ID rollouts complete\n")
        np.save("U_pertb_rollouts.npy", U_pertb)
        np.save("X_pertb_rollouts.npy", X_pertb)
        
        return X_pertb, U_pertb
