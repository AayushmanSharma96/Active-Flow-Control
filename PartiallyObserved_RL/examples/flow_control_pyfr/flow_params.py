import numpy as np


state_dimension = 2 # Unknown
control_dimension = 1
obs_dimension = 2


# Cost parameters for ARMA design
q = 3
q_u = 3
alpha = 1.0

C = np.eye(obs_dimension)
n_z = obs_dimension

n_aug = n_z*q+control_dimension*(q_u-1)
Q_aug = np.zeros((n_aug, n_aug))

diag_weights = [1] + [1]*(q-2) + [1]#[1] + [2]*(q-2) + [1]
qv = 500.0
for i in range(q):
    Q_aug[i*n_z:(i+1)*n_z, i*n_z:(i+1)*n_z] = diag_weights[i] * qv * np.diag([0.1, 1])#np.eye(n_z)
    # if i < q-1:
    #     Q_aug[i*n_z:(i+1)*n_z, (i+1)*n_z:(i+2)*n_z] = -qv * np.eye(n_z)
    #     Q_aug[(i+1)*n_z:(i+2)*n_z, i*n_z:(i+1)*n_z] = -qv * np.eye(n_z)

Q_final_aug = 20*Q_aug
R = 100*np.eye(control_dimension)#/(0.0316174943015858*0.0316174943015858)

# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.5
horizon = 50
nominal_init_stddev = 3.0

# Cost parameters for feedback design
W_x_LQR = 10*np.eye(state_dimension*state_dimension)
W_u_LQR = 2*np.eye(2*control_dimension*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension*state_dimension)

# D2C parameters
feedback_n_samples = 50
n_substeps = 1

# LQR Params

Q_lqr = np.zeros((n_aug, n_aug))
diag_weights = [1]*(q)
qv = 500.0
for i in range(q):
    Q_lqr[i*n_z:(i+1)*n_z, i*n_z:(i+1)*n_z] = diag_weights[i] * qv * np.diag([0.01, 1])#np.eye(n_z)
    

R_lqr = 100*0.05*np.eye(control_dimension)#*15#/2
