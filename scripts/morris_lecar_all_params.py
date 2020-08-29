"""
Morris Lecar model estimated using optimal control 
dynamical state and parameter estimation.

Created by Nirag Kadakia at 12:12 08-28-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys, time
from varanneal import va_ode
sys.path.append('../src/')
from est_funcs import ML_est_all_params


output_data_dir = '../data/results'
meas_data_dir = '../data/meas_data'

seed = int(sys.argv[1])

# Annealing parameters
RM = 1/5.0**2
RF0 = [1e-4, 1e0]
alpha = 2
beta_array = np.linspace(0, 24, 25)

# Truncate data by 1 timepoint for SimpsonHermite -- bug to be fixed
data = np.load('%s/ML_chaotic_data_sigma=5.0.npy' % meas_data_dir)[:-1, :]
Tt = data[:, 0]
dt = Tt[1] - Tt[0]
Nn = len(Tt)

# Stimulus consists of both the injected stimulus and the measured voltage
stim = data[:, [1, 2]]

# Measured data is just the measured voltage
observations = data[:, [2]]

# Parameters now include time-dependent nudging terms; state space is 2D
D = 2
Lidx = [0]
Pidx = range(10)
Uidx = []
state_bounds = [[-100, 100]] + [[0, 1]]
param_bounds =  [[.01, 200]]*10
bounds = state_bounds + param_bounds

# Initial conditions; initial forcing params set randomly to bounds
np.random.seed(seed)
X0 = np.random.uniform(-100, 100, (Nn, D))
np.random.seed(seed)
P0 = [np.array([np.random.uniform(1, 200)])]*10
		
		
# Run the annealing using L-BFGS-B
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 
			    'maxfun':1000000, 'maxiter':1000000}
anneal1 = va_ode.Annealer()
anneal1.set_model(ML_est_all_params, D)
anneal1.set_data(observations, t=Tt, stim=stim)
anneal1.anneal(X0, P0, alpha, beta_array, RM, RF0, Lidx, Pidx, Uidx, 
				action='A_gaussian', dt_model=dt, 
				init_to_data=True, disc='SimpsonHermite', method='L-BFGS-B',
				opt_args=BFGS_options, bounds=bounds, adolcID=0)

# Save the results
anneal1.save_paths("%s/all/paths_%d.npy" % (output_data_dir, seed))
anneal1.save_params("%s/all/params_%d.npy" % (output_data_dir, seed))
anneal1.save_action_errors("%s/all/action_errors_%d.npy" 
						   % (output_data_dir, seed))


