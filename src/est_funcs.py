"""
Holds the functions 

Created by Nirag Kadakia at 16:15 08-28-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np


def ML_est_all_params(t, X, args):
	"""
	"""
	
	params, stim = args
	
	I = stim[:, 0]
	data = stim[:, 1]
	v = X[:, 0]
	w = X[:, 1]
	
	inv_cap = 1./2.5
	g_fast = params[:, 0] #15
	g_slow = params[:, 1] #20
	g_leak = params[:, 2] #2
	E_Na = params[:, 3] #50 
	E_K = -params[:, 4] #-100
	E_leak = -params[:, 5] #-70
	phi_w = params[:, 6]  #.12
	beta_w = 0
	beta_m = -params[:, 7] # -1.2
	inv_gamma_m = 1./params[:, 8] #1./18
	inv_gamma_w = 1./params[:, 9] #1./10
	
	def m_inf(v):
		return 0.5*(1 + np.tanh((v - beta_m)*inv_gamma_m))
		
	def w_inf(v):
		return 0.5*(1 + np.tanh((v - beta_w)*inv_gamma_w))

	def inv_tau_w(v):
		return np.cosh((v - beta_w)/2*inv_gamma_w)
	
	dvdt = inv_cap*(I - g_fast*m_inf(v)*(v - E_Na) - g_slow*w*(v - E_K)\
			- g_leak*(v - E_leak))
	
	dwdt = phi_w*(w_inf(v) - w)*inv_tau_w(v)

	return np.array([dvdt, dwdt]).T

def ML_est_conductances(t, X, args):
	"""
	"""
	
	params, stim = args
	
	I = stim[:, 0]
	data = stim[:, 1]
	v = X[:, 0]
	w = X[:, 1]
	
	inv_cap = 1./2.5
	g_fast = params[:, 0] #20
	g_slow = params[:, 1] #20
	g_leak = params[:, 2] #2
	E_Na = 50 
	E_K = -100
	E_leak = -70
	phi_w = .12
	beta_w = 0
	beta_m = -1.2
	inv_gamma_m = 1./18
	inv_gamma_w = 1./10
	
	def m_inf(v):
		return 0.5*(1 + np.tanh((v - beta_m)*inv_gamma_m))
		
	def w_inf(v):
		return 0.5*(1 + np.tanh((v - beta_w)*inv_gamma_w))

	def inv_tau_w(v):
		return np.cosh((v - beta_w)/2*inv_gamma_w)
	
	dvdt = inv_cap*(I - g_fast*m_inf(v)*(v - E_Na) - g_slow*w*(v - E_K)\
			- g_leak*(v - E_leak))
	
	dwdt = phi_w*(w_inf(v) - w)*inv_tau_w(v)

	return np.array([dvdt, dwdt]).T
