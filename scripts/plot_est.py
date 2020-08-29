"""
Morris Lecar model estimation -- plot optimally estimated
state in a video

Created by Nirag Kadakia at 12:00 08-29-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2


# Load true states and observations of membrance voltage
true = np.load('../data/meas_data/ML_chaotic_true.npy')[:-1, :]
obs = np.load('../data/meas_data/ML_chaotic_data_sigma=5.0.npy')[:-1, :]
num_ests = 1000

# Get minimum error path based on unobserved (gating) variable only
errs = []
for iP in range(num_ests):
	path = np.load('../data/results/all/paths_%d.npy' % iP)
	errs.append(np.sum((true[:, 2] - path[-1, :, 2])**2.0))
opt_idx = np.argmin(errs)	
print ('optimum index: ', opt_idx)

frms_to_plot = [0, 2, 3, 5, 8, 10, 11, 12, 13, 14, 15, 
				16, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24, 24, 24]

# Plot progression of estimate for optimal path through annealing
path = np.load('../data/results/all/paths_%d.npy' % opt_idx)
fig = plt.figure()
fig.set_size_inches(20, 10)
video = None
for frm in frms_to_plot:
	print (frm, end=" ", flush=True)

	# Plot regularly in matplotlib first
	ax1 = plt.subplot(211)
	plt.scatter(obs[:, 0], obs[:, 2], color='blue', s=15)
	plt.plot(true[:, 0], true[:, 1], color='k', lw=4)
	plt.plot(path[0, :, 0], path[frm, :, 1], color='red', lw=6, ls='--')
	plt.xticks([])
	plt.yticks(np.arange(-100, 100, 40), fontsize=35)
	plt.ylabel(r'$V(t)$ (mV)', fontsize=35)
	plt.xlim(true[0, 0], true[-1, 0])
	plt.ylim(-90, 50)
	ax2 = plt.subplot(212)
	plt.plot(true[:, 0], true[:, 2], color='k', lw=7)
	plt.plot(path[0, :, 0], path[frm, :, 2], color='red', lw=6, ls='--')
	plt.xticks(fontsize=35)
	plt.yticks(np.arange(0, 1, 0.2), fontsize=35)
	plt.xlabel('Time (s)', fontsize=35)
	plt.ylabel(r'$n(t)$', fontsize=35)
	plt.xlim(true[0, 0], true[-1, 0])
	plt.ylim(0, 0.7)
	for axis in ['right','top']:
		ax1.spines[axis].set_linewidth(0)
		ax2.spines[axis].set_linewidth(0)
	for axis in ['bottom', 'left']:
		ax1.spines[axis].set_linewidth(3)
		ax2.spines[axis].set_linewidth(3)
	fig.canvas.draw()

	# Save to pixelated array
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	
	# Opencv video container must have same shape as data 
	if video is None:
		fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
		video = cv2.VideoWriter('../data/results/est_paths.avi', fourcc, 3, 
								(data.shape[1], data.shape[0]))
		
	# Video image contains all plots up to this frame; clear to redraw
	video.write(data)
	fig.clear()
video.release()

# Plot progression of parameters of optimal path throuugh annealing
true_params = [15, 20, 2, 50, 100, 70, 0.12, 1.2, 18, 10]
params = np.load('../data/results/all/params_%d.npy' % opt_idx)
fig = plt.figure()
fig.set_size_inches(5, 5)
video = None
x = range(len(true_params)) 
for frm in frms_to_plot:
	print (frm, end=" ", flush=True)

	# Plot regularly in matplotlib first
	ax = plt.subplot(111)
	plt.scatter(x, true_params, color='k', s=100, alpha=0.5)
	plt.scatter(x, params[frm, :], color='red', s=100, alpha=0.5)
	plt.xticks(x, labels=[r'$g_f$', r'$g_s$', r'$g_L$', r'$E_{Na}$', 
						  r'-$E_K$', r'-$E_{L}$', r'$\phi_W$', r'$\beta_m$', 
						  r'$\gamma_m^{-1}$', r'$\gamma_w^{-1}$'], fontsize=16)
	plt.yticks(np.arange(-150, 150, 30), fontsize=16)	
	plt.ylim(-35, 125)
	for axis in ['right','top']:
		ax.spines[axis].set_linewidth(0)
	for axis in ['left', 'bottom']:
		ax.spines[axis].set_linewidth(3)
	fig.canvas.draw()

	# Save to pixelated array
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	
	# Opencv video container must have same shape as data 
	if video is None:
		fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
		video = cv2.VideoWriter('../data/results/est_params.avi', fourcc, 3, 
								(data.shape[1], data.shape[0]))
		
	# Video image contains all plots up to this frame; clear to redraw
	video.write(data)
	fig.clear()
video.release()

