"""
Morris Lecar model estimated using optimal control 
dynamical state and parameter estimation.

Created by Nirag Kadakia at 11:22 05-05-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

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

#params = np.load('../data/results/all/params_%d.npy' % opt_idx)

path = np.load('../data/results/all/paths_%d.npy' % opt_idx)


fig = plt.figure()
fig.set_size_inches(20, 10)
video = None
frms_to_plot = [0, 2, 3, 5, 8, 10, 11, 12, 13, 14, 15, 
				16, 17, 18, 19, 20, 21, 22, 23, 24]
for frm in frms_to_plot:
	print (frm, end=" ", flush=True)

	# Plot regularly in matplotlib first
	ax1 = plt.subplot(211)
	plt.scatter(obs[:, 0], obs[:, 2], color='orangered', s=15)
	plt.plot(true[:, 0], true[:, 1], color='k', lw=4)
	plt.plot(path[0, :, 0], path[frm, :, 1], color='b', lw=6, ls='--')
	plt.xticks([])
	plt.yticks(np.arange(-100, 100, 40), fontsize=30)
	plt.ylim(-90, 50)
	ax2 = plt.subplot(212)
	plt.plot(true[:, 0], true[:, 2], color='k', lw=7)
	plt.plot(path[0, :, 0], path[frm, :, 2], color='b', lw=6, ls='--')
	plt.xticks(fontsize=30)
	plt.yticks(np.arange(0, 1, 0.2), fontsize=30)
	plt.ylim(0, 0.6)
	for axis in ['right','top', 'bottom', 'left']:
		ax1.spines[axis].set_linewidth(5)
		ax2.spines[axis].set_linewidth(5)
	fig.canvas.draw()

	# Save to pixelated array
	data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	
	# Opencv video container must have same shape as data 
	if video is None:
		fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
		video = cv2.VideoWriter('../data/results/anim.avi', fourcc, 3, 
								(data.shape[1], data.shape[0]))
		
	# Video image contains all plots up to this frame; clear to redraw
	video.write(data)
	fig.clear()
video.release()