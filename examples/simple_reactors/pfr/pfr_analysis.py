########################################################################################################################
# Copyright 2024 the authors (see AUTHORS file for full list).
#
#                                                                                                                    #
# This file is part of OpenCCM.
#
#                                                                                                                    #
# OpenCCM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
#
# License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any  later version.                                                                                                       #
#                                                                                                                    #
# OpenCCM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.                                                                                                             #
#                                                                                                                     #
# You should have received a copy of the GNU Lesser General Public License along with OpenCCM. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################

import os 
from pathlib import Path

import numpy as np 
from numpy import heaviside as HV
import matplotlib.pyplot as plt
from matplotlib import animation

from openccm import run, ConfigParser

# Save movie (transient solution simulation)?
save_movie = False

# CFD simulation parameters and reaction rate constant
Vs = 1
tau = 10
Q = Vs/tau
k = 0.1

# Setup analytical solutions
a_bc = 1 # BC
a_ic = 0 # IC
Ca_func = lambda V, t: ( a_ic * np.exp(-k*t) * (1- HV(t-V/Q, 1)) ) + ( a_bc * np.exp(-k*V/Q) * HV(t-V/Q, 1) )
						
b_bc = 0 # BC
b_ic = 0 # IC
Cb_func = lambda V, t: 2*a_ic * ( 1 - HV(t-V/Q, 1) ) * (1-np.exp(-k*t))  + 2*a_bc*HV(t-V/Q, 1) * (1-np.exp(-k*V/Q)) + (b_bc - b_ic)*HV(t-V/Q, 1) + b_ic

# Solve for different volume discretization points in PFR
for pointsPFR in [21, 101, 501]:
	
	# Runtime change of points_per_pfr parameter regardless of what configuration file was given.
	os.chdir(Path(__file__).parents[0])
	config_parser = ConfigParser('CONFIG')
	config_parser['SIMULATION']['points_per_pfr'] = str(pointsPFR)
	run(config_parser)

	# get openccm results for c and t vectors
	outputs = os.listdir(os.getcwd()+'/output_ccm')
	t_vec = np.load(os.getcwd()+'/output_ccm/pfr_t.npy')
	c_vec = np.load(os.getcwd()+'/output_ccm/pfr_concentrations.npy')

	# get V arrays for function calls
	V_inlet = np.zeros((len(t_vec), ))
	V_mid = np.zeros((len(t_vec), )) + 0.5
	V_outlet = np.ones((len(t_vec), ))

	# organize analytical results, use same time domain points as openccm for error calculation purposes
	Ca_inlet_an		= Ca_func(V_inlet,  t_vec)
	Ca_mid_an		= Ca_func(V_mid,    t_vec)
	Ca_outlet_an 	= Ca_func(V_outlet, t_vec)
	Cb_inlet_an 	= Cb_func(V_inlet,  t_vec)
	Cb_mid_an 		= Cb_func(V_mid,    t_vec)
	Cb_outlet_an 	= Cb_func(V_outlet, t_vec)

	# organize openccm results, use same time domain points as openccm for error calculation purposes
	mid = c_vec.shape[1] // 2

	Ca_inlet_num	= c_vec[0,   0, :]
	Ca_mid_num		= c_vec[0, mid, :]
	Ca_outlet_num	= c_vec[0,  -1, :]
	Cb_inlet_num	= c_vec[1,   0, :]
	Cb_mid_num		= c_vec[1, mid, :]
	Cb_outlet_num	= c_vec[1,  -1, :]

	# group all results to simplify plotting code
	inlet = [(Ca_inlet_an, Ca_inlet_num), (Cb_inlet_an, Cb_inlet_num)]
	mid = [(Ca_mid_an, Ca_mid_num), (Cb_mid_an, Cb_mid_num)]
	outlet = [(Ca_outlet_an, Ca_outlet_num), (Cb_outlet_an, Cb_outlet_num)]
	plotlist = [inlet, mid, outlet]
	plotnames = ['inlet', 'midpoint', 'outlet']

	# error computing function
	def err_func(an, num, i_0):
		return np.sum(np.abs(an[i_0:] - num[i_0:])) / len(t_vec[i_0:])

	# Enforce bounds for plot so that it's consistent for all figures
	y_lim = [-0.05, 2.05]

	# plot results
	for i, toplot in enumerate(plotlist):
		plt.figure()

		# Because the sampling is heavily skewed towards the beginning, we don't want to sample them all since it will skew our results
		i_start = min(
			np.argmax(abs(toplot[0][1]) > 1e-10),
			np.argmax(abs(toplot[1][1]) > 1e-10))

		# get error calculation results
		errA_value = err_func(toplot[0][0], toplot[0][1], i_start)
		errA = (f"err [A]_{plotnames[i]}" + " = {:0.3e}").format(errA_value)

		errB_value = err_func(toplot[1][0], toplot[1][1], i_start)
		errB = (f"err [B]_{plotnames[i]}" + " = {:0.3e}").format(errB_value)

		# Plot all results
		plt.text(0, 1.1, '\n'.join((errA, errB)))
		plt.plot(t_vec, toplot[0][0], color='r', label='[A] analytical')
		plt.scatter(t_vec, toplot[0][1], s=8, color='r', label='[A] openccm')

		plt.plot(t_vec, toplot[1][0], color='b', label='[B] analytical')
		plt.scatter(t_vec, toplot[1][1], s=8, color='b', label='[B] openccm')

		plt.xlabel('Time [s]')
		plt.ylabel('Concentration [M]')
		plt.title(f"{pointsPFR} DoF PFR with Rxn: A -> 2B, @ V = {plotnames[i].capitalize()}")
		plt.ylim(y_lim)
		plt.legend(loc='upper left')

		try:
			plt.savefig(f"analysis/{plotnames[i]}_{pointsPFR}_points.png")
		except:
			os.mkdir("analysis")
			plt.savefig(f"analysis/{plotnames[i]}_{pointsPFR}_points.png")
		#plt.show()
			
		plt.close()

	print('Inlet, midpoint, and outlet concentration plots saved.')

	# Visualize the concentration in the reactor as a function of time
	plt.figure()
	fig, ax = plt.subplots()
	plt.xlabel("Volume Along Reactor [V]")
	plt.ylabel("Concentration [M]")
	plt.title(f"PFR with Rxn: A -> 2B, @ t={t_vec[0]:.4f}")
	V = Vs/(c_vec.shape[1]-1) * np.array([i for i in range(c_vec.shape[1])], dtype=float)
	a_analytic, = ax.plot(V, Ca_func(V, t_vec[0]), 'r')
	a_openccm,  = ax.plot(V, c_vec[0, :, 0], 'ro')
	b_analytic, = ax.plot(V, Cb_func(V, t_vec[0]), 'b')
	b_openccm,  = ax.plot(V, c_vec[1, :, 0], 'bo')
	plt.legend(['[A] analytic', '[A] openccm', '[B] analytic', '[B] openccm'], loc='best')
	plt.ylim(y_lim)

	if save_movie:
		def animate(i_t):
			a_analytic.set_ydata(Ca_func(V, t_vec[i_t]))
			a_openccm. set_ydata(c_vec[0, :, i_t])
			b_analytic.set_ydata(Cb_func(V, t_vec[i_t]))
			b_openccm. set_ydata(c_vec[1, :, i_t])
			plt.title(f"PFR with Rxn: A -> 2B, @ t={t_vec[i_t]:.4f}")

		try:
			writer = animation.writers['ffmpeg'](fps=30, bitrate=1800, codec='libx264')
			ext = 'mp4'
			print('Using ffmpeg writer for animation.')
		except RuntimeError: # if ffmpeg not found locally, use PillowWriter
			print('ffmpeg not found. Resorting to Pillow writer...')
			writer = animation.PillowWriter(fps=30, bitrate=1800, codec='libx264')
			ext = 'gif'

		ani = animation.FuncAnimation(fig, animate, frames=range(len(t_vec)))
		ani.save(f"analysis/transient_{pointsPFR}_points.{ext}", writer, dpi=300)

		plt.close()
		print(f'Transient animation saved as transient_{pointsPFR}_points.{ext} in analysis/.')
	else:
		print('Transient movie simulation not created as directed by pfr_analysis.py.')
