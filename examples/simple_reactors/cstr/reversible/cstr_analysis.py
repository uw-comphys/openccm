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
import matplotlib.pyplot as plt 

from openccm import run, ConfigParser

# Reaction rate constants. NOTE: rate constants are not synced with reactions config (manual check is needed to ensure they are identical)
# for large k:
kf = 1 # s^-1
kr = 1 # s^-1
# for small k:
# kf = 1e-2
# kr = 1e-4 

# Initial and "boundary" conditions (inlet feed rates)
tau = 10
a0, b0 = 0, 0
aIN, bIN = 1, 0

# Let C be the total mass, C = A+B.
c0 = a0 + b0
cIN = aIN + bIN

# Useful constants based on hand written derivation
alpha = -kf - 1/tau 
alphaSTAR = kr + kf + 1/tau
omega = a0 - (kr*cIN + aIN/tau)/alphaSTAR - kr*(c0-cIN)/(kr+kf)

# Define analytical solutions
A_t = lambda t: omega*np.exp(-alphaSTAR*t) + kr*(c0-cIN)*np.exp(-t/tau)/(kr + kf) + (kr*cIN + aIN/tau)/alphaSTAR

C_t = lambda t: cIN + (c0 - cIN)*np.exp(-t/tau)

B_t = lambda t: C_t(t) - A_t(t)

# Now get numerical results from openccm
os.chdir(Path(__file__).parents[0])
config_parser = ConfigParser('CONFIG')

for tolerance in [1e-3, 1e-7, 1e-10, 1e-13]:
	
	# Runtime change of tolerance parameters regardless of what configuration file was given.
	tolerance = '{:.0e}'.format(tolerance)
	config_parser['SIMULATION']['atol'] = tolerance
	config_parser['SIMULATION']['rtol'] = tolerance
	run(config_parser)
	
	# get openccm results for c and t vectors
	outputs = os.listdir(os.getcwd()+'/output_ccm')
	t_vec = np.load(os.getcwd()+'/output_ccm/cstr_t.npy')
	c_vec = np.load(os.getcwd()+'/output_ccm/cstr_concentrations.npy')
	c_vec = np.squeeze(c_vec) # remove DOF (just 1 for CSTR)

	# error computation
	err_func = lambda an, num: np.sum(abs(an-num)) / len(t_vec)

	# organize results, use same time domain points as openccm for error calculation purposes
	A_analytical = A_t(t_vec)
	A_openccm = c_vec[0]
	A_error = err_func(A_analytical, A_openccm)

	B_analytical = B_t(t_vec)
	B_openccm = c_vec[1]
	B_error = err_func(B_analytical, B_openccm)

	# Plot all results
	plt.figure()
	textstr = '\n'.join(("err [A] = {:0.3e}".format(A_error),
						 "err [B] = {:0.3e}".format(B_error)))
	plt.text(9,0.2,textstr)

	plt.plot(t_vec, A_analytical, color='r', label='[A] analytical')
	plt.scatter(t_vec, A_openccm, s=8, color='r', label='[A] openccm')

	plt.plot(t_vec, B_analytical, color='b', label='[B] analytical')
	plt.scatter(t_vec, B_openccm, s=8, color='b', label='[B] openccm')

	plt.xlabel('Time [s]')
	plt.ylabel('Concentration [M]')
	plt.title(f'(kf, kr) = ({kf:1.0e}, {kr:1.0e}) | rtol & atol={tolerance}')
	plt.legend()
	try:
		plt.savefig(f"analysis/tol_{tolerance}_kr_{kr:1.0e}_kf_{kf:1.0e}.png")
	except:
		os.mkdir("analysis")
		plt.savefig(f"analysis/tol_{tolerance}_kr_{kr:1.0e}_kf_{kf:1.0e}.png")
	#plt.show()
