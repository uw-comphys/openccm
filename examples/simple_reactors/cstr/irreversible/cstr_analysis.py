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
k1 = 0.5 # hr^-1
k2 = 0.2 # hr^-1
tau = 10
k1T = k1*tau

# Ca(t) analytical
Ca0 = 2 #mol/L the initial condition!
betaA = (k1T + 1) / tau

Ca_func = lambda t: (Ca0 / (1+k1T) ) * (1+ k1T*np.exp(-t*betaA))

# Cb(t) analytical
a1 = (k1*Ca0) / (k1T +1)
a2 = a1 * k1T
betaB = ((k2*tau) + 1) / tau

Cb_func = lambda t: (a1/betaB) * (1-np.exp(-betaB*t)) + (a2/(betaB-betaA)) * (np.exp(-betaA*t) - np.exp(-betaB*t))

# Cc(t) analytical
F = lambda b: 1 / ( (1/tau) - b)
A = (k2*a1/betaB)*(F(betaB)-tau) + (a2*k2/(betaB-betaA))*(F(betaB) - F(betaA))

Cc_func = lambda t: A*np.exp(-t/tau) + (k2*a1/betaB)*(tau - F(betaB)*np.exp(-betaB*t)) + (a2*k2/(betaB-betaA))*( F(betaA)*np.exp(-betaA*t) - F(betaB)*np.exp(-betaB*t) )

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
	Ca_analytical = Ca_func(t_vec)
	Ca_openccm = c_vec[0]
	Ca_err = err_func(Ca_analytical, Ca_openccm)

	Cb_analytical = Cb_func(t_vec)
	Cb_openccm = c_vec[1]
	Cb_err = err_func(Cb_analytical, Cb_openccm)

	Cc_analytical = Cc_func(t_vec)
	Cc_openccm = c_vec[2]
	Cc_err = err_func(Cc_analytical, Cc_openccm)

	# Plot all results
	plt.figure()
	textstr = '\n'.join(("err [A] = {:0.3e}".format(Ca_err),
						 "err [B] = {:0.3e}".format(Cb_err),
						 "err [C] = {:0.3e}".format(Cc_err)))

	plt.plot(t_vec, Ca_analytical, color='r', label='[A] analytical')
	plt.scatter(t_vec, Ca_openccm, s=8, color='r', label='[A] openccm')

	plt.plot(t_vec, Cb_analytical, color='b', label='[B] analytical')
	plt.scatter(t_vec, Cb_openccm, s=8, color='b', label='[B] openccm')

	plt.plot(t_vec, Cc_analytical, color='g', label='[C] analytical')
	plt.scatter(t_vec, Cc_openccm, s=8, color='g', label='[C] openccm')

	plt.xlabel('Time [hr]')
	plt.ylabel('Concentration [M]')
	plt.title(f'rtol & atol = {tolerance}')

	plt.text(1.5,1.5,textstr)
	plt.legend()
	try:
		plt.savefig(f"analysis/tol_{tolerance}.png")
	except:
		os.mkdir("analysis")
		plt.savefig(f"analysis/tol_{tolerance}.png")
	#plt.show()
