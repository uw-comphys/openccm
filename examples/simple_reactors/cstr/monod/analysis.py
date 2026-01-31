########################################################################################################################
# Copyright 2024 the authors (see AUTHORS file for full list).                                                         #
#                                                                                                                      #
#                                                                                                                      #
# This file is part of OpenCCM.                                                                                        #
#                                                                                                                      #
#                                                                                                                      #
# OpenCCM is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public  #
# License as published by the Free Software Foundation,either version 2.1 of the License, or (at your option)          #
# any later version.                                                                                                   #
#                                                                                                                      #
# OpenCCM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied        #
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                                     #
# See the GNU Lesser General Public License for more details.                                                          #
#                                                                                                                      #
# You should have received a copy of the GNU Lesser General Public License along with OpenCCM. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################

import os 
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt 

from openccm import run, ConfigParser

if __name__ == '__main__':
	# Calculate steady_state values
	Y_XS	= 0.5
	K 		= 0.25
	tau		= 10
	Q		= 0.1

	S_in	= 1
	S_ss	= K / (Y_XS * tau - 1)
	X_ss	= Q * (K + S_ss) / S_ss * (S_in - S_ss)

	run('CONFIG')

	outputs = os.listdir(os.getcwd()+'/output_ccm')
	t_vec = np.load(os.getcwd()+'/output_ccm/cstr_t.npy')
	c_vec = np.load(os.getcwd()+'/output_ccm/cstr_concentrations.npy')
	c_vec = np.squeeze(c_vec) # remove DOF (just 1 for CSTR)

	# organize results, use same time domain points as openccm for error calculation purposes
	X, S = c_vec[0], c_vec[1]

	# Plot all results
	plt.figure()
	plt.hlines(X_ss, t_vec[0], t_vec[-1], linestyles='dashed', color='r')
	plt.plot(t_vec, X, color='r', label='[X]')
	plt.hlines(S_ss, t_vec[0], t_vec[-1], linestyles='dashed', color='b')
	plt.plot(t_vec, S, color='b', label='[S]')

	plt.xlabel('Time [hr]')
	plt.ylabel('Concentration [M]')

	plt.legend()
	plt.show()
	Path('analysis').mkdir(exist_ok=True)
	plt.savefig(f"analysis/monod.png")
