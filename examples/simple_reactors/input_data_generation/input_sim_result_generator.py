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


# Create a input OpenCMP result to be converted into a single CSTR/PFR
import os
from pathlib import Path

import ngsolve as ngs
from netgen.read_gmsh import ReadGmsh

Path('output/ins_sol/').mkdir(parents=True, exist_ok=True)

# Define either the velocity or the space time, depending on what's more natural for
# your particular case
# velocity = 1
# tau = 1.0 / velocity
# print(f"Space time is {tau}.")
tau = 10
velocity = 1.0 / tau
print(f"Velocity is {velocity}")

mesh = ngs.Mesh(ReadGmsh("v=1.msh"))

fes_u = ngs.VectorH1(mesh, order=3)
fes_p = ngs.H1(mesh, order=2)
_fes = [fes_u, fes_p]
fes = ngs.FESpace(_fes)

gfu = ngs.GridFunction(fes)
gfu.components[0].Interpolate(ngs.CoefficientFunction((velocity, 0)))
gfu.Save('output/ins_sol/ins_0.0.sol')
ngs.VTKOutput(ma=mesh, coefs=[c for c in gfu.components], names=['velocity', 'fake pressure'],
              filename='output/ins_sol/ins_0.0', subdivision=1).Do()
