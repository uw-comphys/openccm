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

r"""
# Overview
This example is the OpenFOAM equivalent of the pipe_with_recird_2d OpenCMP example.

# CFD
The pre-computed hydrodynamic results are included in the `0/` directory in order to speed up running of the example
and to avoid the need of installing OpenFOAM.

However, due to large file sizes, the results of the inert tracer simulation with OpenFOAM are *not* pre-computed.
Instead, a `README.md` has been included with instructions of how to generate the data. This data is optional and only
required for comparing the compartmental model results to OpenFOAM results.

# Compartmental Modelling
Two scripts are provided, each serving different purposes:
1.  `run_compartment.py`: Create a compartmental model and run a single inert tracer simulation
    This script is equivalent to running `openccm CONFIG` from a terminal opened in the `pipe_with_recirc/` directory.
2.  `analysis.py`: Create two compartmental models (one CSTR-based and one PFR-based), run an inert tracer simulations
    for each, and compare the results to the OpenFOAM tracer simulation. **Note:** In order to use this option, you
    need to have generated the OpenFOAM tracer results, see instructions above.
"""