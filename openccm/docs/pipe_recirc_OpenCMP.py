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
The relevant code for this example are found under `examples/OpenCMP/pipe_with_recird_2d`, or online [here](https://github.com/uw-comphys/openccm/tree/main/examples/OpenCMP/pipe_with_recirc_2d).

This example requires the optional [OpenCMP](https://opencmp.io/index.html) dependency installed,
see the [Installation Instructions](installation_guide) for help installing it.

This example makes heavy use of the configuration file interface.

# Overview
This example will provide you with an overview of how to use the major features of OpenCCM:
- Creating a compartmental model from CFD results
- Running simulations on the compartmental model
- Visualizing the compartmental model and the simulation results
- Post-processing the results

# CFD
The resulting steady-state velocity profile needed for compartmentalization is provided in `output/`,
as is the computed residence time distribution for the CFD tracer simulation in `cfd_rtd.py`, in order to speed up the simulation.
Due to size constraints, the raw data for the CFD tracer experiment is not included.
All CFD data can be re-created by running the `run_opencmp.py`.

Doing so will take a few minutes depending on your computer.
When finished you will have an `output/` directory which includes the hydrodynamic data needed to create the
compartmental model and the tracer data needed to compare the compartmental model inert tracer simulation results to.
The data is also saved in `.vtu` format which can be visualized using [ParaView](https://www.paraview.org).

# Compartments with Inert Tracer Simulation
With the CFD simulation performed, a compartmental model can be created.
Two Python scripts are provided for this purpose:
1.  `run_compartment.py`: Creates a compartmental model and runs an inert tracer experiment on it.
    This script is equivalent to running `openccm CONFIG` from a terminal opened in the `pipe_with_recirc_2d/` directory.
2.  `analysis.py`: Creates two compartmental models, one using CSTRs and one using PFRs.
    It then runs an inert tracer experiment on both models, creating the following of residence time distributions
    between the three simulation results.

![](../../examples/OpenCMP/pipe_with_recirc_2d/figures/CFD vs PFR vs CSTR.pdf)

A network diagram of the compartmental model can be created by uncommenting `;network_diagram       = True` in
the config file, `CONFIG`.
Note that this requires the optional [NetworkX](https://networkx.org) dependency.

# Compartments with Reactive Flow Simulation
A second configuration file, `CONFIG_W_RXN` uses the same compartmentalization parameters but replaces the
inert tracer with a reactive flow. Modelling the following reaction:

$$2\textrm{NaCl}                + \textrm{CaCO}_3 \xrightarrow[]{0.05}   \textrm{Na}_2\textrm{CO}_3 + \textrm{CaCl}_2$$
$$ \textrm{Na}_2\textrm{CO}_3   + \textrm{CaCl}_2 \xrightarrow[]{2}     2\textrm{NaCl}              + \textrm{CaCO}_3 $$

Note that the values for the rate constants were chosen for demonstration purposes only and are
not the real ones for this reaction.

To perform this compartmentalization and simulation run the `run_compartment_w_rxn.py` script.
"""