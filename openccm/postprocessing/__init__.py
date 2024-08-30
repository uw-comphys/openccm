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
Functions related to post-processing the compartmental model and any results from performing simulations on them.
"""
from typing import Optional, Tuple, Dict, List, Set

import numpy as np

from ..config_functions import ConfigParser
from ..mesh import CMesh
from .vtu_output import cstrs_to_vtu_and_save_opencmp, pfrs_to_vtu_and_save_opencmp, cstrs_to_vtu_and_save_openfoam, \
                        create_element_label_gfu, create_compartment_label_gfu, label_compartments_openfoam
from .analysis import network_to_rtd, plot_results, visualize_model_network


def convert_to_vtu_and_save(OpenCMP:        bool,
                            model:          str,
                            system_results: Tuple[
                                                np.ndarray,
                                                np.ndarray,
                                                Dict[int, List[Tuple[int, int]]],
                                                Dict[int, List[Tuple[int, int]]]],
                            model_network:  Tuple[
                                                Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
                                                np.ndarray,
                                                np.ndarray,
                                                Dict[int, List[int]]],
                            compartments:   Dict[int, Set[int]],
                            config_parser:  ConfigParser,
                            cmesh:          CMesh,
                            OpenCMP_mesh:   Optional['ngsolve.Mesh'],
                            n_vec:          Optional[np.ndarray]) -> None:
    """
    Helper function for cleaning up the call site.

    Parameters
    ----------
    * OpenCMP:          Bool indicating if this output format should be OpenCMP (true) or OpenFOAM (false).
    * model:            The kind of model used for the network: pfr or cstr.
    * system_results:   The simulation results.
    * model_network:    The model (pfr or cstr) network on which the simulation was run.
    * compartments:     Mapping between compartment ID and the set of elements IDs which make it up.
    * config_parser:    The OpenCCM ConfigParser used for the simulation.
    * cmesh:            The CMesh object from which compartmental model was built.
    * OpenCMP_mesh:     The OpenCMP mesh on which to output results.
    * n_vec:            The direction vector indexed by mesh element ID.
    """
    if OpenCMP:
        if model == 'pfr':
            pfrs_to_vtu_and_save_opencmp(system_results, model_network, compartments, config_parser, OpenCMP_mesh, n_vec)
        else:
            cstrs_to_vtu_and_save_opencmp(system_results, compartments, config_parser, OpenCMP_mesh)
    else:
        if model == 'pfr':
            raise NotImplementedError("PFR visualization for OpenFOAM input not yet implemented")
        else:
            cstrs_to_vtu_and_save_openfoam(system_results, compartments, config_parser, cmesh)
