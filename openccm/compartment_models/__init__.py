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
All functions related to take a network of compartments and convert them into a network of CSTRs or PFRs.
"""

from typing import Dict, Tuple, List, Set

import numpy as np

from .cstr import create_cstr_network, connect_cstr_compartments
from .pfr import create_pfr_network, connect_pfr_compartments
from ..config_functions import ConfigParser
from ..mesh import CMesh


def create_model_network(model:                 str,
                         compartments:          Dict[int, Set[int]],
                         compartment_network:   Dict[int, Dict[int, Dict[int, int]]],
                         mesh:                  CMesh,
                         dir_vec:               np.ndarray,
                         flows_and_upwind:      np.ndarray,
                         config_parser:         ConfigParser) \
        -> Tuple[
            Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
            np.ndarray,
            np.ndarray,
            Dict[int, List[int]]
        ]:
    """
    Helper function to clean up call sites.
    Calls `openccm.compartment_models.pfr.create_pfr_network` or `openccm.compartment_models.cstr.create_cstr_network`
    based on the value of `model`.

    Parameters
    ----------
    * model:                The kind of model to use for building the network, 'pfr' or 'cstr'.
    * compartments:         The element IDs that make up each compartment, indexed by compartment ID.
    * compartment_network:  The compartment_network created from the compartments.
    * mesh:                 The CMesh from which the compartments were built.
    * dir_vec:              The direction vector in each mesh element, indexed by element ID.
    * flows_and_upwind:     The flow through each ID and indicator of which way is upwind, indexed by facet ID.
    * config_parser:        OpenCCM ConfigParser to use.

    Returns
    -------
    * model_network: The created network object.
    """
    if model == 'pfr':
        return create_pfr_network(compartments, compartment_network, mesh, flows_and_upwind, dir_vec, config_parser)
    else:
        return create_cstr_network(compartments, compartment_network, mesh, flows_and_upwind, dir_vec, config_parser)
