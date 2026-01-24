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
            Dict[int, List[int]],
            List[List[Tuple[float, int]]]
        ]:
    """
    Helper function to clean up call sites.
    Calls `openccm.compartment_models.pfr.create_pfr_network` or `openccm.compartment_models.cstr.create_cstr_network`
    based on the value of `model`.

    Parameters
    ----------
    * model:                The kind of model to use for building the network, 'pfr' or 'cstr'.
    * compartments:         The element IDs that make up each compartment, indexed by compartment ID.
    * compartment_network:  A dictionary representation of the compartments in the network.
                            Keys are compartment IDs, and whose values are dictionary.
                            -   For each of those dictionaries, the keys are the index of a neighboring compartment
                                and the values are another dictionary.
                                -   For each of those dictionaries, the keys are the index of the bounding entity
                                    between the two compartments, and the values Tuples.
                                    - The 1st is the index of the element upwind of that boundary facet.
                                    - The 2nd is the outward facing unit normal for that boundary facet.
    * mesh:                 The CMesh from which the compartments were built.
    * dir_vec:              Numpy array of direction vectors, row i is for element i.
    * flows_and_upwind:     2D object array indexed by facet ID.
                            - 1st column is volumetric flowrate through facet.
                            - 2nd column is a flag indicating which of a facet's elements are upwind of it.
                                - 0, and 1 represent the index into mesh.facet_elements[facet]
                                - -1 is used for boundary elements to represent
    * config_parser:        The OpenCCM ConfigParser to use.

    Returns
    -------
    1. connections:                 A dictionary representing the model network.
                                    The keys are the IDs of each model, the values are tuples of two dictionaries.
                                    - The first dictionary is for the inlet(s) of the model.
                                    - The second dictionary is for the outlet(s) of the model.
                                    For both dictionaries, the key is the connection ID
                                    and the value is the ID of the model on the other end of the connection.
    2. volumes:                     A numpy array of the volume of each model indexed by its ID.
    3. volumetric_flows:            A numpy array of the volumetric flowrate through each connection indexed by its ID.
    4. compartment_to_model_map:    A map between a compartment ID and the model IDs of all models in it.
                                    The PFR IDs are stored in the order in which they appear
                                    (i.e. the most upstream model is first, and the most downstream model is last).
                                    For CSTRs, this will be a 1-to-1 mapping since it's 1 CSTR per compartment.
    5. model_to_element_map:        A mapping between model ID and an ordered list of tuples containing:
                                    (distance_along_compartment, element_id) where distance along compartment is a
                                    float in the range of [0, 1].
    """
    if model == 'pfr':
        return create_pfr_network(compartments, compartment_network, mesh, flows_and_upwind, dir_vec, config_parser)
    else:
        return create_cstr_network(compartments, compartment_network, mesh, flows_and_upwind, dir_vec, config_parser)
