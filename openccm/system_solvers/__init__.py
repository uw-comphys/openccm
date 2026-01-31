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
All functions related to running a simulation on the PFR/CSTR network.
"""

import os
import sys
import importlib
from typing import Tuple, Dict, List, Optional, Callable

import numpy as np
import sympy as sp

from .cstr_system import solve_system as solve_cstr
from .pfr_system import solve_system as solve_pfr
from .. import ConfigParser
from ..mesh import CMesh, create_dof_to_element_map


def solve_system(model:         str,
                 model_network: Tuple[
                                    Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
                                    np.ndarray,
                                    np.ndarray,
                                    Dict[int, List[int]]],
                 config_parser: ConfigParser,
                 cmesh:         CMesh) \
        -> Tuple[
            np.ndarray,
            np.ndarray,
            Dict[int, List[Tuple[int, int]]],
            Dict[int, List[Tuple[int, int]]]]:
    """
    Wrapper function to clean up call sites.

    Calls either `openccm.system_solvers.cstr_system.solve_system` or `openccm.system_solvers.pfr_system.solve_system`
    based on the value of `model`.

    Parameters
    ----------
    * model:            String indicating which model (pfr vs cstr) is being used.
    * model_network:    Network of PFRs/CSTR. See `openccm.compartment_models.pfr.create_pfr_network`
                        or `openccm.compartment_models.cstr.create_cstr_network` for more details.
    * config_parser:    OpenCCM ConfigParser used for getting settings.
    * cmesh:            The CMesh from which the model being simulated was derived.

    Returns
    -------
    * t:            The times at which the ODE was solved (controlled by solve_ivp)
    * c:            The corresponding values for time t.
    * inlet_map:    Map between a domain inlet IDs and the PFRs/CSTRs that they're connected to.
                    Keys are domain inlet IDs, values are a list of tuples.
                    - The first entry in the tuple is the PFR/CSTR id,
                    - The second is the id of the connection between the inlet and the PFR/CSTR.
    * outlet_map:   Map between a domain outlet IDs and the PFRs/CSTRs that they're connected to.
                    Keys are domain inlet IDs, values are a list of tuples.
                    - The first entry in the tuple is the PFR/CSTR id,
                    - The second is the id of the connection between the outlet and the PFR/CSTR.
    """
    if model == 'pfr':
        return solve_pfr(model_network, config_parser, cmesh)
    else:
        return solve_cstr(model_network, config_parser, cmesh)


def load_and_prepare_bc_ic_and_rxn(config_parser:               ConfigParser,
                                   c_shape:                     Tuple[int, int],
                                   points_per_model:            int,
                                   _ddt_reshape_shape:          Optional[Tuple[int, int, int]],
                                   cmesh:                       CMesh,
                                   Q_weight_inlets:             Dict[int, List[float]],
                                   model_volumes:               np.ndarray,
                                   points_for_bc:               Dict[int, List[int]],
                                   t0:                          float,
                                   model_to_element_map:        List[List[Tuple[float, int]]],
                                   connected_to_another_inlet:  Optional[np.ndarray] = None,
                                   Q_weight:                    Optional[np.ndarray] = None
                                   ) -> Tuple[Callable, Callable, np.ndarray, Dict[int, Dict[int, sp.Expr]]]:
    """
    Wrapper function for creating the initial condition array and applying the initial and boundary conditions to it.

    Parameters
    ----------
    * config_parser:                OpenCCM ConfigParser used for getting settings.
    * c_shape:                      The shape for the concentration array at each timestep (num_species, num_points).
    * points_per_model:             Number of discretization points per model (1 for CSTR, >2 for PFR).
    * _ddt_reshape_shape:           Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction
                                    occurring at it. Used by `generate_reaction_system`.
    * cmesh:                        The CMesh from which the model being simulated was derived.
    * Q_weight_inlets:              Lookup all Q_connection / Q_reactor_total for each inlet BC connection.
    * model_volumes:                The volume of each PFR/CSTR, indexed by their ID.
    * points_for_bc:                Lookup for all discretization points on an inlet BC, index by BC ID. Same ordering as Q_weight_inlets.
    * t0:                           Initial timestep, needed in order to evaluate the BCs at the first timestep if a PFR is used.
    * model_to_element_map:         Mapping between model ID and a list of ordered tuples (distance_in_model, element ID)
    * connected_to_another_inlet:   For a PFR network, this specif
    * Q_weight:

    Returns
    -------
    * rxns: Numba JITd function for applying reactions, as described by `generate_reaction_system`.
    * bcs:  Numba JITd function for applying the boundary condition, as described by `create_boundary_conditions`.
    * c0:   The initial conditions for the simulation. The

    """
    from .boundary_and_initial_conditions import load_initial_conditions, create_boundary_conditions
    from .reactions import generate_reaction_system

    dof_to_element_map = create_dof_to_element_map(model_to_element_map, points_per_model)

    c0 = np.zeros(c_shape)

    load_initial_conditions(config_parser, c0, cmesh, dof_to_element_map, points_per_model, connected_to_another_inlet, Q_weight)
    bc_file_name, bc_dict_for_eval = create_boundary_conditions(c0, config_parser, Q_weight_inlets, points_for_bc, t0, points_per_model, cmesh, dof_to_element_map, model_volumes)
    rxn_file_name = generate_reaction_system(config_parser, dof_to_element_map, _ddt_reshape_shape)

    c0 = c0.ravel()  # Required since solve_ivp needs 1D array

    # Add working directory to path so that imports can be found
    tmp_directory_abs_path = os.getcwd() + '/' + config_parser.get_item(['SETUP', 'tmp_folder_path'], str)
    sys.path.append(tmp_directory_abs_path)

    # If the reaction module has previous been imported, need to remove it for the import statement to do anything.
    if rxn_file_name in sys.modules:
        sys.modules.pop(rxn_file_name)
    rxn_module = importlib.import_module(rxn_file_name)

    # If the bc module has previous been imported, need to remove it for the import statement to do anything.
    if bc_file_name in sys.modules:
        sys.modules.pop(bc_file_name)
    bc_module = importlib.import_module(bc_file_name)

    # Remove the working directory to not pollute sys.path
    # so that consecutive runs from the same Python process find the correct files to import.
    # Don't pop last just in case another import happened inbetween
    sys.path.pop(sys.path.index(tmp_directory_abs_path))

    # noinspection PyUnresolvedReferences
    return rxn_module._reactions, bc_module._boundary_conditions, c0, bc_dict_for_eval
