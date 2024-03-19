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

from typing import Tuple, Dict, List, Optional, Callable

import numpy as np

from .cstr_system import solve_system as solve_cstr
from .pfr_system import solve_system as solve_pfr
from .. import ConfigParser
from ..mesh import GroupedBCs


def solve_system(model: str, model_network, config_parser, grouped_bc) \
        -> Tuple[
            np.ndarray,
            np.ndarray,
            Dict[int, List[Tuple[int, int]]],
            Dict[int, List[Tuple[int, int]]]]:
    """Wrapper function to """
    if model == 'pfr':
        return solve_pfr(model_network, config_parser, grouped_bc)
    else:
        return solve_cstr(model_network, config_parser, grouped_bc)


def load_and_prepare_bc_ic_and_rxn(config_parser:       ConfigParser,
                                   c_shape:             Tuple[int, int],
                                   points_per_model:    int,
                                   _ddt_reshape_shape:  Optional[Tuple[int, int, int]],
                                   inlet_map:           Dict[int, List[Tuple[int, int]]],
                                   grouped_bcs:         GroupedBCs,
                                   Q_weight_inlets:     Dict[int, List[float]],
                                   points_for_bc:       Dict[int, List[int]],
                                   t0:                  float) -> Tuple[Callable, Callable, np.ndarray]:
    """
    Wrapper function for creating the initial condition array, applying the initial conditions to it

    Args:
        config_parser:      OpenCCM ConfigParser used for getting settings.
        c_shape:            The shape for the concentration array at each timestep (num_species, num_points).
        points_per_model:   Number of discretization points per model (1 for CSTR, >2 for PFR).
        _ddt_reshape_shape: Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction
                            occurring at it. Used by `generate_reaction_system`.
        inlet_map:          Map between a domain inlet IDs and the PFRs that they're connected to.
                                Keys are domain inlet IDs, values are a list of tuples.
                                    The first entry in the tuple is the PFR id,
                                    The second is the id of the connection between the inlet and the PFR.
        grouped_bcs:        Helper class used for consistent numbering and lookup of boundary conditions by name.
        Q_weight_inlets:    Lookup all Q_connection / Q_reactor_total for each inlet BC connection.
        points_for_bc:      Lookup for all discretization points on an inlet BC, index by BC ID. Same ordering as Q_weight_inlets.
        t0:                 Initial timestep, needed in order to evaluate the BCs at the first timestep if a PFR is used.

    Returns:
        rxns:   Numba JITd function for applying reactions, as described by `generate_reaction_system`.
        bcs:    Numba JITd function for applying the boundary condition, as described by `create_boundary_conditions`.
        c0:     The initial conditions for the simulation. The

    """
    from .boundary_and_initial_conditions import load_initial_conditions, create_boundary_conditions
    from .reactions import generate_reaction_system

    c0 = np.zeros(c_shape)

    load_initial_conditions(config_parser, c0)
    create_boundary_conditions(c0, config_parser, inlet_map, grouped_bcs, Q_weight_inlets, points_for_bc, t0, points_per_model)
    generate_reaction_system(config_parser, _ddt_reshape_shape)

    c0 = c0.ravel()  # Required since solve_ivp needs 1D array

    import sys

    # Add working directory to path so that imports can be found
    working_directory_abs_path = sys.path[0] + '/' + config_parser.get_item(['SETUP', 'working_directory'], str)
    sys.path.append(working_directory_abs_path)

    # If the reaction module has previous been imported, need to remove it for the import statement to do anything.
    if 'reaction_code_gen' in sys.modules:
        sys.modules.pop('reaction_code_gen')
    # noinspection PyUnresolvedReferences
    import reaction_code_gen

    # If the bc module has previous been imported, need to remove it for the import statement to do anything.
    if 'bc_code_gen' in sys.modules:
        sys.modules.pop('bc_code_gen')
    # noinspection PyUnresolvedReferences
    import bc_code_gen

    # Remove the working directory to not pollute sys.path
    # so that consecutive runs from the same Python process find the correct files to import.
    # Don't pop last just in case another import happened inbetween
    sys.path.pop(sys.path.index(working_directory_abs_path))

    return reaction_code_gen.reactions, bc_code_gen.boundary_conditions, c0
