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
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from ..config_functions import ConfigParser
from ..mesh import GroupedBCs


def solve_system(
        cstr_network:   Tuple[
                            Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
                            np.ndarray,
                            np.ndarray,
                            Dict[int, List[int]]],
        config_parser:  ConfigParser,
        grouped_bcs:    GroupedBCs) \
        -> Tuple[
            np.ndarray,
            np.ndarray,
            Dict[int, List[Tuple[int, int]]],
            Dict[int, List[Tuple[int, int]]]
        ]:
    """
    Perform a transient simulation, including reaction, on a network of CSTRs.

    Args:
        cstr_network:   A tuple containing, in order (from `create_cstr_network`):
                            1. connections:         A dictionary representing the CSTR network.
                                                    The keys are the IDs of each CSTR, the values are tuples of two dictionaries.
                                                        - The first dictionary is for flows into the CSTR.
                                                        - The second dictionary is for flows out of the CSTR.
                                                    For both dictionaries, the key is the connection ID
                                                    and the value is the ID of the CSTR on the other end of the connection.
                            2. volumes:             A numpy array of the volume of each CSTR indexed by its ID.
                            3. volumetric_flows:    A numpy array of the volumetric flowrate through each connection indexed by its ID.
        config_parser:  OpenCCM ConfigParser for getting settings.
        grouped_bcs:    GroupedBCs object needed to create the boundary conditions.

    Returns:
        c:                  NxMxT numpy array containing the concentration of each species at each CSTR at each point in time.
                            N is number of CSTRs, M is number of species, and T is number of time steps.
        t:                  The times at which the concentrations were saved at, vector of T entries.
        inlet_map:          A map between the inlet ID and the ID of the CSTR(s) connected to it and the connection ID.
                            Key to dictionary is inlet ID, value is a list of tuples.
                            First entry in tuple is the CSTR ID, second value is the connection ID.
        outlet_map:         A map between the outlet ID and the ID of the CSTR(s) connected to it and the connection ID.
                            Key to dictionary is outlet ID, value is a list of tuples.
                            First entry in tuple is the CSTR ID, second value is the connection ID.
    """
    print("Solving simulation")

    t_span          = config_parser.get_list(['SIMULATION', 't_span'],          float)
    first_timestep  = config_parser.get_item(['SIMULATION', 'first_timestep'],  float)
    atol            = config_parser.get_item(['SIMULATION', 'atol'],            float)
    rtol            = config_parser.get_item(['SIMULATION', 'rtol'],            float)
    solver          = config_parser.get_item(['SIMULATION', 'solver'],          str)

    rxn_species     = config_parser.get_list(['SIMULATION', 'specie_names'],    str)

    assert first_timestep > 0

    if os.path.exists('reaction_code_gen.py'):
        os.remove('reaction_code_gen.py')

    connections, volumes, Q_connections, _ = cstr_network

    inlet_map:  Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    outlet_map: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    num_species = len(rxn_species)
    num_cstrs   = len(connections)

    c_shape = (num_species, num_cstrs)

    # Pre-dividing each volumetric flowrate by V in order to speed up ddt
    # Square matrix to multiple concentrations by
    Q_div_v = np.zeros((num_cstrs, num_cstrs))

    Q_weight_inlets: Dict[int, List[float]] = defaultdict(list)
    points_for_bc: Dict[int, List[int]] = defaultdict(list)

    for id_cstr in sorted(connections.keys()):
        inlets, outlets = connections[id_cstr]

        for id_connection, id_cstr_other in inlets.items():
            if id_cstr_other < 0:  # Domain inlets, id_cstr_other is actually id_bc from grouped_bcs.id()
                # Each cstr only has one discretization point, id of that point is id of the cstr
                points_for_bc[id_cstr_other].append(id_cstr)
                Q_weight_inlets[id_cstr_other].append(Q_connections[id_connection] / volumes[id_cstr])
                inlet_map[id_cstr_other].append((id_cstr, id_connection))
            else:
                Q_div_v[id_cstr_other, id_cstr] += Q_connections[id_connection]

        for id_connection, id_cstr_other in outlets.items():
            if id_cstr_other < 0:  # Domain outlets
                outlet_map[id_cstr_other].append((id_cstr, id_connection))
            Q_div_v[id_cstr, id_cstr] -= Q_connections[id_connection]

        Q_div_v[:, id_cstr]        /= volumes[id_cstr]

    from . import load_and_prepare_bc_ic_and_rxn
    reactions, bcs, c0 = load_and_prepare_bc_ic_and_rxn(config_parser,
                                                        c_shape,
                                                        points_per_model=1,
                                                        _ddt_reshape_shape=None,
                                                        inlet_map=inlet_map,
                                                        grouped_bcs=grouped_bcs,
                                                        Q_weight_inlets=Q_weight_inlets,
                                                        points_for_bc=points_for_bc,
                                                        t0=t_span[0])

    args = (Q_div_v, c_shape, reactions, bcs)

    output = solve_ivp(ddt, t_span, c0, method=solver, atol=atol, rtol=rtol, args=args, first_step=first_timestep)
    t: np.ndarray = output.t
    c: np.ndarray = output.y
    c = c.reshape(c_shape + t.shape)  # (num_species, num_points, num_timesteps)

    print("Done solving simulation")

    return c, t, dict(inlet_map), dict(outlet_map)


@njit
def ddt(t: float,
        c: np.ndarray,
        Q_div_v: np.ndarray,
        c_shape: Tuple[int, ...],
        reactions,
        bcs
        ) -> np.ndarray:
    """
    Models the change in concentration within each CSTR as:
        dcdt = c @ A + b + r
    where
    - A = Q / V             for inlet flows from other CSTRs and all outlet flows
    - b = (Q / V)|inlets    for domain inlets
    - r = R                 for reactions

    Note that the division by volume has been performed ahead of time.

    Args:
        t:                  The current time
        c:                  The concentration indexed by CSTR ID
        Q_div_v:            The sum of volumetric flowrates
        c_shape:            Tuple indicating the shape of c (num_cstr, num_species)
        reactions:          The reactions function in generated_code.py containing system of equations
        bcs:                The boundary conditions on domain inlets (all others don't matter).
    """
    c = c.reshape(c_shape)

    # Inlets between CSTRs and all outlets
    _ddt = c @ Q_div_v
    # Domain inlets
    bcs(t, _ddt)
    # Reactions
    reactions(c, _ddt)

    return _ddt.ravel()

