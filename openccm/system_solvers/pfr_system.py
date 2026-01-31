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
The functions required for solving a simulation on a PFR network.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

from .helper_functions import generate_t_eval
from ..config_functions import ConfigParser
from ..mesh import GroupedBCs, CMesh


def solve_system(
        pfr_network:    Tuple[
                            Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
                            np.ndarray,
                            np.ndarray,
                            Dict[int, List[int]],
                            List[List[Tuple[float, int]]]
                        ],
        config_parser:  ConfigParser,
        cmesh:          CMesh,
) -> Tuple[np.ndarray,
           np.ndarray,
           Dict[int, List[Tuple[int, int]]],
           Dict[int, List[Tuple[int, int]]]]:
    """
    Function to solve the system. Call with the network representation of the compartment model and it will
    Generate the numerical system to solve, solve it in time, and return the results.

    Parameters
    ----------
    * pfr_network:      The network representation of the PFR as produced by `openccm.compartment_models.pfr.create_pfr_network`.
        1. connections:             A dictionary representing the PFR network.
                                    The keys are the IDs of each PFR, the values are tuples of two dictionaries.
                                        - The first dictionary is for the inlet(s) of the PFR.
                                        - The second dictionary is for the outlet(s) of the PFR.
                                    For both dictionaries, the key is the connection ID
                                    and the value is the ID of the PFR on the other end of the connection.
        2. volumes:                 A numpy array of the volume of each PFR indexed by its ID.
        3. volumetric_flows:        A numpy array of the volumetric flowrate through each connection indexed by its ID.
        4. compartment_to_pfr_map:  A map between a compartment ID and the PFR IDs of all PFRs in it.
                                    The PFR IDs are stored in the order in which they appear
                                    (i.e. the most upstream PFR is first, and the most downstream PFR is last).
    * config_parser:    OpenCCM ConfigParser used for getting settings.
    * cmesh:            The CMesh from which the PFR network was created.

    Returns
    -------
    * t:            The times at which the ODE was solved (controlled by solve_ivp)
    * c:            The corresponding values for time t.
    * inlet_map:    Map between a domain inlet IDs and the PFRs that they're connected to.
                    Keys are domain inlet IDs, values are a list of tuples.
                    - The first entry in the tuple is the PFR id,
                    - The second is the id of the connection between the inlet and the PFR.
    * outlet_map:   Map between a domain outlet IDs and the PFRs that they're connected to.
                    Keys are domain inlet IDs, values are a list of tuples.
                    - The first entry in the tuple is the PFR id,
                    - The second is the id of the connection between the outlet and the PFR.
    """
    print("Solving simulation")

    points_per_pfr = config_parser.get_item(['SIMULATION', 'points_per_pfr'],   int)
    t_span         = config_parser.get_list(['SIMULATION', 't_span'],           float)
    first_timestep = config_parser.get_item(['SIMULATION', 'first_timestep'],   float)
    atol           = config_parser.get_item(['SIMULATION', 'atol'],             float)
    rtol           = config_parser.get_item(['SIMULATION', 'rtol'],             float)
    solver         = config_parser.get_item(['SIMULATION', 'solver'],           str)

    rxn_species    = config_parser.get_list(['SIMULATION', 'specie_names'],     str)

    assert first_timestep > 0
    assert points_per_pfr >= 2

    t_eval = generate_t_eval(config_parser)

    connections, volumes, Q_connections, _, pfr_to_element_map = pfr_network

    num_species = len(rxn_species)
    num_pfrs    = len(connections)

    delta_vs = volumes / (points_per_pfr - 1)

    # Volumetric flowrate through a PFR, defined as the sum of the inlets
    Q_pfrs = np.zeros((num_pfrs,))
    for id_pfr in range(num_pfrs):
        for id_connection in connections[id_pfr][0].keys():
            Q_pfrs[id_pfr] += Q_connections[id_connection]

    # Pre-divide Q_connections by Q_pfr for the PFR they're going into
    Q_weight = Q_connections.copy()
    for id_pfr, (inlet_info, _) in connections.items():
        for id_connection in inlet_info:
            Q_weight[id_connection] /= Q_pfrs[id_pfr]
    # Validate that Q_weights sum to 1 for each PFR
    for id_pfr in connections:
        total = sum(Q_weight[id_connection] for id_connection in connections[id_pfr][0])
        assert np.isclose(total, 1)

    # Info for connections between PFRs
        # 1. PFR inlet node     (To know what value to assign it to)
        # 2. Connection ID      (For Q_weight)
        # 3. PFR outlet node    (To know what value to assign)
    connected_to_another_inlet              = []
    Q_weight_inlets: Dict[int, List[float]] = defaultdict(list)
    points_for_bc:   Dict[int, List[int]]   = defaultdict(list)
    for id_pfr in range(num_pfrs):
        _inlets = connections[id_pfr][0]
        for id_connection, id_pfr_other_side in _inlets.items():
            if id_pfr_other_side >= 0:  # Connection to another PFR
                connected_to_another_inlet.append(
                    [points_per_pfr * id_pfr,                       # Index of inlet node for this PFR
                     id_connection,                                 # Index of connection between the nodes
                     points_per_pfr * (id_pfr_other_side + 1) - 1]  # Index of outlet node for other PFR
                )
            elif id_pfr_other_side < 0:  # Domain inlet BC
                Q_weight_inlets[id_pfr_other_side].append(Q_weight[id_connection])
                points_for_bc[id_pfr_other_side].append(points_per_pfr * id_pfr)

    # Convert to numpy array for numba reasons
    # Need the check to handle the special case of a single PFR since numba can't handle it otherwise
    if len(connected_to_another_inlet) > 0:
        connected_to_another_inlet = np.array(connected_to_another_inlet, dtype=int)
    else:
        connected_to_another_inlet = np.array([[-1, -1, -1]], dtype=int)

    c_shape = (num_species, num_pfrs * points_per_pfr)
    # Pre-compute Q/∆V and call it _ddt0.
    # Q/∆V comes from discretizing the spatial derivative
    _ddt0 = np.zeros(c_shape)
    for id_pfr in range(num_pfrs):
        # NOTE: 1st point of each PFR (the inlet) is handled separately below
        p_s = points_per_pfr * id_pfr + 1    # +1 since the first point in the PFR gets handled below
        p_e = points_per_pfr * (id_pfr + 1)  # No -1 since we want the range inclusive of both ends
        _ddt0[:, p_s:p_e] = -Q_pfrs[id_pfr] / delta_vs[id_pfr]

    all_inlet_ids = points_per_pfr * np.arange(0, num_pfrs, dtype=int)

    inlet_map:  Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    outlet_map: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for pfr in connections.keys():
        inlets  = connections[pfr][0]
        outlets = connections[pfr][1]

        for id_connection, id_pfr_other_side in inlets.items():
            if id_pfr_other_side < 0:  # Domain inlets
                inlet_map[id_pfr_other_side].append((pfr, id_connection))

        for id_connection, id_pfr_other_side in outlets.items():
            if id_pfr_other_side < 0:  # Domain outlets
                outlet_map[id_pfr_other_side].append((pfr, id_connection))

    from . import load_and_prepare_bc_ic_and_rxn
    reactions, bcs, c0, _ = load_and_prepare_bc_ic_and_rxn(config_parser,
                                                           c_shape,
                                                           points_per_model=points_per_pfr,
                                                           _ddt_reshape_shape=(num_species, num_pfrs, points_per_pfr),
                                                           cmesh=cmesh,
                                                           Q_weight_inlets=Q_weight_inlets,
                                                           model_volumes=volumes,
                                                           points_for_bc=points_for_bc,
                                                           t0=t_span[0],
                                                           model_to_element_map=pfr_to_element_map,
                                                           connected_to_another_inlet=connected_to_another_inlet,
                                                           Q_weight=Q_weight)

    args = (Q_weight, _ddt0, connected_to_another_inlet, all_inlet_ids, c_shape, reactions, bcs)
    output = solve_ivp(ddt, t_span, c0, method=solver, atol=atol, rtol=rtol, args=args, first_step=first_timestep, t_eval=t_eval)

    ts: np.ndarray = output.t
    c:  np.ndarray = output.y
    c = c.reshape(c_shape + ts.shape)  # (num_species, num_points, num_timesteps)

    # Test to see how well DAE was solved.
    if connected_to_another_inlet[0, 0] > 0:  # If negative, there are no PFR-to-PFR connections
        c_tmp = defaultdict(lambda: np.zeros(num_species))
        abs_error_in_time = []
        for i, t, in enumerate(ts):
            for j in range(connected_to_another_inlet.shape[0]):
                c_tmp[connected_to_another_inlet[j, 0]] += Q_weight[connected_to_another_inlet[j, 1]] * c[:, connected_to_another_inlet[j, 2], i]
            max_err = 0.0
            for index in c_tmp:
                max_err = max(max_err, np.max(np.abs(c_tmp[index] - c[:, index, i])))
                c_tmp[index][:] = 0  # Set to zero so that we can re-use arrays and not re-allocate every iteration.
            abs_error_in_time.append(max_err)
        max_error = max(abs_error_in_time)
        if max_error > atol:
            print(f"WARNING: Maximum absolute error in solving inlet BC for PFRs is {max_error:.3e} but absolute tolerance is {atol:.3e}.")

    print("Done solving simulation")
    return c, ts, dict(inlet_map), dict(outlet_map)


@njit
def ddt(t: float,
        c: np.ndarray,
        Q_weight: np.ndarray,
        _ddt0: np.ndarray,
        connected_to_another_inlet: np.ndarray,
        all_inlet_ids: np.ndarray,
        c_shape: Tuple[int, ...],
        reactions,
        bcs
        ) -> np.ndarray:
    """
    This function calculates the time derivative of the concentration at the different discretization points within
    the reactor.

        dc/dt = ∇·(D∇c) - ∇·(vc) + R
        dc/dt = D*d2c/dx2 - v*dc/dx + R
        dc/dt = D*A*d2c/dV2 - Q*dc/dV + R

    Assuming diffusion is negligible:

        dc/dt = - Q*dc/dV + R

    Discretizing using backwards difference for space:

        dc/dt = - Q * (c_i - c_i-1)/∆V + R

    Refactoring:

        dc/dt = - (Q/∆V) * (c_i - c_i-1) + R

    The convection term is discretized using ONLY 1st order backwards differences.
    - Backwards difference to enforce upwinding
    - 1st order to avoid the spurious oscillations that come with high order schemes (higher order was tested)

    The above equation holds for all spatial locations *except* for those on the inlet of each PFR.
    There, the following equation holds, which would result in a DAE:

        c_i = ∑_j (Q_{j->i} * c_j} / ∑Q_{j->i}

    The time derivative of the equation is taken to avoid using a DAE solver, providing:

        dc_i/dt = ∑_j (Q_weight_j * dc_j/dt)

    where

        Q_weight_j = Q_{j->i} / ∑Q_{j->i}.

    Parameters
    ----------
    * t:                            Time at the current timestep
    * c:                            The state (values at each discretization point)
    * Q_weight:                     (n, 3) Array of info for connections between PFRs
                                    1. PFR inlet node     (To know what value to assign it to)
                                    2. Connection ID      (For Q_weight)
                                    3. PFR outlet node    (To know what value to assign)
    * _ddt0:                        Q/∆V for each degree of freedom precomputed to save time, used as building block
                                    at each timestep for calculating the time derivative.
    * connected_to_another_inlet:   n x 3 array for knowing how to calculate the amount of mass going from one PFR into
                                    another. Each row contains:
                                    1. The index into c for the inlet into which the flow goes.
                                    2. The ID of the connection over which the flow goes.
                                    3. The index into c for the outlet from which the flow comes.
    * all_inlet_ids:                An array containing the index into `c` that represent the inlet of each PFR.
    * c_shape:                      Tuple indicating the shape of c (num_cstr, num_species)
    * reactions:                    Reactions function in generated_code.py containing system of equations
    * bcs:                          Boundary conditions functions on domain inlets (all others don't matter).

    Returns
    -------
    * _ddt: The time derivative at each discretization point for each specie at the given time and concentrations
    """
    c = c.reshape(c_shape)

    _ddt = _ddt0.copy()

    # Calculate the backwards difference approximation of the convection term.
    # Note that we're not setting it for index 0, this fine since the first node in a PFR is treated differently.
    _ddt[:, 1:] *= c[:, 1:] - c[:, :-1]

    # Need to zero out the locations representing a PFR inlet since the lines above set them at this position.
    # Doing this two-step method makes it cleaner to implement
    # These values are correctly calculated below.
    _ddt[:, all_inlet_ids] = 0

    # Account for domain boundary conditions on the inlet nodes of the PFRs.
    bcs(t, _ddt)

    # Reaction must be calculated before calculating the time derivative of the inlets since
    # they impact the concentration of the final node, which is required for the last step
    reactions(c, _ddt)

    if connected_to_another_inlet[0, 0] >= 0:  # Negative value used to specify that this term is not needed.
        # Calculate the change in concentration at the inlet of each PFR due to flow from other connections
        # Note: This MUST be done after the spatial gradient and reactors are calculated since the right hand side here depends on them.
        for (inlet_node, connection, outlet_node) in connected_to_another_inlet:
            _ddt[:, inlet_node] += Q_weight[connection] * _ddt[:, outlet_node]

    return _ddt.ravel()
