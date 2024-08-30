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
Helper functions common to both CSTR- and PFR-based networks.
"""

from typing import Dict, Set, Tuple

import numpy as np
from scipy.optimize import linprog

from ..mesh import GroupedBCs


def check_network_for_disconnected_subgraphs(connection_pairings: Dict[int, Dict[int, int]]) -> None:
    """
    Check if the resulting compartment network contains disconnected subgraphs.
    Disconnected subgraphs are not allowed as it represents simulation domains which do not have mass transport between them.

    Throws a ValueError if there are disconnected subgraphs.

    Parameters
    ----------
    * connection_pairings:  Mapping between compartment ID and its neighbours.
                            Each entry is another mapping between the connection ID and the neighbour ID.
    Returns
    -------
    * Nothing, will through an error if there are disconnected subgraphs.
    """
    uncolored_nodes = set(connection_pairings.keys())
    num_subgraphs = 0

    while uncolored_nodes:
        stack = {uncolored_nodes.pop()}
        while stack:
            nodes_to_color = set.intersection(set(connection_pairings[stack.pop()].values()), uncolored_nodes)
            stack.update(nodes_to_color)
            uncolored_nodes.difference_update(nodes_to_color)
        num_subgraphs += 1

    if num_subgraphs > 1:
        raise ValueError(f"There are {num_subgraphs} disconnected subgraphs in the compartment network."
                         f"Check min_magnitude_threshold, flow_threshold, and boundary conditions (especially any internal ones).")


def tweak_compartment_flows(
        connection_pairing: Dict[int, Dict[int, int]],
        volumetric_flows:   Dict[int, float],
        grouped_bcs:        GroupedBCs,
        atol_opt:           float
) -> None:
    """
    This function is used to adjust the flowrates in the compartment network so that the net flow around a compartment
    is either 0 or slightly positive.

    This is different from `tweak_final_flows` in a few major ways:
    - This method works balances the mass around each COMPARTMENT, that one balances the mass around each CSTR/PFR.
    - That method tries to minimize the error in the conservation of mass, accepting small deviations resulting in either
      a small net inflow or outflow.
    - This method accepts looser tolerances in order to ensure there is NEVER a net outflow.
      This is requirement is needed by the Compartment -> PFR method, having even a small net outflow will result in the
      intra-compartment connections that get made to cause a reversal of flow.

    The first mass balance inequality written around each PFR is to ensure that there is a small netflow (epsilon)
    To ensure it's positive even after any small floating point precision issues.

        Σ_inlets (Q + x) - Σ_outlets(Q + x) >= eps
        (ΣQ_in - ΣQ_out) + (Σx_in - Σx_out) >= eps
        (ΣQ_in - ΣQ_out)                    >= eps  - (Σx_in - Σx_out)
        (-Σx_in + Σx_out)                   <= -eps + (ΣQ_in - ΣQ_out)

    The second mass balance inequality is to put an upper bound on the net flow into the compartment

        Σ_inlets (Q + x) - Σ_outlets(Q + x) <= atol_opt
        (ΣQ_in - ΣQ_out) + (Σx_in - Σx_out) <= atol_opt
        (Σx_in - Σx_out)                    <= atol_opt - (ΣQ_in - ΣQ_out)

    Where:
    - Q is the current flowrate through a connection
    - x is the non-negative adjustment to a given connection
    - eps is a small value used to avoid floating point issues when calculating the net flow.
    - atol_opt is the absolute tolerance for the conservation of mass, the maximum allowable net inflow.

    The system being solved is:

        minimize:   x_vec
        st:         A x_vec <= Σ (Q_in - Q_out)
                    x_vec >= 0

    The right hand side of the inequality is the net flow into a compartment
    * =0 : mass is balanced
    * >0 : net inflow into compartment
    * <0 : net outflow into compartment

    We will accept some net inflow, but we wish to remove all of the net outflows.

    For the first set of constraints (the ones with eps) the coefficients in the corresponding rows of A are:
    *  0: a connection is not contributing to this compartment
    * -1: a connection is an INLET for this compartment
    * +1: a connection is an OUTLET for this compartment
    For the second set of constraints (the ones with atol_opt) the coefficients are reversed; i.e. -1 is an OUTLET.

    Parameters
    ----------
    * connection_pairing:   Dictionary storing info about which other compartments a given compartment is connected to
                            - First key is compartment ID
                            - Values is a Dict[int, int]
                                - Key is connection ID (positive inlet into this compartment, negative is outlet)
                                - Value is the ID of the compartment on the other side
    * volumetric_flows:     Dictionary of the magnitude of volumetric flow through each connection, indexed by connection ID.
                            Connection ID in this dictionary is ALWAYS positive, need to take absolute sign of
                            the value if it's negative (see `connection_pairing` docstring)
    * grouped_bcs:          GroupedBCs object for identifying which connections belong to domain inlets/outlets.
    * atol_opt:             Absolute tolerance for evaluating conservation of mass of the optimized system.

    Returns
    -------
    * Nothing. Values are updated inplace
    """
    # Factor to avoid any precision issues when comparing floats
    safety_factor = 0.9

    # Small epsilon for conservation of mass to ensure that net flow is always positive.
    # Using 0 can cause floating point addition problems when the flow get close to summing to zero.
    eps: float = 1e-6

    if eps >= safety_factor*atol_opt:
        raise ValueError(f"Absolute tolerance must be greater than {eps/safety_factor:.3e}")

    # Scale volumetric flows to avoid numerical issues
    v_min = min(volumetric_flows.values())
    for _id in volumetric_flows:
        volumetric_flows[_id] /= v_min

    # Build coefficients for the objective function
    c = np.ones(len(volumetric_flows))

    # Volumetric flows cannot be assumed to be put into the dictionary in increasing order
    con_to_index = {_id: index for index, _id in enumerate(list(volumetric_flows.keys()))}

    # NOTE: There are no equality constraints

    # Create the inequality constraint
    n_compartments      = len(connection_pairing)
    n_flows             = len(volumetric_flows)
    A = np.zeros((2*n_compartments, n_flows), dtype='b')
    b = np.ones(2*n_compartments)
    b[:n_compartments] *= -eps/v_min
    b[n_compartments:] *= safety_factor*atol_opt/v_min

    domain_inlet_outlet_connections = set()
    for compartment, compartment_connections in connection_pairing.items():
        for connection, compartment_other in compartment_connections.items():
            # Keep track of the connections leading in/out of the domain in order. Needed for sanity check later on.
            if compartment_other < 0:
                domain_inlet_outlet_connections.add(abs(connection))

            if connection > 0:  # Inlet
                A[compartment, con_to_index[connection]] = -1
                b[compartment] += volumetric_flows[connection]

                # Meant to have opposite sign since equation is different
                A[n_compartments + compartment, con_to_index[connection]] = 1
                b[n_compartments + compartment] -= volumetric_flows[connection]
            else:  # Outlet
                A[compartment, con_to_index[abs(connection)]] = 1
                b[compartment] -= volumetric_flows[abs(connection)]

                # Meant to have opposite sign since equation is different
                A[n_compartments + compartment, con_to_index[abs(connection)]] = -1
                b[n_compartments + compartment] += volumetric_flows[abs(connection)]

    # Each column which does not correspond to a domain inlet/out must sum to 0.
    for connection, i in con_to_index.items():
        if connection not in domain_inlet_outlet_connections:
            assert np.sum(A[:, i]) == 0

    # Each row must have at least one positive and one negative value, otherwise mass would accumulate
    for i in range(A.shape[0]):
        assert np.any(A[i, :] > 0)
        assert np.any(A[i, :] < 0)

    for i in range(n_compartments):
        # It's -b[i] since the first n_compartments constraints are multiplied by negative -1
        # because they are lower bounds but must be written as upper bounds.
        if -b[i] >= b[i+n_compartments]:
            raise ValueError(f"Compartment {i} has infeasible net-inflow constraints: {b[i]:.1e} <= net_inflow <= {b[i+n_compartments]:.1e}")

    # Print pre-optimization stats
    print("Net-outflow compartments = {}".format((b[:n_compartments] < eps/v_min).sum()))
    print("BEFORE: MAX abs error: {:.4e}".format(np.max(abs(b[:n_compartments])) * v_min + eps))
    print("BEFORE: AVG abs error: {:.4e}".format(np.mean(abs(b[:n_compartments])) * v_min + eps))

    # Solve the equation
    results = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs', integrality=2)
    if not results["success"]:
        raise Exception("Flow optimization did not converge. \n" +
                        "Message:" + results["message"])

    adjustments: np.ndarray = results["x"]
    # Values should be positive but will sometimes come out negative with values like -1e-12 and v_min is 1e-05.
    # These values are essentially zero.
    assert np.all(adjustments >= -eps / v_min)

    b_new = b - A @ adjustments
    print("AFTER:  MAX abs error: {:.4e}".format(np.max(b_new[:n_compartments]) * v_min + eps))
    print("AFTER:  AVG abs error: {:.4e}".format(np.mean(b_new[:n_compartments]) * v_min + eps))

    # Adjust the volumetric flows
    for connection, i in con_to_index.items():
        volumetric_flows[connection] += adjustments[i]

    # Rescale back
    for _id in volumetric_flows:
        volumetric_flows[_id] *= v_min

    # Check the conservation of mass for the system as a whole and each compartment.
    # Each compartment is checked again to try to catch any issues with the optimization setup above
    missing_flow = 0.0
    for compartment, compartment_connections in connection_pairing.items():
        net_flow = 0.0
        for connection, compartment_other in compartment_connections.items():
            if connection > 0:
                net_flow += volumetric_flows[connection]
                if compartment_other < 0:
                    if compartment_other not in grouped_bcs.domain_inlets:
                        missing_flow += volumetric_flows[connection]
            else:
                connection = abs(connection)
                net_flow -= volumetric_flows[connection]
                if compartment_other < 0:
                    if compartment_other not in grouped_bcs.domain_outlets:
                        missing_flow -= volumetric_flows[connection]
        assert net_flow > 0
        assert np.isclose(net_flow, 0, rtol=0, atol=atol_opt)
    assert np.isclose(missing_flow, 0, rtol=0, atol=atol_opt)


def tweak_final_flows(
        connections: Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
        volumetric_flows: np.ndarray,
        grouped_bcs: GroupedBCs,
        atol_opt: float
) -> None:
    """
    This function is used to adjust the flowrates in the cstr/pfr network so that mass is conserved.
    The adjustment is done by solving a linear programming (optimization) problem over the mass balances.

        Constraints:  Mass balance around each PFR
        Objective:    Minimize the total amount adjustment

    See `tweak_compartment_flows` for an in-depth description

    Parameters
    ----------
    * connections:      Dictionary storing info about which other PFR/CSTR a given PFR/CSTR is connected to
                        - First key is PFR/CSTR ID
                        - Values is a Dict[int, int]
                            - Key is connection ID (positive inlet into this PFR/CSTR, negative is outlet)
                            - Value is the ID of the PFR/CSTR on the other side
    * volumetric_flows: Dictionary of the magnitude of volumetric flow through each connection, indexed by connection ID.
                        Connection ID in this dictionary is ALWAYS positive, need to take absolute sign of
                        the value if it's negative (see `connection_pairing` docstring)
    * grouped_bcs:      GroupedBCs object for identifying which connections belong to domain inlets/outlets.
    * atol_opt:         Absolute tolerance for evaluating conservation of mass of the optimized system

    Returns
    -------
    * Nothing. Values are changed implicitly
    """

    # Scale volumetric flows to avoid numerical issues
    v_min = min(volumetric_flows)
    volumetric_flows /= v_min

    # Build coefficients for the objective function
    c = np.ones(volumetric_flows.shape)

    # NOTE: There are no inequality constraints

    # Build equality constraint (A x = b)
    # Each row represents a PFR/CSTR
    # Each column represents a connection
    a = np.zeros((len(connections), c.size), dtype='b')
    indices_of_domain_inlet_outlet = set()
    for id_model in connections:
        inlets  = connections[id_model][0]
        outlets = connections[id_model][1]

        # Add adjustments for inlets
        for id_connection in inlets:
            if inlets[id_connection] < 0:
                indices_of_domain_inlet_outlet.add(id_connection)
            a[id_model, id_connection] = 1

        # Subtract adjustments for outlets
        # NOTE: This needs to be the reverse of the inlets
        for id_connection in outlets:
            if outlets[id_connection] < 0:
                indices_of_domain_inlet_outlet.add(id_connection)
            a[id_model, id_connection] = -1

    # Check that no connection has been missed
    for id_connection in range(a.shape[1]):
        # Each column which does not correspond to a domain inlet/out must sum to 0.
        # Summing to zero means that each time it was marked an inlet it is also marked as an outlet.
        # Domain inlets/outlet will not match up.
        if id_connection not in indices_of_domain_inlet_outlet:
            assert np.sum(a[:, id_connection]) == 0

        # Each column must have at least one non-zero entry to indicate that it's been connected to something
        assert np.any(a[:, id_connection] != 0)

    # Each row must have at least one positive and one negative value, otherwise mass would accumulate
    for i in range(a.shape[0]):
        assert np.any(a[i, :] > 0)
        assert np.any(a[i, :] < 0)

    # Create the right hand sign of the equality constraints.
    # The negative sum of existing flow (negative since moved to other side of equal sign)
    b = np.zeros((len(connections),))
    id_models = list(connections.keys())
    for i, id_model in enumerate(id_models):
        inlets  = connections[id_model][0]
        outlets = connections[id_model][1]

        # Subtract inlets since moved to other side of equal sign
        for id_connection in inlets.keys():
            b[i] -= volumetric_flows[id_connection]

        # Add outlets since moved to other side of equal sign
        for id_connection in outlets.keys():
            b[i] += volumetric_flows[id_connection]

    print("BEFORE: MAX abs error: {:.4e}".format(np.max(abs(b)) * v_min))
    print("BEFORE: AVG abs error: {:.4e}".format(np.mean(abs(b)) * v_min))
    # Solve the equation
    results = linprog(c, A_eq=a, b_eq=b, bounds=(0, None), integrality=2)
    if not results["success"]:
        raise Exception("Flow optimization did not converge. \n" +
                        "Message:" + results["message"])

    adjustments: np.ndarray = results["x"]
    # Values should be positive but will sometimes come out negative with values like -1e-12 and v_min is 1e-05.
    # These values are essentially zero.
    assert np.all(adjustments >= -1e-6 / v_min)

    b_new = np.abs(a @ adjustments - b)
    print("AFTER:  MAX abs error: {:.4e}".format(np.max(b_new) * v_min))
    print("AFTER:  AVG abs error: {:.4e}".format(np.mean(b_new) * v_min))

    # Add results to the flowrates
    volumetric_flows += adjustments
    # Rescale back
    volumetric_flows *= v_min

    # Check conservation of mass for the system as a whole
    net_flow      = 0.0
    missing_flow  = 0.0
    for inlets, outlets in connections.values():
        for id_connection, id_model_other_side in inlets.items():
            if id_model_other_side < 0:
                if id_model_other_side in grouped_bcs.domain_inlets:
                    net_flow     += volumetric_flows[id_connection]
                else:
                    missing_flow += volumetric_flows[id_connection]
        for id_connection, id_model_other_side in outlets.items():
            if id_model_other_side < 0:
                if id_model_other_side in grouped_bcs.domain_outlets:
                    net_flow     -= volumetric_flows[id_connection]
                else:
                    missing_flow -= volumetric_flows[id_connection]
    assert np.isclose(net_flow,     0, rtol=0, atol=atol_opt)
    assert np.isclose(missing_flow, 0, rtol=0, atol=atol_opt)

    # Check conservation of mass for individual components of the network
    for id_component in connections:
        # Add the inlets
        total = sum(volumetric_flows[id_connection] for id_connection in connections[id_component][0])
        # Subtract the outlets
        total -= sum(volumetric_flows[id_connection] for id_connection in connections[id_component][1])

        if ~np.isclose(total, 0, rtol=0, atol=atol_opt):
            print("Total flowrate around component {} is not balanced. Error {}".format(id_component, total))
            assert False
