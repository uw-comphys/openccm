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
from typing import Dict, Set, Tuple

import numpy as np
from scipy.optimize import linprog

from ..mesh import GroupedBCs


def tweak_compartment_flows(
        connection_pairing: Dict[int, Dict[int, int]],
        volumetric_flows: Dict[int, float],
        grouped_bcs: GroupedBCs,
        atol_opt: float,
        rtol_opt: float
) -> None:
    """
    This function is used to adjust the flowrates in the compartment network so that the net flow around a compartment
    is either 0 or slightly positive.

    This is different from `tweak_cstr_pfr_flows` in a few major ways:
    - This method works balances the mass around each COMPARTMENT, that one balances the mass around each CSTR/PFR.
    - That method tries to minimize the error in the conservation of mass, accepting small deviations resulting in either
      a small net inflow or outflow.
      This method accepts looser tolerances in order to ensure there is NEVER a net outflow.
      This is requirement is needed by the Compartment -> PFR method, having even a small net outflow will result in the
      intra-compartment connections that get made to cause a reversal of flow.

    A balance around each PFR is written as follows:
    Σ_inlets (Q + x) - Σ_outlets(Q + x) >= eps
    Σ (Q_in - Q_out) + Σ (x_in - x_out) >= eps
    Σ (Q_in - Q_out)                    >= -Σ (x_in - x_out) + eps
    Σ (- x_in + x_out)                  <=  Σ (Q_in - Q_out) - eps

    Where:
        - Q is the current flowrate through a connection
        - x is the non-negative adjustment to a given connection

    The system being solved is:
        minimize:   x_vec
        st:         A x_vec <= Σ (Q_in - Q_out)
                    x_vec >= 0

    The right hand side of the inequality is the net flow into a compartment
        =0 : mass is balanced
        >0 : net inflow into compartment
        <0 : net outflow into compartment

    We will accept some net inflow, but we wish to remove all of the net outflows.

    The coefficients in the A matrix are as follows:
        *  0: a connection is not contributing to this compartment
        * -1: a connection is an INLET for this compartment
        * +1: a connection is an OUTLET for this compartment

    Args:
        connection_pairing:     Dictionary storing info about which other compartments a given compartment is connected to
                                    - First key is compartment ID
                                    - Values is a Dict[int, int]
                                        - Key is connection ID (positive inlet into this compartment, negative is outlet)
                                        - Value is the ID of the compartment on the other side
        volumetric_flows:       Dictionary of the magnitude of volumetric flow through each connection,
                                    indexed by connection ID.
                                    Connection ID in this dictionary is ALWAYS positive, need to take absolute sign of
                                    the value if it's negative (see `connection_pairing` docstring)
        domain_inlet_outlets:
        atol_opt:                   Absolute tolerance for evaluating conservation of mass of the optimized system
        rtol_opt:                   Relative tolerance for evaluating conservation of mass of the optimized system
    """
    # Small epsilon for conservation of mass to ensure that net flow is always positive.
    # using 0 can cause floating point addition problems when the flow get close to summing to zero.
    eps = 1e-10

    # Build coefficients for the objective function
    c = np.ones(len(volumetric_flows))

    # NOTE: There are no equality constriants

    # Create the inequality constraint
    A = np.zeros((len(connection_pairing), c.size), dtype='b')
    b = -eps * np.ones(len(connection_pairing))
    domain_inlet_outlet_connections = set()
    for compartment, compartment_connections in connection_pairing.items():
        for connection, compartment_other in compartment_connections.items():
            # Keep track of the connections leading in/out of the domain in order. Needed for sanity check later on.
            if compartment_other < 0:
                domain_inlet_outlet_connections.add(abs(connection))

            if connection > 0:  # Inlet
                A[compartment, connection-1] = -1  # Need to do the -1 since the connections are still index starting at 1 at this point.
                b[compartment] += volumetric_flows[connection]
            else:  # Outlet
                A[compartment, abs(connection)-1] = 1
                b[compartment] -= volumetric_flows[abs(connection)]

    # Each column which does not correspond to a domain inlet/out must sum to 0.
    for i, connection in enumerate(volumetric_flows.keys()):
        if connection not in domain_inlet_outlet_connections:
            assert np.sum(A[:, i]) == 0

    # Each row must have at least one positive and one negative value, otherwise mass would accumulate
    for i in range(A.shape[0]):
        assert np.any(A[i, :] > 0)
        assert np.any(A[i, :] < 0)

    # Print pre-optimization stats
    print("Net-outflow compartments = {}".format((b < 0).sum()))
    print("BEFORE: MAX abs error: {:.4e}".format(np.max(abs(b))))
    print("BEFORE: AVG abs error: {:.4e}".format(np.mean(abs(b))))

    # Build the bounds for x (the adjustment). Only bound is they must be non-negative
    bounds = [(0, np.inf) for _ in range(len(volumetric_flows))]

    # Solve the equation
    results = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
    if not results["success"]:
        raise Exception("Flow optimization did not converge. \n" +
                        "Message:" + results["message"])

    adjustments: np.ndarray = results["x"]

    b_new = np.abs(A @ adjustments - b)
    assert np.all(b_new >= 0)
    print("AFTER:  MAX abs error: {:.4e}".format(np.max(b_new)))
    print("AFTER:  AVG abs error: {:.4e}".format(np.mean(b_new)))

    # Adjust the volumetric flows
    for i, connection in enumerate(volumetric_flows.keys()):
        volumetric_flows[connection] += adjustments[i]

    # Check the conservation of mass for the system as a whole and each compartment.
    # Each compartment is checked again to try to catch any issues with the optimization setup above
    total_inflow  = 0.0
    total_outflow = 0.0
    missing_flow = 0.0
    for compartment, compartment_connections in connection_pairing.items():
        net_flow = 0.0
        for connection, compartment_other in compartment_connections.items():
            if connection > 0:
                net_flow += volumetric_flows[connection]
                if compartment_other < 0:
                    if compartment_other in grouped_bcs.domain_inlets:
                        total_inflow += volumetric_flows[connection]
                    else:
                        # Subtracting on purpose
                        missing_flow -= volumetric_flows[connection]
            else:
                connection = abs(connection)
                net_flow -= volumetric_flows[connection]
                if compartment_other < 0:
                    if compartment_other in grouped_bcs.domain_outlets:
                        total_outflow += volumetric_flows[connection]
                    else:
                        missing_flow += volumetric_flows[connection]

        assert net_flow > 0
        assert np.isclose(net_flow, 0, rtol=rtol_opt, atol=atol_opt)
    assert np.isclose(total_inflow, total_outflow, rtol=rtol_opt, atol=atol_opt)
    assert np.isclose(0.0, missing_flow, rtol=rtol_opt, atol=atol_opt)


def tweak_final_flows(
        connections: Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
        volumetric_flows: np.ndarray,
        grouped_bcs: GroupedBCs,
        atol_opt: float,
        rtol_opt: float
) -> None:
    """
    This function is used to adjust the flowrates in the cstr/pfr network so that mass is conserved.
    The adjustment is done by solving a linear programming (optimization) problem over the mass balances.
    Constraints:  Mass balance around each PFR
    Objective:    Minimize the total amount adjustment

    Args:
        connections:
        volumetric_flows:
        domain_inlet_outlets:
        atol_opt:                   Absolute tolerance for evaluating conservation of mass of the optimized system
        rtol_opt:                   Relative tolerance for evaluating conservation of mass of the optimized system

    Returns:
        Nothing. Values are changed implicitly
    """

    # Build coefficients for the objective function
    c = np.ones(volumetric_flows.shape)

    # NOTE: There are no inequality constraints

    # Build equality constraint (A x = b)
    # Each row represents a compartment
    # Each column represents a flowrate
    a = np.zeros((len(connections), c.size), dtype='b')
    indices_of_domain_inlet_outlet = set()
    for id_pfr in connections:
        inlets  = connections[id_pfr][0]
        outlets = connections[id_pfr][1]

        # Add adjustments for inlets
        for id_connection in inlets:
            if inlets[id_connection] < 0:
                indices_of_domain_inlet_outlet.add(id_connection)
            a[id_pfr, id_connection] = 1

        # Subtract adjustments for outlets
        # NOTE: This needs to be the reverse of the inlets
        for id_connection in outlets:
            if outlets[id_connection] < 0:
                indices_of_domain_inlet_outlet.add(id_connection)
            a[id_pfr, id_connection] = -1

    # Each column which does not correspond to a domain inlet/out must sum to 0.
    # Summing to zero means that each time it was marked an inlet to a domain
    for id_connection in connections:
        if id_connection not in indices_of_domain_inlet_outlet:
            assert np.sum(a[:, id_connection]) == 0

    # Each row must have at least one positive and one negative value, otherwise mass would accumulate
    for i in range(a.shape[0]):
        assert np.any(a[i, :] > 0)
        assert np.any(a[i, :] < 0)

    # Create the right hand sign of the equality constraints.
    # The negative sum of existing flow (negative since moved to other side of equal sign)
    b = np.zeros((len(connections),))
    id_pfrs = list(connections.keys())
    for i, id_pfr in enumerate(id_pfrs):
        inlets  = connections[id_pfr][0]
        outlets = connections[id_pfr][1]

        # Subtract inlets since moved to other side of equal sign
        for id_connection in inlets.keys():
            b[i] -= volumetric_flows[id_connection]

        # Add outlets since moved to other side of equal sign
        for id_connection in outlets.keys():
            b[i] += volumetric_flows[id_connection]

    # Build the bounds for x (the adjustment). Only bound is they must be non-negative
    bounds = [(0, np.inf) for _ in range(volumetric_flows.size)]

    print("BEFORE: MAX abs error: {:.4e}".format(np.max(abs(b))))
    print("BEFORE: AVG abs error: {:.4e}".format(np.mean(abs(b))))
    # Solve the equation
    results = linprog(c, A_eq=a, b_eq=b, bounds=bounds)
    if not results["success"]:
        raise Exception("Flow optimization did not converge. \n" +
                        "Message:" + results["message"])

    adjustments: np.ndarray = results["x"]

    b_new = np.abs(a @ adjustments - b)
    print("AFTER:  MAX abs error: {:.4e}".format(np.max(b_new)))
    print("AFTER:  AVG abs error: {:.4e}".format(np.mean(b_new)))

    # Add results to the flowrates
    volumetric_flows += adjustments

    # Check conservation of mass for the system as a whole
    total_inflow  = 0.0
    total_outflow = 0.0
    missing_flow  = 0.0
    for inlets, outlets in connections.values():
        for id_connection, id_pfr_other_side in inlets.items():
            if id_pfr_other_side < 0:
                if id_pfr_other_side in grouped_bcs.domain_inlets:
                    total_inflow += volumetric_flows[id_connection]
                else:
                    # Subtracting on purpose
                    missing_flow -= volumetric_flows[id_connection]
        for id_connection, id_pfr_other_side in outlets.items():
            if id_pfr_other_side < 0:
                if id_pfr_other_side in grouped_bcs.domain_outlets:
                    total_outflow += volumetric_flows[id_connection]
                else:
                    missing_flow += volumetric_flows[id_connection]
    assert np.isclose(total_inflow, total_outflow, atol=atol_opt, rtol=rtol_opt)
    assert np.isclose(0.0,          missing_flow,  atol=atol_opt, rtol=rtol_opt)

    # Check conservation of mass for individual components of the network
    for id_component in connections:
        # Add the inlets
        total = sum(volumetric_flows[id_connection] for id_connection in connections[id_component][0])
        # Subtract the outlets
        total -= sum(volumetric_flows[id_connection] for id_connection in connections[id_component][1])

        if ~np.isclose(total, 0, atol=atol_opt, rtol=rtol_opt):
            print("Total flowrate around component {} is not balanced. Error {}".format(id_component, total))
            assert False
