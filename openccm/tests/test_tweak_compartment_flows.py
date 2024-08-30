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

import numpy as np

from typing import Dict

from openccm import ConfigParser
from openccm.mesh import GroupedBCs
from openccm.compartment_models.helpers import tweak_compartment_flows


def test_tweak_compartment_flows():
    grouped_bcs = GroupedBCs(ConfigParser('CONFIG'))

    m = 100             # Number of compartments on each side of the centerline
    n = 1 + 2*m         # Total number of compartments
    delta_x = 1. / n    # Size of each compartment's edge

    V_max        = 1e-6
    frac_side    = 1e-5
    noise_factor = 1e-4

    np.random.seed(0)

    def velocity(xl, xr):
        """0th order projection of a parabolic flow profile in a pipe with walls at x=0 and x=1"""
        return 4*V_max / (2 * (xr - xl)) * ((xr ** 2 - xl ** 2) - 2 / 3 * (xr ** 3 - xl ** 3))

    connection_pairings: Dict[int, Dict[int, int]] = {i: {} for i in range(n)}
    volumetric_flows = {}
    # Add inlet and outlet connections
    for i, connections_i in connection_pairings.items():
        volumetric_flows[i+1]   = (1 + noise_factor * np.random.rand()) * delta_x * velocity(delta_x*i, delta_x*(i+1))
        volumetric_flows[n+i+1] = (1 + noise_factor * np.random.rand()) * delta_x * velocity(delta_x*i, delta_x*(i+1))
        connections_i[(i+1)]    = grouped_bcs.id('inlet')   # Inlet BC
        connections_i[-(n+i+1)] = grouped_bcs.id('outlet')  # Outlet BC

    # Bad inter-compartment connections from i to i+1
    for i in range(0, n//2):
        j = 2*n + (i+1)
        volumetric_flows[j] = V_max * frac_side * np.random.rand()
        connection_pairings[i][-j] = i+1    # Outlet for i
        connection_pairings[i+1][j] = i     # Inlet for i+1

    # Add bad inter-compartment connections from i to i-1
    for i in range(n//2+1, n):
        j = 2 * n + i
        volumetric_flows[j] = V_max * frac_side * np.random.rand()
        connection_pairings[i-1][j] = i     # Inlet for i-1
        connection_pairings[i][-j] = i-1    # Outlet for i

    # Calculate CoM before optimization (positive means net-inflow)
    def calculate_com(volumetric_flows, connection_pairings):
        def sign(x): return 1 if x > 0 else -1

        overall_com = sum(volumetric_flows[i+1] for i in range(n)) - sum(volumetric_flows[i+1] for i in range(n, 2*n))
        compartment_com = {}
        for id_compartment, connections in connection_pairings.items():
            compartment_com[id_compartment] = sum(sign(id_connection) * volumetric_flows[abs(id_connection)] for id_connection in connections)
        return overall_com, compartment_com

    # tweak_compartment_flows modifies in place, make a copy to see before and after
    volumetric_flows_orig = volumetric_flows.copy()
    overall_com_before, compartment_com_before = calculate_com(volumetric_flows_orig, connection_pairings)
    tweak_compartment_flows(connection_pairings, volumetric_flows, grouped_bcs, atol_opt=1e-2)
    overall_com_after,  compartment_com_after  = calculate_com(volumetric_flows, connection_pairings)


