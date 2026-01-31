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
Functions related to creating a network of CSTRs out of a network of compartments.
"""

from typing import List, Set, Tuple, Dict

import numpy as np

from .helpers import check_network_for_disconnected_subgraphs, tweak_final_flows
from ..config_functions import ConfigParser
from ..mesh import CMesh


def connect_cstr_compartments(compartment_network:      Dict[int, Dict[int, Dict[int, int]]],
                              mesh:                     CMesh,
                              flows_and_upwind:         np.ndarray,
                              check_level:              int,
                              config_parser:            ConfigParser) \
        -> Tuple[Dict[int, Dict[int, int]],
                 Dict[int, float]]:
    """
    Take an existing network of compartments and model each compartment as a CSTR.

    Unlike the PFR-based approach, this CSTR-based approach calculates the net flow between two compartments.
    If this net flow is below the user-specified threshold, then a connection between the two compartments is not created.

    Parameters
    ----------
    * compartment_network:  A dictionary representation of the compartments in the network.
                            Keys are compartment IDs, and values are dictionary.
                            - For each of those dictionaries, the keys are the index of a neighboring compartment
                              and the values are another dictionary.
                                - For each of those dictionaries, the keys are the index of the bounding entity
                                  between the two compartments, and the values are Tuples of two values.
                                    - The 1st is the index of the element upwind of that boundary facet.
                                    - The 2nd is the outward facing unit normal for that boundary facet.
    * mesh:                 The mesh the problem was solved on.
    * flows_and_upwind:     2D object array indexed by facet ID.
                            - 1st column is volumetric flowrate through facet.
                            - 2nd column is a flag indicating which of a facet's elements are upwind of it.
                                - 0, and 1 represent the index into mesh.facet_elements[facet]
                                - -1 is used for boundary elements to represent
    * check_level:          == 0 : No checks or asserts
                            == 1 : Check flow but no asserts
                            == 2 : Check flow and use asserts
                            Values of 0 and 1 are used by `merge_compartments` since some may be too small for
                            the invariants to be true until merging is done.
    * config_parser:        The OpenCCM ConfigParser.

    Returns
    -------
    1. connection_pairing:   Dictionary storing info about which other compartments a given compartment is connected to
                            - Key is compartment ID
                            - Values is a Dict[int, int]
                                - Key is connection ID (positive inlet into this compartment, negative is outlet)
                                - Value is the ID of the compartment on the other side
    2. volumetric_flows:     Dictionary of the magnitude of volumetric flow through each connection,
                            indexed by connection ID.
                            Connection ID in this dictionary is ALWAYS positive, need to take absolute sign of
                            the value if it's negative (see `connection_pairing` docstring)
    """
    # The minimum volumetric flow required between two compartments for a connection to be added between them.
    flow_threshold = config_parser.get_item(['COMPARTMENT MODELLING', 'flow_threshold'], float)

    connection_pairing: Dict[int, Dict[int, int]] = dict()
    volumetric_flows:   Dict[int, float]          = dict()

    # NOTE: Does not start indexing at 0 so that negative and positive signs can be used as signifier of inlet/outlet
    id_of_next_connection = 1
    rejected_connection_pairings = set()
    def pair(x, y):
        """Cantor pairing function"""
        return int(y + (x+y)*(x+y+1)/2)

    for id_compartment in compartment_network:
        compartment_connections: Dict[int, int] = dict()

        for id_neighbour in compartment_network[id_compartment]:
            # Find if this neighbour-compartment pair has already been done (but from the neighbour's side)
            # Note that the order in which values are sent to `pair` is flipped on purpose.
            need_to_do_neighbour_compartment_pair = pair(id_neighbour, id_compartment) not in rejected_connection_pairings
            if need_to_do_neighbour_compartment_pair:
                for _id_connection, _id_compartment in connection_pairing.get(id_neighbour, {}).items():
                    if id_compartment == _id_compartment:
                        compartment_connections[-_id_connection] = id_neighbour
                        need_to_do_neighbour_compartment_pair = False

                if need_to_do_neighbour_compartment_pair:
                    net_flow = 0.0

                    neighbour_dict: Dict[int, int] = compartment_network[id_compartment][id_neighbour]
                    for facet, element_this_side in neighbour_dict.items():
                        flow_through_facet, upstream_flag = flows_and_upwind[facet]
                        inflow = (upstream_flag == -1) or (element_this_side != mesh.facet_elements[facet][upstream_flag])
                        net_flow += (-1 if inflow else 1) * flow_through_facet

                    # If the flow is below the threshold, don't add the connection.
                    if check_level >= 1 and abs(net_flow) < flow_threshold:
                        rejected_connection_pairings.add(pair(id_compartment, id_neighbour))
                        continue

                    if net_flow < 0:  # Outward facing normal used, so negative value means flow into compartment
                        compartment_connections[id_of_next_connection] = id_neighbour
                    else:
                        compartment_connections[-id_of_next_connection] = id_neighbour

                    volumetric_flows[id_of_next_connection] = abs(net_flow)
                    id_of_next_connection += 1

        if check_level == 2:
            # Must have at least two connections, otherwise mass would accumulate inside the compartment
            assert len(compartment_connections) > 1

        connection_pairing[id_compartment] = compartment_connections

    return connection_pairing, volumetric_flows


def create_cstr_network(compartments:           Dict[int, Set[int]],
                        compartment_network:    Dict[int, Dict[int, Dict[int, int]]],
                        mesh:                   CMesh,
                        flows_and_upwind:       np.ndarray,
                        dir_vec:                np.ndarray,
                        config_parser:          ConfigParser)\
        -> Tuple[
            Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
            np.ndarray,
            np.ndarray,
            Dict[int, List[int]],
            List[List[Tuple[float, int]]]
        ]:
    """
    Function to create the CSTR network representation of the compartment model.

    Each compartment will be represented as single CSTR.

    Parameters
    ----------
    * compartments:         A dictionary representation of the elements in the compartments.
                            Keys are compartment IDs, values are sets containing the indices
                            of the elements in the compartment.
    * compartment_network:  A dictionary representation of the compartments in the network.
                            Keys are compartment IDs, and values are dictionary.
                            - For each of those dictionaries, the keys are the index of a neighboring compartment
                              and the values are another dictionary.
                                - For each of those dictionaries, the keys are the index of the bounding entity
                                  between the two compartments, and the values are Tuples of two values.
                                    - The 1st is the index of the element upwind of that boundary facet.
                                    - The 2nd is the outward facing unit normal for that boundary facet.
    * mesh:                 The mesh containing the compartments.
    * flows_and_upwind:     2D object array indexed by facet ID.
                            - 1st column is volumetric flowrate through facet.
                            - 2nd column is a flag indicating which of a facet's elements are upwind of it.
                                - 0, and 1 represent the index into mesh.facet_elements[facet]
                                - -1 is used for boundary elements to represent
    * dir_vec:              Numpy array of direction vectors, row i is for element i.
    * config_parser:        The OpenCCM ConfigParser

    Returns
    -------
    1. connections:             A dictionary representing the CSTR network.
                                The keys are the IDs of each CSTR, the values are tuples of two dictionaries.
                                    - The first dictionary is for flows into the CSTR.
                                    - The second dictionary is for flows out of the CSTR.
                                For both dictionaries, the key is the connection ID
                                and the value is the ID of the CSTR on the other end of the connection.
    2. volumes:                 A numpy array of the volume of each CSTR indexed by its ID.
    3. volumetric_flows:        A numpy array of the volumetric flowrate through each connection indexed by its ID.
    4. compartment_to_cstr_map: A new_id_for between a compartment ID and the ID of the CSTR representing it.
                                Here in order to preserve consistency with create_pfr_network.
    5. cstr_to_element_map:     A mapping between model ID and an ordered list of tuples containing:
                                    (element_id, 0). The zero is there to maintain the same type as the PFR version.
    """
    print('Creating CSTR network')

    atol_opt = config_parser.get_item(['COMPARTMENT MODELLING', 'atol_opt'], float)

    ####################################################################################################################
    # 1. Create connections between compartments
    ####################################################################################################################
    connection_pairing, _volumetric_flows = connect_cstr_compartments(compartment_network, mesh, flows_and_upwind, 2, config_parser)
    check_network_for_disconnected_subgraphs(connection_pairing)

    ####################################################################################################################
    # 2. Calculate volume of each compartment
    ####################################################################################################################
    volumes = np.zeros(len(compartments))
    for id_compartment, compartment_elements in compartments.items():
        volumes[id_compartment] = sum(mesh.element_sizes[element] for element in compartment_elements)

    ####################################################################################################################
    # 3. Post-Processing
    ####################################################################################################################
    # Re-label connections to [0, N-1] and split connections into inputs and outputs
    ids_old = list(_volumetric_flows.keys())
    ids_old.sort()
    new_id_for: Dict[int, int] = dict()

    volumetric_flows = np.zeros(len(ids_old))

    for id_new, id_old in enumerate(ids_old):
        new_id_for[id_old] = id_new
        volumetric_flows[id_new] = _volumetric_flows.pop(id_old)

    assert len(_volumetric_flows) == 0

    connections = dict()
    for id_cstr, connections_cstr in connection_pairing.items():
        inlets:  Dict[int, int] = dict()
        outlets: Dict[int, int] = dict()

        for id_old, other_csrt in connections_cstr.items():
            # Need the abs() call since the ids are negative in connection_pairings but not in the new_id_for
            if id_old > 0:
                inlets[new_id_for[id_old]]       = other_csrt
            else:
                outlets[new_id_for[abs(id_old)]] = other_csrt

        assert len(inlets) > 0
        assert len(outlets) > 0
        connections[id_cstr] = (inlets, outlets)

    tweak_final_flows(connections, volumetric_flows, mesh.grouped_bcs, atol_opt)

    compartment_to_cstr_map: Dict[int, List[int]] = {cstr: [cstr] for cstr in range(len(connections))}

    cstr_to_element_map: List[List[Tuple[float, int]]] = []
    for compartment in sorted(compartments.keys()):
        cstr_to_element_map.append([(0.0, element) for element in compartments[compartment]])

    print("Done creating CSTR network")
    return connections, volumes, volumetric_flows, compartment_to_cstr_map, cstr_to_element_map
