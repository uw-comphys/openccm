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
Functions related to creating a network of PFRs out of a network of compartments.
"""

from collections import Counter
from copy import deepcopy
from typing import Dict, List, Set, Tuple


import numpy as np

from ..config_functions import ConfigParser
from ..mesh import CMesh, GroupedBCs
from .helpers import check_network_for_disconnected_subgraphs, tweak_compartment_flows, tweak_final_flows


def create_pfr_network(compartments:        Dict[int, Set[int]],
                       compartment_network: Dict[int, Dict[int, Dict[int, int]]],
                       mesh:                CMesh,
                       flows_and_upwind:    np.ndarray,
                       dir_vec:             np.ndarray,
                       config_parser:       ConfigParser)\
        -> Tuple[
            Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
            np.ndarray,
            np.ndarray,
            Dict[int, List[int]],
            List[List[Tuple[float, int]]]
        ]:
    """
    Function to create the PFR network representation of the compartment model.
    Each compartment will be represented as one or more PFRs in series (depending on the number of inlets/outlets).
    The PFRs will be created so that while each many have more than one inlet/outlet, all inlets are at the start of
    the PRF and all outlets are at the end.

    Parameters
    ----------
    * compartments:         A dictionary representation of the elements in the compartments.
                            Keys are compartment IDs, values are sets containing the indices
                            of the elements in the compartment.
    * compartment_network:  A dictionary representation of the compartments in the network.
                            Keys are compartment IDs, and whose values are dictionary.
                            - For each of those dictionaries, the keys are the index of a neighboring compartment
                              and the values are another dictionary.
                                - For each of those dictionaries, the keys are the index of the bounding entity
                                  between the two compartments, and the values Tuples.
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
    5. pfr_to_element_map:      A mapping between PFR ID and an ordered list of tuples containing:
                                (element_id, distance_along_pfr) where distance_along_compartment in range of [0.0, 1.0].
    """
    print("Creating PFR network")

    atol_opt        = config_parser.get_item(['COMPARTMENT MODELLING', 'atol_opt'],       float)
    dist_threshold  = config_parser.get_item(['COMPARTMENT MODELLING', 'dist_threshold'], float)

    # The volume of each compartment
    volumes_compartments: Dict[int, float] = dict()
    for id_compartment, compartment_elements in compartments.items():
        volumes_compartments[id_compartment] = sum(mesh.element_sizes[element] for element in compartment_elements)

    ####################################################################################################################
    # 1. Create connections between compartments
    ####################################################################################################################
    # 1.1 Initial connections
    results_1 = connect_pfr_compartments(compartment_network, compartments, mesh, dir_vec, flows_and_upwind, 2, config_parser)
    id_next_connection, connection_distances, element_distances, connection_pairing, compartment_network, _volumetric_flows = results_1
    check_network_for_disconnected_subgraphs(connection_pairing)

    # 1.2 If it's a closed system, add extra inlet flows to allow for the flows to be tweaked
    need_extra_flows = len(mesh.grouped_bcs.domain_in_out_names) == 0
    if need_extra_flows:
        config_parser['INPUT']['domain_inlet_names'] = "('extra_inlet',)"
        tmp_grouped_bcs = GroupedBCs(config_parser)

        compartment_to_extra_connection: Dict[int, int] = {}
        id_extra_compartment = max(compartments) + 1
        extra_compartment: Dict[int, int] = {id_next_connection: tmp_grouped_bcs.id('extra_inlet')}
        _volumetric_flows[id_next_connection] = 0
        id_next_connection += 1
        for compartment, connections in connection_pairing.items():
            connections[id_next_connection] = id_extra_compartment
            extra_compartment[-id_next_connection] = compartment
            _volumetric_flows[id_next_connection] = 0
            compartment_to_extra_connection[compartment] = id_next_connection
            id_next_connection += 1
        connection_pairing[id_extra_compartment] = extra_compartment
    else:
        tmp_grouped_bcs = mesh.grouped_bcs

    # 1.3 Optimize connections to prevent flow reversal when creating the intra-compartment flows.
    tweak_compartment_flows(connection_pairing, _volumetric_flows, tmp_grouped_bcs, atol_opt)

    # 1.4 Reorder connections and merge their locations as needed.
    connection_locations: Dict[int, List[Tuple[float, List[int]]]] = dict()
    for id_compartment, connections_distances_i in connection_distances.items():
        # Check that the compartment connections are correct
        compartment_connections = connection_pairing[id_compartment]
        _connections = np.array(list(compartment_connections.keys()))
        assert np.any(_connections > 0)
        assert np.any(_connections < 0)

        # 1.4a If needed, add the artificial flows to the compartment
        if need_extra_flows:
            # Put in to be the second inlet into the compartment
            i_first_inlet = next(i for i, (_, id_connection) in enumerate(connections_distances_i) if id_connection > 0)
            connections_distances_i.insert(i_first_inlet+1, (connections_distances_i[i_first_inlet][0], compartment_to_extra_connection[id_compartment]))

        # 1.4b Reorder connections
        connections_distances_i = _fix_connection_ordering(connections_distances_i, _volumetric_flows, id_compartment, atol_opt)
        # 1.4c Re-order domain inlets/outlets
        connections_distances_i = _fix_domain_boundary_connection_ordering(connections_distances_i, compartment_connections, mesh.grouped_bcs, id_compartment)
        # 1.4d Merge connection locations
        connection_locations[id_compartment] = _merge_connections(connections_distances_i, _volumetric_flows, dist_threshold, compartment_connections, atol_opt)

    ####################################################################################################################
    # 2. Split compartment into PFRs
    ####################################################################################################################
    output = _compartments_to_pfrs(connection_locations, connection_pairing, volumes_compartments, _volumetric_flows,
                                   max(connection_pairing.keys()) + 1, id_next_connection, element_distances, skip_last_compartment=need_extra_flows)
    _volumes, _volumetric_flows, connection_pairing, compartment_to_pfr_map, pfr_to_element_map  = output

    # If added, remove extra compartment info
    if need_extra_flows:
        config_parser['INPUT']['domain_inlet_names']  = "None"
        id_extra_compartment = max(connection_pairing.keys())
        extra_connections_info = connection_pairing.pop(id_extra_compartment)

        for connection, neighbour in extra_connections_info.items():
            _volumetric_flows.pop(abs(connection))

            if neighbour < 0:  # domain_inlet
                pass  # Domain inlet, won't be in connection_pairing
            else:
                connection_pairing[neighbour].pop(-connection)

    ####################################################################################################################
    # 3. Post Processing
    ####################################################################################################################
    # Re-number connections from 0 to N-1
    # The mapping must be made here but is only applies in the loop since the positive and negative signs
    # on the connection ID are still needed in order to split them into inlets/outlets.
    id_new = 0
    ids_old = sorted(_volumetric_flows.keys())
    map: Dict[int, int] = dict()

    for id_old in ids_old:
        map[id_old] = id_new
        _volumetric_flows[id_new] = _volumetric_flows.pop(id_old)
        id_new += 1

    # Re-numbing inlet and outlet information and adding it to the final dictionary
    connections: Dict[int, Tuple[Dict[int, int], Dict[int, int]]] = dict()
    for pfr, _connections in connection_pairing.items():
        inlets:  Dict[int, int] = dict()
        outlets: Dict[int, int] = dict()

        for id_connection, pfr_other in _connections.items():
            # Update the connection info in volumetric_flows
            if id_connection > 0:
                inlets[map[int(id_connection)]] = pfr_other
            else:
                outlets[map[int(-id_connection)]] = pfr_other

        connections[pfr] = (inlets, outlets)

    # Convert to numpy values for numba usage later
    volumes = np.zeros((len(_volumes),))
    for pfr_id in _volumes:
        volumes[pfr_id] = _volumes[pfr_id]

    assert np.all(volumes > 0)

    volumetric_flows = np.zeros((len(_volumetric_flows,)))
    for connection_id in _volumetric_flows:
        volumetric_flows[connection_id] = _volumetric_flows[connection_id]

    for pfr, connections_i in connections.items():
        assert len(connections_i[0].values()) > 0  # There should be inlets
        assert len(connections_i[1].values()) > 0  # There should be outlets

    # Final optimization of flowrates in order to ensure mass is conserved around each PFR
    tweak_final_flows(connections, volumetric_flows, mesh.grouped_bcs, atol_opt)

    print("Done creating PFR network")
    return connections, volumes, volumetric_flows, compartment_to_pfr_map, pfr_to_element_map


def _compartments_to_pfrs(connection_locations:      Dict[int, List[Tuple[float, List[int]]]],
                          all_connection_pairing:    Dict[int, Dict[int, int]],
                          volumes_compartments:      Dict[int, float],
                          volumetric_flows:          Dict[int, float],
                          id_of_next_pfr:            int,
                          id_of_next_connection:     int,
                          element_distances:         Dict[int, List[Tuple[float, int]]],
                         skip_last_compartment:     bool) \
        -> Tuple[
            Dict[int, float],
            Dict[int, float],
            Dict[int, Dict[int, int]],
            Dict[int, List[int]],
            List[List[Tuple[float, int]]]
        ]:
    """
    Function to convert a network of compartments into a network of PFRs.

    This function assumes that the compartments are numbered sequentially [0, N) without any gaps.

    Parameters
    ----------
    * connection_locations:     Dictionary to lookup locations of inlets/outlets for a compartment.
                                For a detailed description see the documentation in method_name.
    * all_connection_pairing:   Dictionary for looking up connectivity of a compartment.
                                For a detailed description see the documentation in create_pfr_network.
    * volumes_compartments:     Volume of each compartment, indexed by compartment ID.
    * volumetric_flows:         Volumetric flowrate through each connection, lookup using connection ID.
    * id_of_next_pfr:           ID to use for the next PFR to be made.
    * id_of_next_connection:    ID to use for the next connection if one needs to be made.
                                This value is used to ensure that each connection has a unique ID.
    * element_distances:        Dictionary storing the distances for each element
                                Outer Dict indexed by compartment ID, inner Dict indexed by element ID.
    * skip_last_compartment:    Whether the last compartment should be split into PFRs or not.
                                If True, the last compartment represents the extra compartment used to balance
                                mass in a closed system.

    Returns
    -------
    1. volumes_pfr:             Volume of each PFR, lookup using PFR ID.
    2. volumetric_flows:        Flows through each compartment, updated with all new flows
    3. all_connection_pairing:  Same as input, but now updated to contain the pairings between PFRs rather than
                                between compartments
    4. compartment_to_pfr_map:  Mapping between the compartment ID and the PFR(s) that are inside it.
    5. pfr_to_element_map:      A mapping between PFR ID and an ordered list of tuples containing:
                                (element_id, distance_along_pfr) where distance_along_compartment in range of [0.0, 1.0].
    """
    volumes_pfr: Dict[int, float] = dict()
    compartment_to_pfr_map: Dict[int, List[int]] = dict()
    pfr_to_element_map: List[List[Tuple[float, int]]] = []

    # ID to use for the PFR
    # If a compartment needs multiple PFRs then they each need an ID
    # To make things simpler, ALL PFRs will be renumbered even those coming from a compartment with a single PFR.
    # The renumbering occurs by compartment, and then within a compartment the PFRs are labeled from smallest to largest
    # in the direction of the flow (i.e. smaller numbers are found upstream).
    """
            E.g.
            Compartment IDs | # FPRs | Final PFRs IDs
            ----------------|--------|---------------
            1               | 2      | 5, 6
            2               | 1      | 7
            3               | 3      | 8, 9, 10
            4               | 1      | 11
    """
    # Split the compartment into PFRs based on the ordering of the inlets and outlets
    for id_compartment_i, connection_locations_i in connection_locations.items():
        # Information for ensure that the number of connections is correct
        id_of_next_pfr_pre = id_of_next_pfr
        num_connections_pre = sum(len(connections) for _, connections in connection_locations_i)

        element_distances_i = element_distances[id_compartment_i]

        '''1. Split the compartment at each connection location (skipping the first)'''
        # Starting at 1 rather than 0 since we need 2 ends (either [i, i+1] or [i-1, i])
        # and it's easier to make sure that we don't index out of bounds if we use the [i-1, i].
        for i in range(1, len(connection_locations_i)):
            id_pfr = id_of_next_pfr
            id_of_next_pfr += 1

            # Add the PFR to the connection list if it's not already in it.
            # It may already have been added when adding a previous PFR and adding its intra-compartment connection
            if id_pfr not in all_connection_pairing:
                all_connection_pairing[id_pfr] = dict()

            '''2. Create a PFR by using the inlets between the previous and the current location'''
            # The connections are ordered based on the projection onto the average flow direction.
            # This means that a higher value is ALWAYS downstream of a lower value. Liquid flows from 0 to 1.
            # So higher values of i are downstream of lower values.
            connections_inlet_edge  = deepcopy(connection_locations_i[i - 1])  # Deep copy since we'll remove values
            connections_outlet_edge = deepcopy(connection_locations_i[i])

            dist_inlet  = connections_inlet_edge[0]
            dist_outlet = connections_outlet_edge[0]
            dist_delta  = dist_outlet - dist_inlet
            i_left = next((_i for _i, entry in enumerate(element_distances_i) if entry[0] > dist_outlet), len(element_distances_i))
            if i_left == 0:
                print(f"WARNING: Compartment {id_compartment_i} has connection spacings which results in PFR {len(pfr_to_element_map)} that does not map "
                      f"to any element. Consider increasing `dist_threshold`, `flow_threshold`, and/or `flow_threshold_facet`.")
            pfr_to_element_map.append([((dist - dist_inlet)/dist_delta, element) for dist, element in element_distances_i[:i_left]])
            element_distances_i = element_distances_i[i_left:]

            # Remove the connections which do not belong in this PFR
            '''
            Consider the following compartment with inlets (A)&(B) and outlet (C).
                (A, B, C, D) are integers
            It gets split into 2 PFRs, PFR 1 and PFR 2.
            -(A)> [PFR 1] -(D)> [PFR 2] -(C)>
                    ^
                   (B)
                    |
            At the split, the side feed, inlet (B) must be added to ONE of the two PFRs.

            When creating PFR 1, connections_inlet_edge and connections_outlet_edge will have the following contents:
                connections_inlet_edge:  (d_in, [A])
                connections_outlet_edge: (d_out, [B])
                    d_in and d_out are floats representing the distance along the compartment, 
                    their values are irrelevant, the important thing is that:
                        - d_in in [0.0, 1.0)
                        - d_out in (0.0, 1.0]
                        - d_in < d_out
            We know not to add (B) to PFR 1 since it's an inlet but occurs on PFR 1's outlet side.
            Thus it must be added ONLY to PFR 2.
            The opposite would be true if (B) was an outlet (i.e. must be added to PFR 1 and NOT PFR 2). 
            '''
            # Remove from inlet
            _id_index = 0
            while _id_index < len(connections_inlet_edge[1]):
                if connections_inlet_edge[1][_id_index] < 0:
                    connections_inlet_edge[1].pop(_id_index)
                else:
                    # Increment only if we don't pop, otherwise popping an element is the same as incrementing
                    _id_index += 1

            # Remove from outlet
            _id_index = 0
            while _id_index < len(connections_outlet_edge[1]):
                if connections_outlet_edge[1][_id_index] > 0:
                    connections_outlet_edge[1].pop(_id_index)
                else:
                    # Increment only if we don't pop, otherwise popping an element is the same as incrementing
                    _id_index += 1

            # If this is NOT the last PFR in this compartment, calculate the flowrate between this PFR and the next one
            # in this compartment and add it to the list of connections.
            if i < len(connection_locations_i) - 1:
                # Calculate the net flow in an out of the compartment
                # At this point it will NOT balance since we need to add a new flow from this PFR to the next one
                flow_in = sum([volumetric_flows[_connection_id] for _connection_id in connections_inlet_edge[1]])
                flow_out = sum([volumetric_flows[abs(_connection_id)] for _connection_id in connections_outlet_edge[1]])

                # Add any intra-compartment flows created previously (these are never stored in connection_locations)
                for id_intra_connection in all_connection_pairing[id_pfr]:  # Anything in here at this point are intra-compartment connections created on a previous iteration.
                    if id_intra_connection > 0:
                        flow_in += volumetric_flows[id_intra_connection]
                    else:
                        flow_out += volumetric_flows[abs(id_intra_connection)]

                # Flowrate of the new intra-compartment flowrate between this and the next PFR
                flow_intra = flow_in - flow_out
                # If the net flow is out of this compartment, then the intra-compartment connection between this
                # and the next DOWNSTREAM compartment will flow backwards in order to balance mass.
                # This is unphysical.
                assert flow_intra > 0
                id_new_connection = -id_of_next_connection
                id_of_next_connection += 1

                # Create a new connection between this PFR and the next PFR,
                # and add it directly to all_connection_pairing
                # Intra-compartment connections are between this PFR and the next one downstream
                # NOTE: These are not added to connection_locations so that they not be double counted later
                all_connection_pairing[id_pfr][id_new_connection] = id_of_next_pfr
                # Create a new entry for the downstream PFR and add this connection to it.
                # Since they are labeled sequentially, we know that it's ID is (this ID + 1)
                all_connection_pairing[id_of_next_pfr] = {-id_new_connection: id_pfr}

                # Save the volumetric flow through this connection
                volumetric_flows[int(abs(id_new_connection))] = flow_intra

            # Save the volume of this compartment
            # (total volume * fractional distance between upstream and downstream cutoffs)
            volume_i = volumes_compartments[id_compartment_i] * (connections_outlet_edge[0] - connections_inlet_edge[0])
            assert volume_i > 0
            volumes_pfr[id_pfr] = volume_i

            '''3. Get the compartment on the other side of connection using all_connection_pairing'''
            '''4. Update from compartment number to FPR ID in all_connection_pairing and elsewhere'''
            # Update all connections coming in and out of this PFR to point to this PFR instead of the compartment
            for connection in connections_inlet_edge[1] + connections_outlet_edge[1]:
                # Get the compartment/PFR on the other side of this connection,
                # and de-associate this connection from the compartment
                # Note: this will never delete an intra-compartment connection since those are stored
                #       under the ID of the PFR and NOT of the compartment.
                id_compartment_other_side = all_connection_pairing[id_compartment_i].pop(connection)
                # Add this connection under this PFR's ID
                all_connection_pairing[id_pfr][connection] = id_compartment_other_side
                # Update the connection info for the compartment on the other side
                # NOTE: Negative ID means mesh boundary, they are skipped since they don't have entries
                if id_compartment_other_side > 0:
                    # NOTE: The negative sign in from on connection is used this the type of connection
                    # (inlet/outlet) is flipped for the other end of the connection.
                    # An inlet for this PFR is an outlet for whatever the PFR is connected to.
                    all_connection_pairing[id_compartment_other_side][-connection] = id_pfr

        '''5. Remove the compartment ID from all_connection_pairing when its empty'''
        assert len(all_connection_pairing[id_compartment_i]) == 0
        all_connection_pairing.pop(id_compartment_i)

        # For N PFRs there are N-1 intra-compartment connections added between them.
        # 2x that number since each intra-compartment connection appears twice, once for each PFR that it connects.
        num_intra_compartment_connections = 2*((id_of_next_pfr - id_of_next_pfr_pre) - 1)

        # The number of connections before and after splitting the compartment into PFRs should stay the same after
        # accounting for the intra-compartment connections.
        num_connections_post = sum(len(all_connection_pairing[id_pfr]) for id_pfr in range(id_of_next_pfr_pre, id_of_next_pfr))
        assert num_connections_pre == (num_connections_post - num_intra_compartment_connections)

        # Store the list of PFRs in this compartment
        compartment_to_pfr_map[id_compartment_i] = list(range(id_of_next_pfr_pre, id_of_next_pfr))

    # If needed, renumber the skipped compartment in order to give it the largest ID
    # So that after removing it later the remaining compartments are labeled [0, N-1]
    if skip_last_compartment:
        id_to_skip = min(all_connection_pairing.keys())
        id_new_for_skipped = max(all_connection_pairing.keys()) + 1

        # Re-number inside all_connection_pairing
        connections = all_connection_pairing.pop(id_to_skip)
        all_connection_pairing[id_new_for_skipped] = connections
        # Update numbering of this PFR in all referenced locations
        for id_connection, id_pfr_other in connections.items():
            if id_pfr_other >= 0:  # Negative value is domain inlet/outlet
                all_connection_pairing[id_pfr_other][-id_connection] = id_new_for_skipped
    else:
        id_new_for_skipped = -1
    # Re-number PFRs
    for id_new, id_old in enumerate(sorted(all_connection_pairing.keys())):
        # Re-number inside all_connection_pairing
        connections = all_connection_pairing.pop(id_old)
        all_connection_pairing[id_new] = connections
        # Update numbering of this PFR in all referenced locations
        for id_connection, id_pfr_other in connections.items():
            if id_pfr_other >= 0:  # Negative value is domain inlet/outlet
                all_connection_pairing[id_pfr_other][-id_connection] = id_new

        if id_old == id_new_for_skipped:
            # Don't need the rest, so skip it.
            continue

        # Re-number inside compartment_to_pfr_map
        for pfrs_in_compartment in compartment_to_pfr_map.values():
            if id_old in pfrs_in_compartment:
                pfrs_in_compartment[pfrs_in_compartment.index(id_old)] = id_new
                break  # Break since each PFR only shows up in one compartment.

        # Re-number inside volumes
        volumes_pfr[id_new] = volumes_pfr.pop(id_old)

    for volume in volumes_pfr.values():
        assert volume > 0

    # Each pfr must have at least one input and one output
    for connection_pairing in all_connection_pairing.values():
        _connections = np.array(list(connection_pairing.keys()))
        assert np.any(_connections > 0)
        assert np.any(_connections < 0)

    return volumes_pfr, volumetric_flows, all_connection_pairing, compartment_to_pfr_map, pfr_to_element_map


def _merge_connections(inlets_and_outlets:      List[Tuple[float, int]],
                       volumetric_flows:        Dict[int, float],
                       dist_threshold:          float,
                       compartment_connections: Dict[int, int],
                       atol_opt:                float)\
        -> List[Tuple[float, List[int]]]:
    """
    Function to merge inlets/outlet by changing their location along the compartment.

    Merging is based on the distance between adjacent inlets/outlet.
    E.g.: for a threshold of 5% and (A,B,C,D) being all inlets:
          A --2%-- B --1%-- C --1%-- D   ==> ABCD           <- All merged since A-to-D is 4%, which is <= 5%

    Merging of multiple inlets will happen only if there isn't an outlet inbetween them, same with outlet.
    Merging occurs from 0% of the length to 100% of the length. (left-to-right)

    More than two inlets/outlets can be merged together only if no two are further apart than the threshold.
    If a 3rd, or 4th or subsequent, inlet/outlet is being considered to be merged, the decision to merge it
    is based on the distance between it and the furthest inlet/outlet in the group.
          A --2%-- B --2%-- C --2%-- D   ==> ABC  ---- D    <- Only ABC are merged since D is 6% away from A.
                                                               A ---- BCD is not a valid result, and neither is
                                                               AB ---- CD.

    The merged location is calculated one of two ways:
    1. At 0% or 100%, if-and-only-if one of the inlets/outlets to be merged is the first inlet (or last outlet).
    2. Weighted average (using volumetric flowrate) of all locations to merge.

    Parameters
    ----------
    * inlets_and_outlets:       List of tuples containing the pre-merging inlets/outlets and their position.
                                Each tuple has two values, first is a float and the second is an integer.
                                <p>
                                - The float represents the position along the length of the compartment of this inlet/outlet.
                                  From 0.0 to 1.0 (inclusive of both ends).
                                <p>
                                - The integer represents the ID of this inlet/outlet.
                                  A positive value signifies an inlet and a negative value signifies and outlet.
    * volumetric_flows:         Dictionary to look up the flowrate through each connection by its ID
    * dist_threshold:           The maximum distance between any two points in a merge connection location
    * compartment_connections:  Dictionary mapping the connection IDs for a compartment
                                to the compartment on the other side.
    * atol_opt:                 Absolute tolerate used for conservation of mass.

    Returns
    -------
    * connections_merged: A copy of the `inlets_and_outlets` but with any required merging performed.
    """
    connections_merged: List[Tuple[float, List[int]]]   = []
    flows_in:           List[float]                     = []
    flows_out:          List[float]                     = []
    flows_intra:        List[float]                     = []
    inlets_and_outlets                                  = inlets_and_outlets.copy()

    # Ensure that the first and last connections are of the correct types and in the correct locations
    assert inlets_and_outlets[0][0] == 0.0
    assert inlets_and_outlets[0][1] > 0
    assert inlets_and_outlets[-1][0] == 1.0
    assert inlets_and_outlets[-1][1] < 0


    flow_profile: List[Tuple[float, float]] = []
    net_flow = 0.0
    # Check that mass is conserved within the compartments
    for pos, connection in inlets_and_outlets:
        if connection > 0:  # Inlet
            net_flow += volumetric_flows[connection]
        else:  # Outlet
            net_flow -= volumetric_flows[abs(connection)]
        flow_profile.append((pos, net_flow))

    # The compartments won't be fully conservative due to using a 0th order projection of the velocity.
    # Ensure that the error is not very large.
    assert 0 < net_flow < atol_opt

    # The connections which link to a domain inlet/outlet.
    # They should occur at the end of a compartment, and CANNOT be merged with so that the VTU output looks like it
    # Conserves mass (which it does)
    domain_inlet_outlet_connections: Set[int] = set()
    for pos, id in inlets_and_outlets:
        if compartment_connections[id] < 0:
            domain_inlet_outlet_connections.add(id)

    # Need to use a while loop since we're going to be modifying the length of the list as we go.
    i = 0
    while len(inlets_and_outlets) > 0:
        flows_in.append(0.0)
        flows_out.append(0.0)
        flows_intra.append(0.0)
        # Find all values to merge
        pos_i, id_inlet_outlet = inlets_and_outlets.pop(0)

        ids = [id_inlet_outlet]
        positions = [pos_i]

        # The current flow out of this PFR
        # Used in order to make sure that merging of outlets does not cause reversal of flow
        if ids[0] < 0:
            flows_out[i] += volumetric_flows[abs(ids[0])]
        else:
            flows_in[i] += volumetric_flows[ids[0]]

        # Iterate over the rest of the inlets/outlets trying to merge things in.
        while len(inlets_and_outlets) > 0:
            pos_candidate, id_candidate = inlets_and_outlets[0]
            assert pos_candidate >= pos_i

            if id_inlet_outlet in domain_inlet_outlet_connections:
                # Only merge other domain intets/outlets together
                if id_candidate not in domain_inlet_outlet_connections:
                    break

                # Do not merge and inlet with an outlet
                if id_candidate * id_inlet_outlet < 0:
                    break
            else:
                # If the next value is further away than the threshold then we can't add it or any values after it
                # (The list is ordered in terms of distance from 0.0 to 1.0).
                if pos_candidate - pos_i > dist_threshold:
                    break

                # Don't merge into a domain inlet/outlet
                if id_candidate in domain_inlet_outlet_connections:
                    break

                # Do not merge connections of differing signs (i.e. do not merge inlets & outlets together) if
                # one of those connections are the main inlet/outlet of the PFR (identified by a position of 0.0 or 1.0).
                # Use the fact that inlets and outlets have an index with an opposite sign (0 is not used)
                if (pos_i == 0 or pos_candidate == 1.0) and np.any([id_i * id_candidate < 0 for id_i in ids]):
                    break

                # Check that this will not cause a backwards flowing intra-compartment connection
                # Only need to check if the connection to add is an outlet
                # Don't check if we're merging at the last location since all the outlets have to be added
                if id_candidate < 0 and pos_candidate != 1.0:
                    if (flows_out[i] + volumetric_flows[abs(id_candidate)]) > flows_in[i-1] + flows_intra[i-1]:
                        # Intra-compartment flow would be reverse direction
                        break

            # Remove the current inlet/outlet from the list since we're merging it and done with it
            inlets_and_outlets.pop(0)

            if id_candidate < 0:
                flows_out[i] += volumetric_flows[abs(id_candidate)]
            else:
                flows_in[i] += volumetric_flows[id_candidate]
            positions.append(pos_candidate)
            ids.append(id_candidate)

        if (0.0 in positions) or (1.0 in positions):
            pass  # Don't have to do anything
            # flows_intra.append(0.0)
        else:
            flows_intra[i] = (flows_in[i-1] + flows_intra[i-1]) - flows_out[i]
            assert flows_intra[i] > 0

        # Calculate the location of the merged positions
        if 0.0 in positions:  # One of the merged inlets is the first inlet
            position_merged = 0.0
        elif 1.0 in positions:  # One of the merged outlets is the last outlet
            position_merged = 1.0
        else:  # None of the inlets/outlets are at the compartment extremes
            # Calculate position based on weighed average
            position_merged = 0.0
            volumetric_sum = 0.0

            for j in range(len(ids)):
                # Call to absolute needed since index is stored as negative for outlets
                flow = volumetric_flows[abs(ids[j])]
                position_merged += flow * positions[j]
                volumetric_sum += flow

            position_merged /= volumetric_sum

        # Save all inlets/outlets at their new locations
        connections_merged.append((position_merged, ids))
        i += 1

    # The first entry must be an inlet (inlets are represented by a positive connection ID)
    assert np.all([id_connection > 0 for id_connection in connections_merged[0][1]])
    # The last entry must be an outlet (outlets are represented by a negative connection ID)
    assert np.all([id_connection < 0 for id_connection in connections_merged[-1][1]])

    # A domain BC should not have been merged into any of the locations:
    for _, connections in connections_merged:
        other_side = np.array([compartment_connections[connection] for connection in connections])
        assert np.all(other_side < 0) or np.all(other_side >= 0)

    assert len(flows_in) == len(flows_out) == len(flows_intra)

    for i in range(len(flows_in)):
        if i == 0:  # Special case at main compartment inlet
            assert flows_in[0]    >  0
            assert flows_intra[0] == 0
            assert flows_out[0]   == 0
        elif i == len(flows_in)-1:  # Special case at main compartment outlet
            assert flows_in[i]    == 0
            assert flows_intra[i] == 0  # No next PFR, all flow must go through the outlet connections
            assert (flows_in[i-1] + flows_intra[i-1]) - flows_out[i] > 0
        else:  # At position i, we need to do a balance over the PFR between i-1 and i
            # Flows in INCLUDING the intra-compartment flow must be greater than the outflows
            assert flows_in[i-1] + flows_intra[i-1] > flows_out[i]
            assert flows_intra[i] == (flows_in[i-1] + flows_intra[i-1]) - flows_out[i]

            # Check that the flow from this PFR [i-1, i] into the next PFR [i, i+1] is positive
            assert flows_intra[i] > 0

    return connections_merged


def _fix_connection_ordering(inlets_and_outlets:    List[Tuple[float, int]],
                             volumetric_flows:      Dict[int, float],
                             id_compartment:        int,
                             atol_opt:              float) \
        -> List[Tuple[float, int]]:
    """
    The current approach for ordering the inlets and outlets isn't perfect.
    It will fail in a few instances, causing the first inlet to occur slightly AFTER the first outlet.
    This is not correct for a unidirectional compartment.
    This function handles those cases and logs that they occurred.

    First, the first connection must be an inlet and the last connection must be an outlet.
    If either of these conditions is not met, the connections are searched for the nearest connection of the correct kind
    And the positions are swapped.

    Next, connections are rearranged by identifying locations where flow reversal happens, and moving upstream as many
    downstream inlets as are required to prevent the reversal.
    The outlets are not moved.

    Parameters
    ----------
    * inlets_and_outlets: The ordered list of inlets and outlets.
    * volumetric_flows:   The volumetric flow through each connection, indexed by connection ID.
    * id_compartment:     The ID of the compartment whose inlets & outlets are being fixed.
                          Used only for logging purposes.
    * atol_opt:           Absolute tolerate used for conservation of mass.

    Returns
    -------
    * inlets_and_outlets: The re-ordered list of inlets and outlets
    """

    # Fix the first inlet
    if inlets_and_outlets[0][1] < 0:
        # Find the index of the connection to swap with
        index = None
        for i in range(1, len(inlets_and_outlets)):
            if inlets_and_outlets[i][1] > 0: # Looking for an inlet, which has a positive value
                index = i
                break

        if index is None:
            raise Exception("All connections to this compartment have the same sign")

        # Swap
        id_pre  = inlets_and_outlets[0][1]
        id_post = inlets_and_outlets[index][1]
        dist = inlets_and_outlets[index][0] - inlets_and_outlets[0][0]
        num_inbetween = index - 1

        inlets_and_outlets[index]    = (inlets_and_outlets[index][0],  id_pre)
        inlets_and_outlets[0]        = (inlets_and_outlets[0][0],     id_post)

        # Calculate and print information of about the swap
        print("Had to swap main inlet for compartment {}. Swapped {}<->{}, dist {}, # inbetween {}."
              .format(id_compartment, id_pre, id_post, dist, num_inbetween))

    # Fix the last outlet
    if inlets_and_outlets[-1][1] > 0:
        # Find the index of the connection to swap with
        index = None
        for i in reversed(range(len(inlets_and_outlets)-1)):
            if inlets_and_outlets[i][1] < 0:  # Looking for an outlet, which has a negative value
                index = i
                break

        if index is None:
            raise Exception("All connections to this compartment have the same sign")

        # Swap
        id_pre  = inlets_and_outlets[-1][1]
        id_post = inlets_and_outlets[index][1]
        dist    = inlets_and_outlets[-1][0] - inlets_and_outlets[index][0]
        num_inbetween = len(inlets_and_outlets) - index - 1

        inlets_and_outlets[index] = (inlets_and_outlets[index][0], id_pre)
        inlets_and_outlets[-1] = (inlets_and_outlets[-1][0], id_post)

        # Calculate and print information of about the swap
        print("Had to swap main outlet for compartment {}. Swapped {}<->{}, dist {}, # inbetween {}."
              .format(id_compartment, id_pre, id_post, dist, num_inbetween))

    net_flow = 0.0
    # Check that mass is conserved within the compartments
    for pos, connection in inlets_and_outlets:
        if connection > 0:  # Inlet
            net_flow += volumetric_flows[connection]
        else:  # Outlet
            net_flow -= volumetric_flows[abs(connection)]

    # The compartments won't be fully conservative due to using a 0th order projection of the velocity.
    # The flows will be tweaked after the compartments are modelled as PFRs.
    # For now, we need to ensure that the error is not very large.
    assert 0 < net_flow < atol_opt

    # Move inlets or outlets around so that the flow inside the compartment
    # (and therefore the intra-compartment connections)
    # always stays in the correct direction.
    net_flow_cumulative = 0.0
    i = 0
    while i < len(inlets_and_outlets):  # Use a while loop so that we can re-order entries inside `inlets_and_outlets`.
        pos_i, connection_i = inlets_and_outlets[i]
        if connection_i > 0:  # Inlet
            # Easy to deal with, will never cause issues.
            # Add to net_flow and move on.
            net_flow_cumulative += volumetric_flows[connection_i]
        else:  # Outlet
            _net_flow_cumulative = net_flow_cumulative - volumetric_flows[abs(connection_i)]
            if _net_flow_cumulative > 0:  # There is a net inflow, carry on
                net_flow_cumulative = _net_flow_cumulative
            else:  # Flow reversal, need to rearrange connections
                # The rearranging will happen by moving inlets
                _to_move: List[int] = []

                # 1. Collect all inlets that will be moved
                j = i+1
                while j < len(inlets_and_outlets)-1:
                    _connection_j = inlets_and_outlets[j][1]
                    if _connection_j > 0:
                        _to_move.append(j)
                        _net_flow_cumulative += volumetric_flows[_connection_j]
                        if _net_flow_cumulative > 0:
                            break
                    j += 1

                if len(_to_move) == 0 or _net_flow_cumulative < 0:
                    raise ValueError(f"Could not rearrange inlets of compartment {id_compartment} to have valid intra-compartment flows")

                print(f"Had to move connections {[inlets_and_outlets[j][1] for j in _to_move]} in compartment {id_compartment} in order to have positive intra-compartment flows")
                # 2. Move them
                for j in _to_move:
                    k = j
                    while k > i:
                        pos_right, id_right  = inlets_and_outlets[k]
                        pos_left,  id_left   = inlets_and_outlets[k-1]

                        inlets_and_outlets[k]   = (pos_right, id_left)
                        inlets_and_outlets[k-1] = (pos_left, id_right)
                        k -= 1

                # Update the net flow with the new connection
                net_flow_cumulative += volumetric_flows[inlets_and_outlets[i][1]]
        i += 1

    assert np.isclose(net_flow, net_flow_cumulative)

    return inlets_and_outlets


def _fix_domain_boundary_connection_ordering(inlets_and_outlets:        List[Tuple[float, int]],
                                             compartment_connections:   Dict[int, int],
                                             grouped_bcs:               GroupedBCs,
                                             id_compartment:            int)\
        -> List[Tuple[float, int]]:
    """
    Currently, the merging of small compartments can result in compartments with multiple connections to domain inlets/outlets.
    Not all of these connections occur sequentially.

    This function ensures that these connections happen sequentially so that they can be merged together by a later step.

    This is done by moving the connections to domain inlets/outlets to the correct end of the compartment without changing
    the relative ordering of the other connections.

    Parameters
    ----------
    * inlets_and_outlets:       The ordered list of inlets and outlets.
    * compartment_connections:  Dictionary for this compartment between the connection ID and compartment on other side.

    Returns
    -------
    * inlets_and_outlets: The re-ordered list of inlets and outlets
    """
    counter = Counter(compartment_connections.values())
    if min(counter) < 0:  # Domain boundaries are indicated with negative IDs, a negative minimum value means there is at least one domain boundary
        i_in = 0  # Insertion point were we move the next value
        i_out = len(inlets_and_outlets) - 1
        for id_other, count in counter.items():
            if id_other >= 0:  # Not a domain boundary
                continue

            print(f"Compartment {id_compartment} has multiple connections to domain inlet/outlet: {id_other}. Reordering as necessary.")
            fixed_connections = 0
            if id_other in grouped_bcs.domain_inlets:
                j = i_in  # Current point we're looking at
                while fixed_connections < count:
                    if compartment_connections[inlets_and_outlets[j][1]] != id_other:
                        j += 1
                    else:
                        j_prev = j
                        while j != i_in:  # Sequentially shift connection ids to the right but leave positions alone
                            io_at_j             = (inlets_and_outlets[j  ][0], inlets_and_outlets[j-1][1])
                            io_at_j_minus_one   = (inlets_and_outlets[j-1][0], inlets_and_outlets[j  ][1])
                            inlets_and_outlets[j]   = io_at_j
                            inlets_and_outlets[j-1] = io_at_j_minus_one
                            j -= 1
                        i_in += 1
                        j = int(max(j_prev, i_in))
                        fixed_connections += 1
            elif id_other in grouped_bcs.domain_outlets:
                j = i_out  # Current point we're looking at
                while fixed_connections < count:
                    if compartment_connections[inlets_and_outlets[j][1]] != id_other:
                        j -= 1
                    else:
                        j_prev = j
                        while j != i_out:  # Sequentially shift connection ids to the left but leave positions alone
                            io_at_j             = (inlets_and_outlets[j+1][0], inlets_and_outlets[j  ][1])
                            io_at_j_plus_one    = (inlets_and_outlets[j  ][0], inlets_and_outlets[j+1][1])
                            inlets_and_outlets[j]       = io_at_j
                            inlets_and_outlets[j + 1]   = io_at_j_plus_one
                            j += 1
                        i_out -= 1
                        j = int(min(j_prev, i_out))
                        fixed_connections += 1
            elif id_other in grouped_bcs.ignored:
                pass
            else:
                raise ValueError

        assert inlets_and_outlets[ 0][0] == 0.0
        assert inlets_and_outlets[-1][0] == 1.0
        # Ordering of the inlets/outlets must always be in increasing order
        for i in range(1, len(inlets_and_outlets)):
            assert inlets_and_outlets[i][0] >= inlets_and_outlets[i-1][0]

        # If there are domain inlets they must be at the front in a contiguous block
        for i_left in range(len(inlets_and_outlets)):
            if compartment_connections[inlets_and_outlets[i_left][1]] >= 0:
                break  # The first non-domain connection

        # If there are domain outlets they must be at the back in a contiguous block
        for i_right in reversed(range(len(inlets_and_outlets))):
            if compartment_connections[inlets_and_outlets[i_right][1]] >= 0:
                break  # The last non-domain connection

        # Nothing outside of those contiguous blocks should be a domain inlet/outlet
        for i in range(i_left, i_right+1):
            if compartment_connections[inlets_and_outlets[i][1]] < 0:
                raise AssertionError(f"Domain inlet/outlets for compartment {id_compartment} were not ordered properly")

    return inlets_and_outlets


def connect_pfr_compartments(compartment_network:   Dict[int, Dict[int, Dict[int, int]]],
                             compartments:          Dict[int, Set[int]],
                             mesh:                  CMesh,
                             dir_vec:               np.ndarray,
                             flows_and_upwind:      np.ndarray,
                             check_level:           int,
                             config_parser:         ConfigParser) \
        -> Tuple[
            int,
            Dict[int, List[Tuple[float, int]]],
            Dict[int, List[Tuple[float, int]]],
            Dict[int, Dict[int, int]],
            Dict[int, Dict[int, Dict[int, int]]],
            Dict[int, float]
        ]:
    """
    Convert a network of compartments connected by facets into a network of compartments connected by labeled connections.
    Each connection representing multiple facets and having an ID and flowrate associated with it.

    This function can return a network with multiple connections between two compartments.

    A surface shared by two compartments is made up of facets, all facets below the specified threshold
    `flow_threshold_facet` are removed from the surface.
    Then the surface is split into contiguous subsurfaces which all have the same direction of flow.
    Any subsurface which has a flow rate below `flow_threshold` is removed.
    Each subsurface is treated as a separate connection between the two compartments that it's between.

    Parameters
    ----------
    * compartment_network:  A dictionary representation of the compartments in the network.
                            Keys are compartment IDs, and whose values are dictionary.
                                For each of those dictionaries, the keys are the index of a neighboring compartment
                                and the values are another dictionary.
                                    For each of those dictionaries, the keys are the index of the bounding facet
                                    between the two compartments, and the values Tuples.
                                        - The 1st is the index of the element upwind of that boundary facet.
                                        - The 2nd is the outward facing unit normal for that boundary facet.
    * compartments:         A dictionary representation of the elements in the compartments.
                            Keys are compartment IDs, values are sets containing the indices
                            of the elements in the compartment.
    * mesh:                 The mesh the problem was solved on.
    * dir_vec:              Numpy array of direction vectors, row i is for element i.
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
    * config_parser:        The OpenCCM ConfigParser

    Returns
    -------
    1. id_next_connection:      The integer to use for the next connection ID.
    2. connection_distances:    Mapping between comparment ID and the information about connection distances.
                                Key is compartment ID, value is a list of tuples (distance, connection_id)
                                which has been sorted by `distance`.
    3. element_distances:       Mapping between compartment ID and the information about element distances.
                                Key is compartment ID, value is a list of tuples (distance, element_id)
                                which has been sorted by `distance`.
    4. connection_pairing:      Dictionary storing info about which other compartments a given compartment is connected to.
                                - First key is compartment ID
                                - Values is a Dict[int, int]
                                    - Key is connection ID (positive inlet into this compartment, negative is outlet)
                                    - Value is the ID of the compartment on the other side
    5. compartment_network:     The compartment network passed in.
    6. compartments             The compartments passed in.
    7. volumetric_flows:        Dictionary of the magnitude of volumetric flow through each connection,
                                indexed by connection ID.
                                Connection ID in this dictionary is ALWAYS positive, ensure index using
                                `volumetric_flows[abs(ID)]` if the connection ID is negative
                                (see `connection_pairing` docstring).
    """
    print("Finding connections between compartments")

    # The minimum volumetric flow required between two compartments for a connection to be added between them.
    flow_threshold          = config_parser.get_item(['COMPARTMENT MODELLING', 'flow_threshold'],       float)
    # The minimum volumetric flow through a facet for it to considered, facets with flow below this amount are removed.
    flow_threshold_facet    = config_parser.get_item(['COMPARTMENT MODELLING', 'flow_threshold_facet'], float)
    log_folder_path         = config_parser.get_item(['SETUP',                 'log_folder_path'],      str)
    DEBUG                   = config_parser.get_item(['SETUP',                 'DEBUG'],                bool)

    # Dictionary used to store the merged connection information
    connection_distances: Dict[int, List[Tuple[float, int]]] = dict()
    element_distances:    Dict[int, List[Tuple[float, int]]] = dict()
    # Dictionary storing info about which other compartments a given compartment is connected to
    connection_pairing:   Dict[int, Dict[int, int]]   = dict()
    volumetric_flows:     Dict[int, float]            = dict()

    # ID to use for the next connection
    # NOTE: Does not start indexing at 0 so that negative and positive signs can be used as signifier of inlet/outlet
    id_of_next_connection = 1

    # Lookup for center of fluxes
    #   The center of flux is calculated as the weight average position of each flux facet
    #   The weighting is the fraction of the total flux that comes through each facet
    #   The position of each facet is the mean of all of its vertices
    centers_of_flow_for_connection: Dict[int, np.ndarray] = dict()
    with open(log_folder_path + 'connect' + ('' if check_level == 2 else '_for_merge') + '.txt', 'w') as logging:
        # Calculate the number of connections between compartments and their locations
        for id_compartment in compartment_network:
            # Inlets and outlets for this compartment.
            #   Key is connection ID, value is the ID of the other compartment
            #   A negative key represents an outlet and a positive an inlet
            compartment_connections: Dict[int, int] = dict()

            # Calculate flowrate, center of flow, and inlet/outlet status for all connection for this compartment
            for neighbour in compartment_network[id_compartment]:
                # Iterate over all of connection_pairing and grab all connections from there
                id_connections_for_compartment_neighbour_pair: List[int] = []
                if id_compartment in connection_pairing.get(neighbour, {}).values():
                    for _id_connection, _id_compartment in connection_pairing.get(neighbour, {}).items():
                        if _id_compartment == id_compartment:
                            # If connection was an inlet to the other compartment, it's an outlet to this one.
                            id_connections_for_compartment_neighbour_pair.append(-_id_connection)
                            # Do not exit early since there could be multiple connections

                if len(id_connections_for_compartment_neighbour_pair) > 0:  # This pairing already processed, grab data
                    for id_connection in id_connections_for_compartment_neighbour_pair:
                        compartment_connections[id_connection] = neighbour
                else:
                    """
                    1. Calculate and store the flowrate through each facet.
                    2. Calculate the total flowrate with all facets.
                    3. If it's below the total flow threshold, delete this connection and stop.
                    4. Mark and remove all facets which are below the individual flow threshold.
                    5. Calculate the total flowrate with the remaining facets
                    6. Split the surface into contiguous blocks with the same sign (inlet/outlet)
                    7. For each resulting contiguous surface calculate the total flowrate through it and if its an inlet/outlet
                       - Calculate the fraction of the total flow remaining (the total flow obtained) passing through each surface
                       - Multiply that fraction by the total flowrate obtained when ALL edges are considered.
                       - NOTE: This is done is order for mass to be better conserved.
                    8. Calculate the center of flow for each surface.
                    9. Store the results
                    """

                    ###
                    # 1. Calculate and store the flowrate through each facet.
                    ###
                    flow_through_facet: Dict[int, float] = dict()
                    for facet, element_this_side in compartment_network[id_compartment][neighbour].items():
                        flow_through_facet_i, upstream_flag = flows_and_upwind[facet]
                        inflow = (upstream_flag == -1) or (element_this_side != mesh.facet_elements[facet][upstream_flag])
                        flow_through_facet[facet] = (-1 if inflow else 1) * flow_through_facet_i

                    ###
                    # 2. Calculate the total flowrate with all facets.
                    ###
                    total_flow = sum(flow_through_facet.values())

                    ###
                    # 3. If it's below the total flow threshold, delete this connection and stop.
                    ###
                    if check_level >= 1 and np.abs(total_flow) < flow_threshold:
                        continue

                    ###
                    # 4. Mark and remove all facets which are below the individual flow threshold.
                    ###
                    for facet in list(flow_through_facet.keys()):
                        if check_level >= 1 and np.abs(flow_through_facet[facet]) < flow_threshold_facet:
                            flow_through_facet.pop(facet)

                    ###
                    # 5. Calculate the total flowrate with the remaining facets
                    ###
                    total_flow_thresholded = sum(flow_through_facet.values())

                    ###
                    # 6. Split the surface into contiguous blocks with the same sign (inlet/outlet)
                    ###
                    # NOTE: One potential issue to look out for is that this may result a single surface being broken up
                    #       into many tiny surfaces.
                    inlet_facets: Set[int] = set()
                    outlet_facet: Set[int] = set()
                    for facet, flow_value in flow_through_facet.items():
                        # Flow calculated with normal facing out of the compartment
                        # Negative values are in the opposite direction of the
                        if flow_value < 0:
                            inlet_facets.add(facet)
                        else:
                            outlet_facet.add(facet)

                    surfaces: List[List[int]] = []
                    surfaces.extend(_group_facets_into_surfaces(inlet_facets, mesh))
                    surfaces.extend(_group_facets_into_surfaces(outlet_facet, mesh))

                    ###
                    # 7. For each resulting contiguous surface calculate the total flowrate through it and if its an inlet/outlet
                    ###
                    flow_through_surfaces: Dict[int, float] = {}
                    for i, surface in enumerate(surfaces):
                        _flow = sum(flow_through_facet[id_facet] for id_facet in surface) * total_flow / total_flow_thresholded
                        if check_level == 0 or abs(_flow) >= flow_threshold:
                            flow_through_surfaces[i] = _flow
                        else:
                            pass

                    ###
                    #  8. Calculate the center of flow for each surface.
                    #  9. Store the results
                    ###
                    for i, flowrate in flow_through_surfaces.items():
                        volumetric_flows[id_of_next_connection] = abs(flowrate)
                        if check_level >= 1:
                            center_of_flow = np.sum([abs(flow_through_facet[facet]) * mesh.facet_centers[facet] for facet in surfaces[i]], 0)
                            center_of_flow /= abs(flow_through_surfaces[i])

                            centers_of_flow_for_connection[id_of_next_connection] = center_of_flow

                        if flowrate < 0:  # Inlet
                            compartment_connections[id_of_next_connection] = neighbour
                        else:  # Outlet
                            compartment_connections[-id_of_next_connection] = neighbour
                        id_of_next_connection += 1

            if check_level == 2:
                # Must have at least one inlet or outlet. Otherwise, something has gone horrible wrong.
                assert len(compartment_connections) >= 1

                # Calculate average direction
                avg_direction = np.mean(dir_vec[sorted(compartments[id_compartment])], 0)
                avg_direction /= np.linalg.norm(avg_direction)
                if DEBUG:
                    logging.write(f"avg_direction {id_compartment} = {avg_direction}\n")

                # Calculate the ordering of the inlets/outlets
                # Project centers of flux onto average direction
                # Min and max values for normalizing between 0.0 and 1.0
                # (setting to inf and -inf so that the min and max always work).
                dot_min, dot_max = np.inf, -np.inf
                _connections_distances_i: Dict[int, float] = dict()
                for id_connection in compartment_connections:
                    center_of_flow = centers_of_flow_for_connection[abs(id_connection)]
                    dot_val = avg_direction.dot(center_of_flow)

                    dot_min, dot_max = min(dot_min, dot_val), max(dot_max, dot_val)
                    _connections_distances_i[id_connection] = dot_val

                # Project the centroid of each element onto the average diraction
                element_distances_i: Dict[int, float] = dict()
                for id_element in compartments[id_compartment]:
                    element_distances_i[id_element] = avg_direction.dot(mesh.element_centroids[id_element])
                    # Purposefully not updating dot_min and dot_max so that the connections distances keep min and max values

                # Normalize between 0.0 and 1.0
                delta_dot = dot_max - dot_min
                for id_connection in _connections_distances_i:
                    _connections_distances_i[id_connection] = (_connections_distances_i[id_connection] - dot_min) / delta_dot
                for id_element in element_distances_i:
                    element_distances_i[id_element] = np.clip((element_distances_i[id_element] - dot_min) / delta_dot, a_min=0.0, a_max=1.0)

                connection_distances[id_compartment] = sorted(((dist, connection) for connection, dist in _connections_distances_i.items()),
                                                              key=lambda entry: entry[0])
                element_distances   [id_compartment] = sorted(((dist, element) for element, dist in element_distances_i.items()),
                                                              key=lambda entry: entry[0])
            connection_pairing[id_compartment] = compartment_connections

        if DEBUG:
            logging.write("id_of_next_connection: {}\n".format(id_of_next_connection))
            logging.write("connection_pairing: {}\n".format(connection_pairing))
            logging.write("compartment_network: {}\n".format(compartment_network))
            logging.write("compartments: {}\n".format(compartments))
            logging.write("volumetric_flows: {}\n".format(volumetric_flows))

    print("Done finding connections between compartments")

    if check_level == 2:
        for connection, flow in volumetric_flows.items():
            if flow < flow_threshold:
                raise ValueError(f"Connection {connection} has a flowrate ({flow}) below the threshold ({flow_threshold})")

    return id_of_next_connection, connection_distances, element_distances, connection_pairing, compartment_network, volumetric_flows


def _group_facets_into_surfaces(facets: Set[int], mesh: CMesh) -> List[List[int]]:
    """
    Take a set of facets and group them together into several contiguous surfaces.

    Facets are considered as being connected if they share one entity of a dimension lower than them.
    E.g. For a 3D mesh, facets are surfaces. Two facets are considered connected if they share an edge.
    The creation of this connectivity is handled by the creation of the CMesh.
    This function queries the CMesh for which facet(s) a given facet is connected it.

    Parameters
    ----------
    * facets: The set of facets to group.
    * mesh:   The CMesh on which the facets below. Used to get their connectivity.

    Returns
    -------
    * surfaces:   A list of contiguous surfaces, each represented as a list of facet IDs.
    """
    surfaces: List[List[int]] = []

    while len(facets) > 0:
        new_surface = [facets.pop()]
        # The facet(s) last added to the compartment and whose connected facet must now be checked
        facet_ids_to_check = {new_surface[0]}

        while True:
            neighbours = set()
            # Get all the neighbours of the current facet
            for facet in facet_ids_to_check:
                neighbours.update(neighbour_facet for neighbour_facet in mesh.facet_connectivity[facet])

            # Remove all those that are not available (have already been added or where never part of the set)
            neighbours_to_add = facets.intersection(neighbours)

            if len(neighbours_to_add) == 0:
                break
            else:
                facet_ids_to_check = neighbours_to_add
                new_surface.extend(neighbours_to_add)
                facets.difference_update(neighbours_to_add)

        surfaces.append(new_surface)

    return surfaces
