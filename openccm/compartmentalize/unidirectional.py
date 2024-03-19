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

from typing import Dict, List, Set, Tuple

import numpy as np

from ..config_functions import ConfigParser
from ..mesh import CMesh


def create_compartment_network(compartments:    Dict[int, Set[int]],
                               mesh:            CMesh,
                               dir_vec:         np.ndarray,
                               vel_vec:         np.ndarray,
                               config_parser:   ConfigParser) \
        -> Tuple[
            Dict[int, Set[int]],
            Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]]
        ]:
    """
    This function takes identified compartments and converts them into a network of compartments.

    When creating the network, any compartments which are found to have only one connection are removed from the network.

    The resulting compartment network is cleaned up by merging together neighbouring compartments which are below
    a user-specified size threshold.

    Finally, the compartments are renumbered to the range [0, N) in a sequential order without gaps.

    Args:
        compartments:       A list of set. The ith set contains the indices of the elements belonging to the ith compartment.
        mesh:               The mesh from which the compartments were built.
        dir_vec:            Direction vector, NxD where N is the number of mesh elements and D the dimension of the mesh.
        vel_vec:            Velocity vector, NxD where N is the number of mesh elements and D the dimension of the mesh.
        config_parser:      The OpenCCM ConfigParser

    Returns:
        compartments:           Same as input, except renumbered and modified after merging small compartments
        compartment_network:    A dictionary whose keys are the compartment index and whose values are dictionary.
                                    For each of those dictionaries, the keys are the index of a neighboring compartment
                                    and the values are another dictionary.
                                        For each of those dictionaries, the keys are the index of the bounding facet between the two compartments,
                                        and the values are Tuples.
                                            The first entry in the tuple is the index of the element on the boundary inside the compartment.
                                            The second is a numpy array representing the unit normal for that boundary facing out of the compartment.
    """
    print("Creating compartment network")

    DEBUG           = config_parser.get_item(['SETUP', 'DEBUG'],            bool)
    log_folder_path = config_parser.get_item(['SETUP', 'log_folder_path'],  str)

    # Make a copy of compartments since this function will be changing it.
    compartments: Dict[int, Set[int]] = {_id: set(element_set) for _id, element_set in compartments.items()}

    # Dictionary to store the compartment network in
    compartment_network: Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]] = {i: dict() for i in range(len(compartments))}

    # A dictionary to quickly look up which compartment a given element is in
    element_to_compartment_map: Dict[int, int] = dict()
    for id_compartment in compartments:
        for element in compartments[id_compartment]:
            element_to_compartment_map[element] = id_compartment

    # Add BCs to the element compartment_map
    for _id in mesh.grouped_bcs.domain_inlets + mesh.grouped_bcs.domain_outlets + mesh.grouped_bcs.ignored:
        element_to_compartment_map[_id] = _id
    element_to_compartment_map[mesh.grouped_bcs.no_flux] = mesh.grouped_bcs.no_flux

    with open(log_folder_path + "network.txt", 'w') as logging:
        # Group the bounding entities of this compartment based on the compartment on the other side of them
        for id_compartment in list(compartments.keys()):
            if DEBUG: logging.write("[ Compartment {}:\n".format(id_compartment))

            # Get the bounding entities for the compartment
            bounding_facets_info = _calculate_compartment_bounds(compartments[id_compartment], id_compartment, mesh,
                                                                 element_to_compartment_map)
            if DEBUG: logging.write("[ bounding_facets_info: {}\n".format(bounding_facets_info))

            # Remove all bounding entities which are no-flux (or have 0 magnitude in them)
            ids_facets = list(bounding_facets_info.keys())  # Need to save seperately in order to modify the dictionary inside the loop
            for facet in ids_facets:
                # Get the "element" on the other side of the bounding facet.
                # If the facet is on the bounds of the mesh, then the element_id will be negative
                element_on_other_side_of_bound = bounding_facets_info[facet][1]

                if element_on_other_side_of_bound == mesh.grouped_bcs.no_flux or \
                        element_on_other_side_of_bound not in element_to_compartment_map:
                    bounding_facets_info.pop(facet)

            # If the compartment has 1 or fewer bounding entities left then it means that it does not have
            # a significant flow through it, and thus needs to be removed.
            # It's checked for 1 or fewer since it could end up with only one small flow,
            # which must be very close to zero if the others are no-flux.
            if len(bounding_facets_info) <= 1:
                id_compartment_chase = id_compartment

                # This list will either have 0 or 1 elements in it.
                ids_neighbour = [element_to_compartment_map[element_pairs[1]] for element_pairs in bounding_facets_info.values()]
                while True:
                    elements_in_compartment = compartments.pop(id_compartment_chase)
                    # We don't need the contents since it has 0 or 1 neighbours.
                    # If it's 0 neighbours this will be empty, and if it has 1 then all the info we need is in
                    # compartment_network[id_compartment_chase's neighbour]
                    compartment_network.pop(id_compartment_chase)

                    # Set the edges of this compartment as no-flux BC
                    # AND remove the element from the mapping since it's no longer in a compartment
                    for element in elements_in_compartment:
                        element_to_compartment_map.pop(element)
                        for facet in mesh.element_facets[element]:
                            mesh.facet_to_bc_map[facet] = mesh.grouped_bcs.no_flux

                    # If it has a single connection, also remove the connection from that compartment
                    if len(ids_neighbour) > 0:
                        id_neighbour = ids_neighbour[0]

                        # Neet to check this way since compartment_network is pre-filled with empty dictionaries.
                        if id_compartment_chase not in compartment_network[id_neighbour]:
                            break  # This can happen if id_neighbour has not been processes yet by the main for-loop
                        else:
                            compartment_network[id_neighbour].pop(id_compartment_chase)
                            # Need to check if removing this connection from the compartment results in a compartment
                            # with 1 or fewer connections.
                            num_connections = len(compartment_network[id_neighbour])
                            if num_connections == 0 or num_connections > 1:
                                # Nothing else we need to do
                                # If it has 0 connections then there's nothing to remove from its info
                                # And if it has more 1 then we also don't have to change it
                                break
                            else:  # num_connections == 1
                                id_compartment_chase = id_neighbour
                                ids_neighbour = [list(compartment_network[id_neighbour].keys())[0]]
            else:
                for facet in bounding_facets_info:
                    # Get the element on the other side of the bounding facet.
                    # If the facet is on the bounds of the mesh, then the element_id will be negative
                    element_on_other_side_of_bound = bounding_facets_info[facet][1]

                    # Calculate the outward facing normal
                    element_on_this_side_of_bound = bounding_facets_info[facet][0]
                    normal = mesh.get_outward_facing_normal(facet, element_on_this_side_of_bound)

                    # Get the compartment index of the compartment on the other side of the bounding facet
                    i_compartment_on_other_side_of_bound = element_to_compartment_map[element_on_other_side_of_bound]

                    # Add bounding facet and outward facing normal to compartment network
                    # Get the dictionary
                    d = compartment_network[id_compartment].get(i_compartment_on_other_side_of_bound, dict())
                    d[facet] = (element_on_this_side_of_bound, normal)
                    compartment_network[id_compartment][i_compartment_on_other_side_of_bound] = d

        # Check compartments
        for id_compartment in compartment_network:
            if DEBUG: logging.write("id: {} network: {}\n".format(id_compartment, compartment_network[id_compartment]))

            assert id_compartment not in compartment_network[id_compartment]

            if len(compartment_network[id_compartment]) == 0:
                assert id_compartment not in compartments
                compartment_network.pop(id_compartment)
            elif len(compartment_network[id_compartment]) == 1:
                # Only one neighbour, it could happen if one compartment is contained within another
                # Make sure that the two compartments share more than one facet
                neighbour_id = list(compartment_network[id_compartment].keys())[0]
                assert len(compartment_network[id_compartment][neighbour_id]) > 1
            else:
                # Make sure A lists B as a neighbour that B also lists A as a neighbour
                for id_neighbour in compartment_network[id_compartment]:
                    if id_neighbour > 0:
                        assert id_compartment in compartment_network[id_neighbour]

    merge_compartments(compartments, compartment_network, mesh, dir_vec, vel_vec, config_parser)
    renumber_compartments(compartments, compartment_network)

    print("Done creating compartment network")
    return compartments, compartment_network


def renumber_compartments(compartments:         Dict[int, Set[int]],
                          compartment_network:  Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]]) \
        -> None:
    """
    This function re-numbers the compartments from their current numbering, which can include holes for any compartments
    removed or merged, to [0, N-1] inclusive of both ends, where N is the number of compartments.

    Args:
        compartments:           See documentation in {calculate_compartments}
        compartment_network:    See documentation in {create_compartment_network}
    Returns:
        compartments:           Same data as the input, except with the compartments renumbered
        compartment_network:    Same data as the input, except with the compartments renumbered
    """
    print("Renumbering compartments")

    offset = max(compartments.keys()) + 1
    old_to_new_compartment_id_map: Dict[int, int] = {old: new for new, old in enumerate(compartments.keys())}
    old_to_tmp_compartment_id_map: Dict[int, int] = {old: new + offset for new, old in enumerate(compartments.keys())}

    # 1. Map compartments
    tmp_compartments: Dict[int, Set[int]] = {id_new: compartments[id_old] for id_old, id_new in old_to_new_compartment_id_map.items()}

    compartments.clear()
    for id_new, elements in tmp_compartments.items():
        compartments[id_new] = elements

    # 2. Map compartment network
    # 2.1 Need to map to outside the final range first
    tmp_compartment_network: Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]] = dict()
    for id_old, id_tmp in old_to_tmp_compartment_id_map.items():
        compartment_info = compartment_network.pop(id_old)
        tmp_compartment_network[id_tmp] = compartment_info
        # Renumber id_old to id_tmp inside the network info for each neighbour
        for id_neighbour in compartment_info:
            if id_neighbour in compartment_network:  # id_neighbour not yet processed (happens when id_neighbour > id_old)
                compartment_network[id_neighbour][id_tmp] = compartment_network[id_neighbour].pop(id_old)
            elif id_neighbour in tmp_compartment_network:  # id_neighbour already been processed (id_neighbour < id_old)
                tmp_compartment_network[id_neighbour][id_tmp] = tmp_compartment_network[id_neighbour].pop(id_old)
            elif id_neighbour < 0:
                # If the id is negative, then the neighbour is a BC and there's nothing to do.
                pass
            else:
                print(id_neighbour)
                print(id_old)
                raise Exception

    # 2.2 Map from temporary range back onto the final range
    # Note: Purposefully using .values here since the tmp ids are stored as values in old_to_tmp_compartment_id_map
    tmp_to_new_compartment_id_map: Dict[int, int] = {tmp: new for new, tmp in enumerate(old_to_tmp_compartment_id_map.values())}
    for id_tmp, id_new in tmp_to_new_compartment_id_map.items():
        compartment_info = tmp_compartment_network.pop(id_tmp)
        compartment_network[id_new] = compartment_info
        # Renumber id_old to id_tmp inside the network info for each neighbour
        for id_neighbour in compartment_info:
            if id_neighbour in tmp_compartment_network:
                # This will only happen when id_neighbour > id_old.
                # The neighbour compartment has not yet been processed
                # If id_neighbour < id_old, then id_neighbour has already been processed
                tmp_compartment_network[id_neighbour][id_new] = tmp_compartment_network[id_neighbour].pop(id_tmp)
            elif id_neighbour in compartment_network:
                # The neighbour has already been processed (id_neighbour < id_old)
                compartment_network[id_neighbour][id_new] = compartment_network[id_neighbour].pop(id_tmp)
            elif id_neighbour < 0:
                # If the id is negative, then the neighbour is a BC and there's nothing to do.
                pass
            else:
                raise Exception

    print("Done renumbering compartments")


def calculate_compartments(dir_vec:         np.ndarray,
                           mesh:            CMesh,
                           config_parser:   ConfigParser) \
        -> Tuple[Dict[int, Set[int]], Set[int]]:
    """
    Wrapper for _calculate_compartments to allow for numba
    This function takes the elements of a mesh and groups them together into compartments.

    Args:
        dir_vec:            The numpy array containing the direction vector.
        mesh:               The CMesh to compartmentalize
        config_parser:      ConfigParser to use.

    Returns:
        compartments:                       A dictionary. Keys are compartment ids (ints) and the corresponding value
                                            is a set containing the facet id (int) of each facet in this compartment.
        compartment_of_removed_elements:    A set containing the facet id of all faces which had a 0 velocity in them.
    """
    print("Calculating compartments")

    compartment_of_removed_elements, valid_elements = _remove_unwanted_elements(mesh, dir_vec)

    # Invert dictionary
    bc_names_for_seeds = config_parser.get_expression(['COMPARTMENTALIZATION', 'bc_names_for_seeds'])
    if bc_names_for_seeds is None:
        bc_names_for_seeds = []
    bcs_ids_for_seeds = set(mesh.grouped_bcs.id(bc_name) for bc_name in bc_names_for_seeds)
    bcs_for_seed_to_facet_map = {bc_id: np.where(mesh.facet_to_bc_map == bc_id)[0] for bc_id in bcs_ids_for_seeds}

    # Convert facet IDs to element IDs
    bc_elements_for_seed = []
    for bc_id, facet_ids in bcs_for_seed_to_facet_map.items():
        for id_facet in facet_ids:
            id_element = mesh.facet_elements[id_facet]
            assert len(id_element) == 1  # If on a BC, this facet should only be shared by one element
            bc_elements_for_seed.append(id_element[0])
    bc_elements_for_seed = list(set(bc_elements_for_seed).difference(compartment_of_removed_elements))

    # Wrapping function to allow for numba usage (which has not been implemented)
    compartment_list = _calculate_compartments(valid_elements, dir_vec, mesh, bc_elements_for_seed, config_parser)

    compartments = {i: compartment for i, compartment in enumerate(compartment_list)}

    print("Done calculating compartments")
    return compartments, compartment_of_removed_elements


def merge_compartments(compartments:        Dict[int, Set[int]],
                       compartment_network: Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]],
                       mesh:                CMesh,
                       dir_vec:             np.ndarray,
                       vel_vec:             np.ndarray,
                       config_parser:       ConfigParser) -> None:
    """
    This function takes the list of compartments and merges any compartments which are below the specified threshold
    into the neighbouring compartment with the closest average direction vector.

    Compartments which have only a single connection are also merged together.

    The threshold size has different interpretations depending on the dimension of the mesh:
    - 1D: Represents distance.
    - 2D: Represents area.
    - 3D: Represents volume.

    Args:
        compartments:           Dictionary representation of a compartment.
                                Keys are compartment ID, values are sets of element IDs.
        compartment_network:    A dictionary representation of the network.
        mesh:                   The mesh.
        dir_vec:                Array indexed by element ID for the direction vector.
        vel_vec:                Array indexed by element ID for the velocity vector.
        config_parser:          ConfigParser to use.
    """
    print("Merging small compartments")

    # Check compartments
    for compartment_id, network in compartment_network.items():
        assert len(network) > 0
        assert compartment_id not in network

    DEBUG                = config_parser.get_item(['SETUP',                'DEBUG'],                bool)
    log_folder_path      = config_parser.get_item(['SETUP',                'log_folder_path'],      str)
    # The minimum size of a compartment. Units are the same as those used by the mesh.
    min_compartment_size = config_parser.get_item(['COMPARTMENTALIZATION', 'min_compartment_size'], float)

    # Calculate the average direction vector for each compartment
    compartment_avg_directions = np.zeros((max(compartments.keys()) + 1, dir_vec.shape[1]))
    for i_compartment in compartments:
        compartment_avg_directions[i_compartment, :] = np.mean(dir_vec[list(compartments[i_compartment]), :], axis=0)
    magnitude = np.linalg.norm(compartment_avg_directions, axis=1)[:, np.newaxis]
    if np.any(magnitude == 0):
        raise Exception("Compartment with 0 velocity magnitude found")
    compartment_avg_directions /= magnitude

    # Calculate compartment sizes
    compartment_sizes = np.inf * np.ones(max(compartments.keys())+1)
    for id_compartment, compartment_elements in compartments.items():
        compartment_sizes[id_compartment] = sum(mesh.element_sizes[element] for element in compartment_elements)

    # Calculate the connections between compartments so that downstream and upstream can be calculated
    # NOTE: Using the PFR version, rather than the CSTR one (which calculates NET flow between two compartments)
    #       Since the net flow version my show that flow goes A -> B, but the PFR version may show multiple connections
    #       between A and B. This is a problem since this function must return results that can be used by both the
    #       PFR and CSTR modelling approach.
    connection_pairing = connect_compartments_using_unidirectional_assumptions(compartment_network, compartments, mesh, dir_vec, vel_vec, False, config_parser)[2]

    # Remove all connections that lead to a BC since we do not want to consider them for the purpose of merging compartments
    for _id_pfr_tmp in connection_pairing:
        _connections_for_pfr = connection_pairing[_id_pfr_tmp]
        for i_connection in list(_connections_for_pfr.keys()):  # Need to make the copy with list() since we're modifying the dict during the loop
            if _connections_for_pfr[i_connection] < 0:  # Negative PFR if represents a
                _connections_for_pfr.pop(i_connection)
        # If there are no connections available after removing the domain boundaries, then this compartment cannot be merged with something else
        if len(_connections_for_pfr) == 0:
            # Setting to infinite so that a numpy array indexed by compartment id can be used
            compartment_sizes[_id_pfr_tmp] = np.inf
    with open(log_folder_path + "merge.txt", 'w') as logging:
        if DEBUG:
            logging.write("[ Pre\n")
            for i, size in enumerate(compartment_sizes):
                logging.write("{}: {:.8f}\n".format(i, size))
            logging.write("]\n")
            logging.write("[ Merging\n")

        # Iterate over compartments, and merge together any that only have one connection.
        # At this point it is possible for compartments to have a single connection.
        # Only after the network has been built is it guaranteed that each compartment has at least 2 connections.
        compartments_with_one_connection = set()
        for id_compartment, network in compartment_network.items():
            if len(network) <= 1:
                compartments_with_one_connection.add(id_compartment)
        while len(compartments_with_one_connection) > 0:
            id_compartment = compartments_with_one_connection.pop()
            keys = list(compartment_network[id_compartment].keys())
            assert len(keys) == 1
            id_merge_into = keys[0]
            if id_merge_into in compartments_with_one_connection:
                raise AssertionError(f"Compartments {id_compartment} and {id_merge_into} are only connected to each other.")

            _merge_two_compartments(id_merge_into, id_compartment, compartments, compartment_network,
                                    compartment_sizes, connection_pairing, compartment_avg_directions,
                                    dir_vec)

        while np.any(compartment_sizes < min_compartment_size):
            # Find the smallest compartment
            id_smallest: int = np.argmin(compartment_sizes)
            if DEBUG: logging.write("smallest: {} ".format(id_smallest))
            compartment_value = compartment_avg_directions[id_smallest]
            """
            1. Filter those which do not have the flow in the correct direction
                1.1. If possible, merge only into downstream compartments (ones with negative connection id)
                1.2. If 1.1. is not possible, merge into an upstream compartment but print info
            """
            # keys are connection id, negative means that the connection is an outlet (and thus the other compartment is downstream of this one)
            # Values are compartment on the other side of the connection
            connections = connection_pairing[id_smallest]

            # Find all connections which lead to a downstream compartment
            downstream_connection_ids = []
            for i_connection in connections:
                if i_connection < 0:
                    downstream_connection_ids.append(i_connection)

            if len(downstream_connection_ids) > 0:
                # 1.1. If possible, merge only into downstream compartments (ones with negative connection id)
                id_neighbour_compartment = [connections[i_connection] for i_connection in downstream_connection_ids]
            else:
                # 1.2. If 1.1. is not possible, merge into an upstream compartment but print info
                id_neighbour_compartment = list(connections.values())
                print("Had to merge compartment {} into an upstream compartment since there were no downstream compartments".format(id_smallest))

            neighbour_values = compartment_avg_directions[id_neighbour_compartment]

            # 2. Pick the ones with the closest direction
            dot_products = neighbour_values.dot(compartment_value)
            id_merge_into = id_neighbour_compartment[np.argmax(dot_products)]

            if DEBUG: logging.write("merged into: {}\n".format(id_merge_into))
            # Merge this compartment with its neighbour
            _merge_two_compartments(id_merge_into, id_smallest, compartments, compartment_network,
                                    compartment_sizes, connection_pairing, compartment_avg_directions,
                                    dir_vec)

        if DEBUG: logging.write("]\n")

        # Check compartments
        for compartment_id, network in compartment_network.items():
            assert len(network) > 1
            assert compartment_id not in network

        if DEBUG:
            logging.write("[ Post\n")
            for i, size in enumerate(compartment_sizes):
                logging.write("{}: {:.8f}\n".format(i, size))
            logging.write("compartments_new: {}\n".format(compartments))
            logging.write("compartment_network_new: {}\n".format(compartment_network))
            logging.write("]\n")

    print("Done merging small compartments")


def connect_compartments_using_unidirectional_assumptions(compartment_network:  Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]],
                                                          compartments:         Dict[int, Set[int]],
                                                          mesh:                 CMesh,
                                                          dir_vec:              np.ndarray,
                                                          vel_vec:              np.ndarray,
                                                          final:                bool,
                                                          config_parser:        ConfigParser) \
        -> Tuple[int,
                 Dict[int, Dict[int, float]],
                 Dict[int, Dict[int, int]],
                 Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]],
                 Dict[int, Set[int]],
                 Dict[int, float]]:
    """
    Function to take a compartment network and compartments and create the connections between them.
    Calculates the flowrates and direction of flow between connected compartments.
    CAN RETURN MULTIPLE CONNECTIONS BETWEEN TWO COMPARTMENTS.

    A surface shared by two compartments is made up of facets, all facets below the specified threshold
    (flow_threshold_i) are removed from the surface.
    Then the surface is split into contiguous subsurfaces which all have the same direction of flow.
    Each subsurface is treated as a separate connection between the two compartments that it's between.

    Args:
        compartment_network:    A dictionary representation of the compartments in the network.
                                Keys are compartment IDs, and whose values are dictionary.
                                    For each of those dictionaries, the keys are the index of a neighboring compartment
                                    and the values are another dictionary.
                                        For each of those dictionaries, the keys are the index of the bounding facet
                                        between the two compartments, and the values Tuples.
                                            - The first entry in the tuple is the index of the element
                                              on the boundary inside the compartment.
                                            - The second entry in the tuple is a numpy array representing
                                              the unit normal for that boundary facing out of the compartment.
        compartments:           A dictionary representation of the elements in the compartments.
                                Keys are compartment IDs, values are sets containing the indices
                                of the elements in the compartment.
        mesh:                   The mesh the problem was solved on.
        dir_vec:                Numpy array of direction vectors, row i is for element i.
        vel_vec:                Numpy array of velocity vectors, row i is for element i.
        final:                  If asserts and all calculations should be performed.
                                This is set to True when function is called from merge_compartments since some may
                                be too small for the invariants to be true.
        config_parser:          The OpenCCM ConfigParser

    Returns:
        Tuple of two values:
            id_next_connection:     The integer to use for the next connection ID.
            connection_distances:   Dictionary storing the distance.
            connection_pairing:     Dictionary storing info about which other compartments a given compartment is connected to.
                                        - First key is compartment ID
                                        - Values is a Dict[int, int]
                                            - Key is connection ID (positive inlet into this compartment, negative is outlet)
                                            - Value is the ID of the compartment on the other side
            compartment_network:    The compartment network passed in.
            compartments            The compartments passed in.
            volumetric_flows:       Dictionary of the magnitude of volumetric flow through each connection,
                                    indexed by connection ID.
                                    Connection ID in this dictionary is ALWAYS positive, need to take absolute sign of
                                    the value if it's negative (see `connection_pairing` docstring).
    """
    print("Finding connections between compartments")

    # The minimum volumetric flow required between two compartments for a connection to be added between them.
    flow_threshold          = config_parser.get_item(['COMPARTMENT MODELLING', 'flow_threshold'],       float)
    # The minimum volumetric flow through a facet for it to considered, facets with flow below this amount are removed.
    flow_threshold_facet    = config_parser.get_item(['COMPARTMENT MODELLING', 'flow_threshold_facet'], float)
    log_folder_path         = config_parser.get_item(['SETUP',                 'log_folder_path'],      str)
    DEBUG                   = config_parser.get_item(['SETUP',                 'DEBUG'],                bool)

    # Dictionary used to store the merged connection information
    connection_distances: Dict[int, Dict[int, float]] = dict()
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
    with open(log_folder_path + 'connect' + ('' if final else '_for_merge') + '.txt', 'w') as logging:
        # Calculate the average direction of each compartment
        avg_directions: Dict[int, np.ndarray] = dict()
        for id_compartment in compartment_network:
            avg_directions[id_compartment] = np.mean(dir_vec[sorted(compartments[id_compartment])], 0)
            avg_directions[id_compartment] /= np.linalg.norm(avg_directions[id_compartment])

        if DEBUG:
            logging.write("avg_direction\n")
            for i, flow in avg_directions.items():
                logging.write("{}: {}\n".format(i, flow))

        # Calculate the number of connections between compartments and their locations
        for id_compartment in compartment_network:
            # Inlets and outlets for this compartment.
            #   Key is connection ID, value is the ID of the other compartment
            #   A negative key represents an outlet and a positive an inlet
            compartment_connections: Dict[int, int] = dict()

            avg_direction = avg_directions[id_compartment]

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
                    neighbour_dict: Dict[int, Tuple[int, np.ndarray]] = compartment_network[id_compartment][neighbour]

                    ###
                    # 1. Calculate and store the flowrate through each facet.
                    ###
                    flow_through_facet: Dict[int, float] = dict()
                    for id_facet in neighbour_dict:
                        element_on_this_side_of_bound_id, normal = neighbour_dict[id_facet]
                        velocity_vector = vel_vec[element_on_this_side_of_bound_id]
                        flux = np.dot(velocity_vector, normal)
                        flow_through_facet[id_facet] = flux * mesh.facet_size[id_facet]

                    ###
                    # 2. Calculate the total flowrate with all facets.
                    ###
                    total_flow = sum(flow_through_facet.values())

                    ###
                    # 3. If it's below the total flow threshold, delete this connection and stop.
                    ###
                    if np.abs(total_flow) < flow_threshold:
                        continue

                    ###
                    # 4. Mark and remove all facets which are below the individual flow threshold.
                    ###
                    for id_facet in list(flow_through_facet.keys()):
                        if np.abs(flow_through_facet[id_facet]) < flow_threshold_facet:
                            flow_through_facet.pop(id_facet)

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
                    for id_facet, flow_value in flow_through_facet.items():
                        # Flow calculated with normal facing out of the compartment
                        # Negative values are in the opposite direction of the
                        if flow_value < 0:
                            inlet_facets.add(id_facet)
                        else:
                            outlet_facet.add(id_facet)

                    surfaces: List[List[int]] = []
                    surfaces.extend(_group_facets_into_surfaces(inlet_facets, mesh))
                    surfaces.extend(_group_facets_into_surfaces(outlet_facet, mesh))

                    ###
                    # 7. For each resulting contiguous surface calculate the total flowrate through it and if its an inlet/outlet
                    ###
                    flow_through_surfaces: List[float] = [sum(flow_through_facet[id_facet] for id_facet in surface) for surface in surfaces]
                    flow_through_surfaces = [total_flow * flow_i / total_flow_thresholded for flow_i in flow_through_surfaces]

                    is_inlet = [flow < 0 for flow in flow_through_surfaces]

                    if final:
                        # Make all flows positive, the inlet/outlet information is stored separately
                        flow_through_surfaces = list(np.abs(flow_through_surfaces))
                        abs_flow_through_surfaces: List[float] = [sum(abs(flow_through_facet[id_facet]) for id_facet in surface) for surface in surfaces]

                        ###
                        # 8. Calculate the center of flow for each surface.
                        ###
                        centers_of_flow: List[np.ndarray] = []
                        for i, surface in enumerate(surfaces):
                            center_of_flow = np.sum([abs(flow_through_facet[facet]) * mesh.facet_centers[facet] for facet in surface], 0)
                            center_of_flow /= abs_flow_through_surfaces[i]
                            centers_of_flow.append(center_of_flow)

                    ###
                    #  9. Store the results
                    ###
                    for i in range(len(flow_through_surfaces)):
                        if final:
                            centers_of_flow_for_connection[id_of_next_connection] = centers_of_flow[i]
                            volumetric_flows[id_of_next_connection] = flow_through_surfaces[i]
                        if is_inlet[i]:
                            compartment_connections[id_of_next_connection] = neighbour
                        else:
                            compartment_connections[-id_of_next_connection] = neighbour

                        id_of_next_connection += 1

            if final:
                # Must have at least one inlet or outlet. Otherwise, something has gone horrible wrong.
                assert len(compartment_connections) > 1

                # Calculate the ordering of the inlets/outlets
                # Project centers of flux onto average direction
                # Min and max values for normalizing between 0.0 and 1.0
                # (setting to inf and -inf so that the min and max always work).
                dot_min = np.inf
                dot_max = -np.inf
                connections_distances_i: Dict[int, float] = dict()
                for id_connection in compartment_connections:
                    center_of_flow = centers_of_flow_for_connection[abs(id_connection)]
                    dot_result = avg_direction.dot(center_of_flow)

                    dot_min = min(dot_min, dot_result)
                    dot_max = max(dot_max, dot_result)
                    connections_distances_i[id_connection] = dot_result

                # Normalize between 0.0 and 1.0
                for id_connection in connections_distances_i:
                    connections_distances_i[id_connection] = (connections_distances_i[id_connection] - dot_min) / (
                                dot_max - dot_min)

                connection_distances[id_compartment] = connections_distances_i
            connection_pairing[id_compartment] = compartment_connections

        if DEBUG:
            logging.write("id_of_next_connection: {}\n".format(id_of_next_connection))
            logging.write("connection_pairing: {}\n".format(connection_pairing))
            logging.write("compartment_network: {}\n".format(compartment_network))
            logging.write("compartments: {}\n".format(compartments))
            logging.write("volumetric_flows: {}\n".format(volumetric_flows))

    print("Done finding connections between compartments")
    return id_of_next_connection, connection_distances, connection_pairing, compartment_network, compartments, volumetric_flows


def _calculate_compartments(elements_not_in_a_compartment:  Set[int],
                            n_vec:                          np.ndarray,
                            mesh:                           CMesh,
                            bc_facets_for_seed:             List[int],
                            config_parser:                  ConfigParser) \
        -> List[Set[int]]:
    """
    Wrapped function to allow for numba usage.

    Iterating one element at a time, group neighsabouring elements into compartments.
    The algorith works as follows:
        1. (Arbitrarily) pick a seed element from the list of seed elements
            - This list is original the list of all elements which share a facet with certain user-specified boundaries
        2. Find all neighbours not yet in a compartment
            2.1 If there are no neighbours go to 6.
        3. Compare the alignment of the seed element to that of all the compartment's neighbours
            3.1 If there are no neighbours above the specified threshold go to 6.
        4. Compare the angle between the normal of the surface between each passing neighbour
           and the director of the cell inside the compartment on the other side
            4.1 If there are no neighbours above the specified threshold go to 6.
        5. Add the element from 4 to this compartment, and go back to 2.
        6. The compartment is done, save it and go back to 1.

    Args:
        elements_not_in_a_compartment:  The set of elements which are available for compartmentalization.
        n_vec:                          The direction vector, NxD where N is the number of elements in the mesh
                                        and D the dimension of the mesh.
        mesh:                           The mesh on which the compartmentalization is occurring.
        bc_facets_for_seed:             List of facets IDs to use as the initial seeds.
        config_parser:                  ConfigParser from which to get parameters.
    """
    ####################################################################################################################
    # 0. Setup
    ####################################################################################################################
    # Return variable
    angle_threshold = config_parser.get_item(['COMPARTMENTALIZATION', 'angle_threshold'], int)
    flow_threshold  = config_parser.get_item(['COMPARTMENTALIZATION', 'flow_threshold'],  int)
    log_folder_path = config_parser.get_item(['SETUP',                'log_folder_path'], str)
    DEBUG           = config_parser.get_item(['SETUP',                'DEBUG'],           bool)

    compartments: List[Set[int]] = []

    # Copy since we'll be changing it
    seeds = bc_facets_for_seed.copy()

    with open(log_folder_path + 'compartments.txt', 'w') as logging:
        while len(elements_not_in_a_compartment) > 0:
            ################################################################################################################
            # 1. Pick the first element from the list of seed elements
            ################################################################################################################
            if DEBUG: logging.write("[ Compartment {}:\n".format(len(compartments)))
            if DEBUG: logging.write("* All seeds: {}\n".format(seeds))

            if len(seeds) == 0:
                element_seed = elements_not_in_a_compartment.pop()
            else:
                element_seed = seeds.pop(0)
                elements_not_in_a_compartment.remove(element_seed)

            if DEBUG: logging.write("* Seed used: {}".format(element_seed))
            compartment_curr = {element_seed}

            # Get the direction vector of the compartment
            director_seed = n_vec[element_seed, :]

            ################################################################################################################
            # 2. Find all neighbours not yet in a compartment
            ################################################################################################################
            # Get the neighbouring elements which are not in a compartment
            neighbouring_candidates = elements_not_in_a_compartment.intersection(mesh.element_connectivity[element_seed])
            # The elements which have already been rejected and can't be added to this compartment
            rejected_for_compartment: Set[int] = set()

            ######
            # 2.1 If there are no neighbours go to 6.
            ######
            if len(elements_not_in_a_compartment) < 0:
                if DEBUG: logging.write("No elements left")
            if len(neighbouring_candidates) < 0:
                if DEBUG: logging.write("No neighbours")

            num_iter = 0
            while len(elements_not_in_a_compartment) > 0 and len(neighbouring_candidates) > 0:
                neighbouring_candidates_np = np.array(list(neighbouring_candidates))
                if DEBUG: logging.write("{}. Candidates: {} ".format(num_iter, sorted(neighbouring_candidates)))
                # Get the direction vector of the neighbouring elements
                director_neighbours = n_vec[neighbouring_candidates_np, :]

                ############################################################################################################
                # 3. Compare the alignment of the seed element to that of all the neighbours
                ############################################################################################################
                # Calculate the dot product
                angle_result = np.arccos(director_neighbours.dot(director_seed)) * 180/np.pi

                # Compare the dot product to the tolerance
                check = angle_result <= angle_threshold

                ######
                # 3.1 If there are no neighbours above the specified threshold go to 6.
                ######
                _new_seeds = []
                if ~np.any(check):
                    if DEBUG: logging.write(' - Failed 3.1\n')
                    for neighbour in neighbouring_candidates:
                        if neighbour not in seeds:
                            _new_seeds.append(neighbour)

                    if len(_new_seeds) > 0:
                        _new_seeds.sort()
                        if DEBUG: logging.write("\t new seeds: {}\n".format(_new_seeds))
                        seeds.extend(_new_seeds)
                    break
                neighbours_rejected = neighbouring_candidates_np[~check]
                neighbouring_candidates_np = neighbouring_candidates_np[check]

                ############################################################################################################
                # 4. Compare the angle between the normal of the surface between each passing neighbour and
                #    the director of the cell inside the compartment on the other side
                ############################################################################################################
                check = np.zeros(neighbouring_candidates_np.shape, dtype=np.bool_)

                # NOTE: Only blacklist if it fails the first check.
                #       DO NOT blacklist based on this one
                for i, id_neighbour_i in enumerate(neighbouring_candidates_np):
                    check[i] = _check_flow_requirement(id_neighbour_i, compartment_curr, n_vec, mesh, flow_threshold)

                ######
                # 4.1 If there are no neighbours above the specified threshold go to 6.
                ######
                _new_seeds = []
                if ~np.any(check):
                    if DEBUG: logging.write(' - Failed 4.1\n')
                    for neighbour in neighbouring_candidates:
                        if neighbour not in seeds:
                            _new_seeds.append(neighbour)

                    if len(_new_seeds) > 0:
                        _new_seeds.sort()
                        if DEBUG: logging.write("\t new seeds: {}\n".format(_new_seeds))
                        seeds.extend(_new_seeds)
                    break
                neighbours_to_add = neighbouring_candidates_np[check]
                if DEBUG: logging.write("Added: {}\n".format(sorted(e for e in neighbours_to_add)))
                ############################################################################################################
                # 5. Add the element from 4 to this compartment, and go back to 2.
                ############################################################################################################
                # Add/remove neighbours_to_add to/from the relevant sets
                elements_not_in_a_compartment.difference_update(neighbours_to_add)
                compartment_curr.update(neighbours_to_add)
                for neighbour in neighbours_to_add:
                    if neighbour in seeds:
                        seeds.remove(neighbour)

                # Add/remove the rejected neighbours to/from the relevant sets
                rejected_for_compartment.update(neighbours_rejected)
                _new_seeds = []
                for neighbour in neighbours_rejected:
                    if neighbour not in seeds:
                        _new_seeds.append(neighbour)
                if len(_new_seeds) > 0:
                    _new_seeds.sort()
                    if DEBUG: logging.write("\t new seeds @5: {}\n".format(_new_seeds))
                    seeds.extend(_new_seeds)

                # Add (neighbour to add)'s available neighbour to the list of available neighbours for this compartment
                all_neighbours = set()
                for neighbour in neighbours_to_add:
                    all_neighbours.update(mesh.element_connectivity[neighbour])

                # Find the next set of neighbours to consider:
                #   1. Neighbours of compartment
                #   2. NOT already in a compartment (i.e. still in elements_not_in_a_compartment)
                #   3. NOT yet rejected (i.e. not in rejected_for_compartment)
                neighbouring_candidates = elements_not_in_a_compartment.intersection(all_neighbours).difference(rejected_for_compartment)

                num_iter += 1  # Only for logging purposes

            ################################################################################################################
            # 6. The compartment is done, save it and go back to 1.
            ################################################################################################################
            compartments.append(compartment_curr)
            if DEBUG: logging.write(']\n')

    return compartments


def _remove_unwanted_elements(mesh: CMesh, dir_vec: np.ndarray) -> Tuple[Set[int], Set[int]]:
    """
    Function to take in the mesh and remove unwanted elements from the element connectivity matrix, thus removing them
    from consideration.

    NOTE: This function modifies the mesh object!

    Elements are removed if they meet any of the following criteria:
    1. Have a velocity magnitude of 0.
    2. Have any facet on a no-slip boundary condition
    3. Elements that end up with only one neighbouring element after the above criteria have been applied.

    Args:
        mesh:               The mesh to operate on.
        dir_vec:            The velocity direction vector for the mesh, indexed by element ID.

    Returns:
        removed_elements:   A set containing the IDs for removed elements.
        valid_elements:     A set containing the IDs of the elements that are still valid for compartmentalization.

    """
    no_slip_facets = np.where(mesh.facet_to_bc_map == mesh.grouped_bcs.no_flux)[0]

    # Remove the elements which have a magnitude of 0
    magnitude = np.linalg.norm(dir_vec, axis=1)
    removed_elements = set(np.where(magnitude == 0)[0])

    # Remove elements which are on a no-slip boundary condition
    for facet in no_slip_facets:
        # A facet on a mesh boundary should only have one element
        elements = mesh.facet_elements[facet]
        assert len(elements) == 1

        # 1. Add the element which has this facet
        removed_elements.add(elements[0])

        # 2. Add all elements that has any of the vertices from that facet
        bc_vertices = set(mesh.facet_vertices[facet])
        for element_neighbour in mesh.element_connectivity[elements[0]]:
            if len(bc_vertices.intersection(mesh.element_vertices[element_neighbour])) > 0:
                removed_elements.add(element_neighbour)

    # A set to hold all elements which have had at least one element removed as a neighbour
    # Entries are added when one of their neighbours has been removed from the mesh
    # Entries are removed when they have been checked to ensure they still have at least 2 neighbours left
    # At least one of those neighbours can be a flux-bc
    # By the time an element has been added to its list, the neighbour that was removed is now longer in the mesh
    # and so it will no longer show up as a neighbour.
    elements_with_neighbour_removed: Set[int] = set()

    # Remove all of these elements from the mesh connectivity
    for element in removed_elements:
        for element_neighbour in mesh.element_connectivity.pop(element):
            mesh.element_connectivity[element_neighbour].remove(element)
            if element_neighbour not in removed_elements:
                elements_with_neighbour_removed.add(element_neighbour)

    # Remove all elements which have only 1 (neighbour + BC)
    # NOTE: elements on a boundary will have 1 fewer neighbour per BC that they're on
    # To start, the element elements in here are those neighbouring an element with 0 velocity or which have a facet on
    # a no flux-BC.
    #   - None of the elements in `elements_with_neighbour_removed` or left in `valid_elements` are on a no-flux BC
    #   - All elements in `elements_with_neighbour_removed` have had at least one of their neighbours removed
    while len(elements_with_neighbour_removed) > 0:
        element = elements_with_neighbour_removed.pop()
        num_neighbours = len(mesh.element_connectivity[element])
        if num_neighbours > 1:
            # If it has more than 1 neighbour, we don't have to look at the number of BCs that it's on.
            continue

        num_bcs = 0
        for facet in mesh.element_facets[element]:
            if mesh.facet_to_bc_map[facet] < 0:
                num_bcs += 1

        # If the element is connected to only one other element/flux-BC we have to remove it.
        if num_neighbours + num_bcs <= 1:
            removed_elements.add(element)
            for element_neighbour in mesh.element_connectivity.pop(element):
                elements_with_neighbour_removed.add(element_neighbour)
                mesh.element_connectivity[element_neighbour].remove(element)

    # Create a set of all elements and remove the removed elements from it
    valid_elements = set(i for i in range(dir_vec.shape[0]))
    valid_elements.difference_update(removed_elements)

    assert len(valid_elements) > 0

    return removed_elements, valid_elements


def _merge_two_compartments(id_merge_into:              int,
                            id_to_merge:                int,
                            compartments:               Dict[int, Set[int]],
                            compartment_network:        Dict[int, Dict[int, Dict[int, Tuple[int, np.ndarray]]]],
                            compartment_sizes:          np.ndarray,
                            connection_pairing:         Dict[int, Dict[int, int]],
                            compartment_avg_directions: np.ndarray,
                            dir_vec:                    np.ndarray) -> None:
    """
    Helper function to perform the merging of two compartments and update the relevant places.

    If merging these two compartments would result in another compartment having a single neighbour the function
    will merge that other compartment too.
    This process will iterate until all compartments connected to id_merge_into and id_to_merge have more than
    one neighbour.

    Args:
        id_merge_into:              The initial ID of the compartment to merge into.
        id_to_merge:                The initial ID of the compartment to merge.
        compartments:               Dictionary representation of a compartment.
                                    Keys are compartment ID, values are sets of element IDs.
        compartment_network:        A dictionary representation of the compartments in the network.
                                    Keys are compartment IDs, and whose values are dictionary.
                                        For each of those dictionaries, the keys are the index of a neighboring compartment
                                        and the values are another dictionary.
                                            For each of those dictionaries, the keys are the index of the bounding facet
                                            between the two compartments, and the values Tuples.
                                                - The first entry in the tuple is the index of the element
                                                  on the boundary inside the compartment.
                                                - The second entry in the tuple is a numpy array representing
                                                  the unit normal for that boundary facing out of the compartment.
        compartment_sizes:          The size of each compartment, indexed by compartment ID.
        connection_pairing:         Dictionary storing info about which other compartments a given compartment is connected to
                                    - First key is compartment ID
                                    - Values is a Dict[int, int]
                                        - Key is connection ID (positive inlet into this compartment, negative is outlet)
                                        - Value is the ID of the compartment on the other side
        compartment_avg_directions: Average direction vector for each compartment, indexed by compartment ID.
        dir_vec:                    Direction vector for each compartment, indexed by compartment ID.
    """
    compartments_with_one_connections = set()

    # Merging two compartments can result in one of their neighbours now having a single neighbour.
    # We will keep iterating and merging until there are no such neighbours left
    while True:
        compartments[id_merge_into].update(compartments.pop(id_to_merge))

        # Update compartment sizes
        compartment_sizes[id_merge_into] += compartment_sizes[id_to_merge]
        # Setting to infinity to ensure it's above the min_compartment_size
        # Doing this so that I don't have to recompute compartment_sizes and can continue using compartment id to index
        # into compartment_sizes.
        compartment_sizes[id_to_merge] = np.inf

        # Update connections between compartments
        for id_connection in list(connection_pairing[id_to_merge].keys()):
            # Remove entries from under id_smallest
            id_neighbour = connection_pairing[id_to_merge].pop(id_connection)
            if id_neighbour == id_merge_into:
                # Remove the entry for id_smallest from id_merge_into's dict
                connection_pairing[id_merge_into].pop(-id_connection)
            else:
                # Take all connections with other compartments and hook them up to id_merge_into
                connection_pairing[id_merge_into][id_connection] = id_neighbour
                connection_pairing[id_neighbour][-id_connection] = id_merge_into

        assert len(connection_pairing[id_to_merge]) == 0
        connection_pairing.pop(id_to_merge)

        # Update compartment averages
        compartment_avg_directions[id_merge_into, :] = np.mean(dir_vec[list(compartments[id_merge_into]), :], axis=0)

        # Normalize the average direction vector
        magnitude = np.linalg.norm(compartment_avg_directions[id_merge_into, :])
        if magnitude == 0:
            raise Exception("Compartment with 0 velocity magnitude found after merging: ID {}".format(id_merge_into))
        compartment_avg_directions[id_merge_into, :] /= magnitude

        # Compartment network
        for id_neighbour in compartment_network[id_to_merge]:
            if id_neighbour == id_merge_into:
                # Remove id_to_merge from the list of neighbours of id_merge_into
                compartment_network[id_merge_into].pop(id_to_merge)
                # No need to do anything else
                continue

            # if the neighbour is not shared by id_merge_into, add an empty dictionary for it so that the dictionary
            # merging syntax {**a, **b} can be used.
            if id_neighbour not in compartment_network[id_merge_into]:
                compartment_network[id_merge_into][id_neighbour] = dict()
                if id_neighbour >= 0:
                    compartment_network[id_neighbour][id_merge_into] = dict()

            # Update shared bounding facets
            # NOTE: We don't use a .pop() here so that compartment_network[id_to_merge] does not change size.
            #       This is needed in order to be able to iterate through the keys.
            compartment_network[id_merge_into][id_neighbour] = {**compartment_network[id_merge_into][id_neighbour],
                                                                **compartment_network[id_to_merge][id_neighbour]}

            # Update information in id_neighbour to point to id_merge_into instead of id_to_merge
            # But only do so if id_neighbour is not a mesh boundary
            if id_neighbour >= 0:
                compartment_network[id_neighbour][id_merge_into] = {**compartment_network[id_neighbour][id_merge_into],
                                                                    **compartment_network[id_neighbour].pop(id_to_merge)}

                if len(compartment_network[id_neighbour]) == 1:
                    compartments_with_one_connections.add(id_neighbour)

        # Remove the merged compartment from the network
        compartment_network.pop(id_to_merge)

        if len(compartment_network[id_merge_into]) == 1:
            compartments_with_one_connections.add(id_merge_into)

        if len(compartments_with_one_connections) == 0:
            break  # Only way out of the loop
        else:
            id_to_merge = compartments_with_one_connections.pop()
            keys = list(compartment_network[id_to_merge].keys())
            assert len(keys) == 1
            id_merge_into = keys[0]


def _check_flow_requirement(element: int, compartment: Set[int], vec: np.ndarray, mesh: CMesh, flow_threshold: float) -> bool:
    """
    Check if the provided face has enough flow between it and the compartment to be considered part of it
    AND that flow is going in the correct direction.

    The flow must go from the compartment INTO this new element, thus new elements are always downstream of those already in the compartment.

    This algorithm uses the amount of flow in and out of the cell to make this decision.
    id_element is considered to have enough flow into/out of a compartment if the following two criteria are met:
        1. The flow in/out of the cell across the facet shared with the compartment is at least (flow_threshold)%
           of the total flow in/out of the id_element.
        2. The flow in/out of the cell across the facet shared with the compartment is at least (flow_threshold)%
           of the total flow in/out of A neighbouring face of id_element inside the compartment.

    Args:
        element:        Integer representing the number (id) of the element that is being considered
        compartment:    The compartment we wish to try to add id_element to.
        vec:            The velocity vector on the mesh (currently must be a 0th order approximation)
        mesh:           The mesh, needed for finding neighbouring elements and facets
        flow_threshold: The minimum % of an element's total inflow/outflow needed over the connection.

    Returns:
        ~: Boolean indicating if id_element passed the flow requirement to be added to the compartment.
    """
    # Find the element(s) inside the compartment which are neighbours to neighbour_i and the face they have in common

    ####################################################################################################################
    # 1. Generate pairing of neighbour_id:shared_facet
    ####################################################################################################################
    #####
    # 1.1 Generate neighbour_id:shared_facet pairings
    #####
    # Keys are facet ids (int) of id_element which have a face inside the compartment on the other side.
    # The value for each key is the element id (int) which is on the other side of the facet and inside the compartment
    facet_to_neighbour_element_map: Dict[int, int] = dict()
    for facet in mesh.element_facets[element]:
        # Get the element shared by this facet which are inside the compartment
        #   - Can be 0 if the other face is not in the compartment
        elements: Set[int] = compartment.intersection(mesh.facet_elements[facet])
        if len(elements) > 0:
            facet_to_neighbour_element_map[facet] = elements.pop()

    ####
    # 1.2 Generate flowrate information for each edge of the face
    ####
    # The volumetric flowrate through each edge
    flow_for_facet: Dict[int, float] = dict()
    total_flow_this_element = 0.0
    director_this_face = vec[element, :]
    for facet in mesh.element_facets[element]:
        normal_to_facet = mesh.get_outward_facing_normal(facet, element)
        length_of_facet = mesh.facet_size[facet]
        flow_through_edge_i = director_this_face.dot(normal_to_facet) * length_of_facet
        flow_for_facet[facet] = flow_through_edge_i
        total_flow_this_element += np.abs(flow_through_edge_i)

    # Need to divide by 2 since so far it has been calculating the sum of the absolute value of the
    total_flow_this_element /= 2

    ####################################################################################################################
    # 2. Remove the edges of id_element which do not have flow coming into it in the right direction.
    #    Since the normal used for the flux is the outward normal, flow coming in will be negative
    ####################################################################################################################
    for facet in list(flow_for_facet.keys()):
        if flow_for_facet[facet] >= 0:
            flow_for_facet.pop(facet)
            if facet in facet_to_neighbour_element_map:
                facet_to_neighbour_element_map.pop(facet)
        else:
            # If it's not being removed, then invert it so that it's positive for later calculations
            flow_for_facet[facet] = -flow_for_facet[facet]

    ####################################################################################################################
    # 3. Check which edge of id_element have a flow through them above the threshold
    ####################################################################################################################
    # Remove any facets which do not pass the flow threshold
    for facet in list(flow_for_facet.keys()):
        if flow_for_facet[facet] / total_flow_this_element * 100 <= flow_threshold:
            flow_for_facet.pop(facet)
            if facet in facet_to_neighbour_element_map:
                facet_to_neighbour_element_map.pop(facet)

    if len(flow_for_facet) == 0:
        return False  # No facets passed the flow requirement

    ####################################################################################################################
    # 4. Of the remaining shared_facets, check which of them have flow above the threshold for the neighbouring element
    ####################################################################################################################
    for facet, element_neighbour in facet_to_neighbour_element_map.items():
        director_in_face: np.ndarray = vec[element_neighbour, :]

        flow_through_shared_edge = np.inf

        # Calculate the total flow into/out of the element
        total_flow = 0.0
        for facet_neighbour in mesh.element_facets[element_neighbour]:
            normal_to_facet = mesh.get_outward_facing_normal(facet_neighbour, element_neighbour)
            length_of_facet = mesh.facet_size[facet_neighbour]
            flow_through_facet = np.abs(director_in_face.dot(normal_to_facet)) * length_of_facet

            total_flow += flow_through_facet

            if facet_neighbour == facet:
                flow_through_shared_edge = flow_through_facet

        # Since we add the absolute value, the current value will be 2*flux_in (or 2*flux_out)
        total_flow /= 2

        if flow_through_shared_edge / total_flow * 100 >= flow_threshold:
            return True

    return False


def _calculate_compartment_bounds(compartment: set[int], id_compartment: int, mesh: CMesh,
                                  element_to_compartment_map: Dict[int, int]) \
        -> Dict[int, Tuple[int, int]]:
    """
    This function calculates which entities make up the bounds of the passed in compartment.

    This function assumes that the mesh does NOT have hanging nodes.

    Entities that bound an element are shared by up two different elements.
        - If it only has one element, then the facet is on the boundary of the mesh, and is therefore a bound of the compartment
        - If only one of the elements is inside the compartment, then the facet is a bound for the compartment.
        - If both of those elements are inside the compartment, then the facet is not a bound for the compartment.

    Args:
        compartment:                A set of integers representing the elements number in a compartment.
        id_compartment:             The id of the compartment.
        mesh:                       The mesh containing the compartment.
        element_to_compartment_map: A dictionary mapping each element to the compartment it is in.

    Returns:
        ~:  A dictionary whose keys are the bounding facet number and values a tuple of two integers.
            The first integer of the tuple is the index of the element on this side of the bounding facet (i.e. inside the compartment).
            The second integer of the tuple is the index of the element on the other side of the bounding facet (i.e. outside the compartment).
            NOTE: A value of -1 is used for the second index of entities which are on the boundary of the mesh.
            The type of these bounds depends on the dimension of the mesh:
                1D: Bounds are points.
                2D: Bounds are edges.
                3D: Bounds are faces.
    """
    # Indices of the bounding entities for the compartment
    bounding_facets: Dict[int, Tuple[int, int]] = dict()

    # Get all facets in the compartment
    all_facets: Set[int] = set()
    for element in compartment:
        for facet in mesh.element_facets[element]:
            all_facets.add(facet)

    # Iterate through each facet in the compartment and see if it's on the boundary
    for facet in all_facets:
        elements = mesh.facet_elements[facet]
        assert len(compartment.intersection(elements)) > 0

        # A facet is connected to 1 or 2 elements. Facets with 1 element are on the mesh boundary
        if len(elements) == 1:  # On the boundary of mesh and therefore on the boundary of the compartment
            bounding_facets[facet] = (elements[0], mesh.facet_to_bc_map[facet])
        else:  # Since there are 2 elements, need to check if both are in the compartment
            # NOTE: The "not in" check accounts for elements which had a velocity magnitude of 0
            #       and were removed from the set of elements.
            if elements[0] not in element_to_compartment_map or element_to_compartment_map[elements[0]] != id_compartment:
                # Element 0 is not in the compartment, therefore the facet bounds the compartment
                bounding_facets[facet] = (elements[1], elements[0])
            else:
                if elements[1] not in element_to_compartment_map or element_to_compartment_map[elements[1]] != id_compartment:
                    # Element 1 is not in the compartment, therefore the facet bounds the compartment
                    bounding_facets[facet] = elements
                else:
                    # Both elements are in the compartment, therefore the facet is NOT a bound of the compartment
                    pass

    return bounding_facets


def _group_facets_into_surfaces(facets: Set[int], mesh: CMesh) -> List[List[int]]:
    """
    Take a set of facets and group them together into several contiguous surfaces.

    Facets are considered as being connected if they share one entity of a dimension lower than them.
    E.g. For a 3D mesh, facets are surfaces. Two facets are considered connected if they share an edge.
    The creation of this connectivity is handled by the creation of the CMesh.
    This function queries the CMesh for which facet(s) a given facet is connected it.

    Args:
        facets: The set of facets to group.
        mesh:   The CMesh on which the facets below. Used to get their connectivity.

    Returns:
        surfaces:   A list of contiguous surfaces, each represented as a list of facet IDs.

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
