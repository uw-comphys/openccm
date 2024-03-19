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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np

from ..config_functions import ConfigParser
from ..mesh import CMesh


def network_to_rtd(system_results:   Tuple[
                                        np.ndarray,
                                        np.ndarray,
                                        Dict[int, List[Tuple[int, int]]],
                                        Dict[int, List[Tuple[int, int]]]],
                   c_mesh:           CMesh,
                   config_parser:    ConfigParser,
                   network:          Tuple[
                                        Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
                                        np.ndarray,
                                        np.ndarray,
                                        Dict[int, List[int]]]
                   ) -> np.ndarray:
    """
    Calculate the RTD of the network in response to a unit step in inlet concentration.
    One residence time distribution is calculated for each specie in the system.

    NOTE: This function assumes the input is a unit step input and is hardcoded into the analysis.
          A smoothed unit step can be used, with only minor impact on the result, as long as it ramps quickly enough.

    Args:
        system_results: The results from the CM simulation, in the standard format.
        c_mesh:         The CMesh used to create the compartment network.
        config_parser:  The OpeCCM ConfigParser used for the simulation and for creating the network.
        network:        The compartment network on which the results were simulated.

    Returns:
        ~: A single (num species)x(num times)x3 array containing 3 sub-items:
            [:, :, 0] is time.
            [:, :, 1] is E(t), the probability density function derived from F(t)
            [:, :, 2] is F(t), the cumulative probability function, calculated directly.
    """
    print("Start calculating RTD")
    id_inlet  = c_mesh.grouped_bcs.id(config_parser.get_item(['POST-PROCESSING', 'inlet_bc_name'], str))
    id_outlet = c_mesh.grouped_bcs.id(config_parser.get_item(['POST-PROCESSING', 'outlet_bc_name'], str))
    points_per_model = config_parser.get_item(['SIMULATION', 'points_per_pfr'], int)
    species_names    = config_parser.get_list(['SIMULATION', 'specie_names'],   str)

    y, t, inlets_map, outlets_map = system_results
    _, _, q_connections, _ = network

    # Calculate the mass flow into the reactor.
    inlet_mass_flow_total = sum(q_connections[inlet_info[1]] for inlet_info in inlets_map[id_inlet])

    # Calculate the mass flow out of the reactor at every point in time
    outlet_mass_flows = np.zeros((len(species_names), len(outlets_map[id_outlet]), len(t)))
    for specie_id, specie_name in enumerate(species_names):
        for i, outlet_info in enumerate(outlets_map[id_outlet]):
            id_model, id_connection = outlet_info
            id_end = points_per_model * (id_model + 1) - 1

            outlet_mass_flows[specie_id, i, :] = y[specie_id, id_end, :] * q_connections[id_connection]
    outlet_mass_flow_total = outlet_mass_flows.sum(1)  # Sum up each outlet DOF

    # In the last dimensions:
    # 1st entry is time
    # 2nd entry is the probability density function, E(t). This is derived from F(t).
    # 3rd entry is the cumulative density function, F(t).
    rtd = np.zeros((len(species_names), len(t), 3))
    rtd[:, :, 0] = t
    # Calculate F(t) = m_out(t) / m_in
    rtd[:, :, 2] = outlet_mass_flow_total / inlet_mass_flow_total
    # Calculate E(t) as dF/dt
    for i in range(len(species_names)):
        rtd[i, :, 1] = np.gradient(rtd[i, :, 2], rtd[i, :, 0], edge_order=2)

    print('Done calculating RTD')
    return rtd


def plot_results(system_results:    Tuple[
                                        np.ndarray,
                                        np.ndarray,
                                        Dict[int, List[Tuple[int, int]]],
                                        Dict[int, List[Tuple[int, int]]]],
                 c_mesh:            CMesh,
                 config_parser:     ConfigParser,
                 rtd:               Optional[np.ndarray] = None,) -> None:
    """

    Args:

    """
    print('Start plotting results')

    id_inlet  = c_mesh.grouped_bcs.id(config_parser.get_item(['POST-PROCESSING', 'inlet_bc_name'],  str))
    id_outlet = c_mesh.grouped_bcs.id(config_parser.get_item(['POST-PROCESSING', 'outlet_bc_name'], str))
    points_per_model = config_parser.get_item(['SIMULATION', 'points_per_pfr'], int)
    species_names    = config_parser.get_list(['SIMULATION', 'specie_names'],   str)

    # y is of shape of (n_species, n_states, n_timesteps)
    y, t, inlets_map, outlets_map = system_results

    # Plot inlet and outlet concentrations over time
    for specie_id, specie_name in enumerate(species_names):
        plt.figure()
        legend = []
        for model, _ in inlets_map[id_inlet]:
            id_start = points_per_model * model
            plt.plot(t, y[specie_id, id_start, :])
            legend.append("inlet id: {}".format(id_start))
        for model, _ in outlets_map[id_outlet]:
            id_end = points_per_model * (model + 1) - 1
            plt.plot(t, y[specie_id, id_end, :], '--')
            legend.append("outlet id: {}".format(id_end))
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.legend(legend)
        plt.title(f"System's Step Response for '{specie_name}'")
        plt.show()

        # Plot Residence Time Distribution: E(t) and F(t)
        if rtd is not None:
            plt.figure()
            plt.plot(rtd[specie_id, :, 0], rtd[specie_id, :, 1])
            plt.xlabel("Time [T]")
            plt.ylabel("Fraction of Mass per Unit Time [1/T]")
            plt.axhline(y=0, color='grey', linestyle='--')
            plt.title(f"Residence Time Distribution Function for '{specie_name}'")
            plt.show()

            plt.figure()
            plt.plot(rtd[specie_id, :, 0], rtd[specie_id, :, 2])
            plt.xlabel("Time [T]")
            plt.ylabel("Fraction of Mass [-]")
            plt.axhline(y=0, color='grey', linestyle='--')
            plt.axhline(y=1, color='grey', linestyle='--')
            plt.title(f"Cumulative Distribution Function for '{specie_name}'")
            plt.show()
    print('Done plotting results')


def visualize_model_network(model_network:  Tuple[
                                                Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
                                                np.ndarray,
                                                np.ndarray,
                                                Dict[int, List[int]]],
                            compartments:   Dict[int, Set[int]],
                            mesh:           CMesh,
                            n:              np.ndarray,
                            config_parser:  ConfigParser
                            ) -> None:
    """
    Take in a network of PFRs/CSTR and create a visual representation of them using the NetworkX package.

    This representation is really meant for 2D systems since it produces a 2D image.
    If a 3D system is provided the network will be projected on the xy-plane.

    Args:
        model_network:  The network of reactors (current PFRs/CSTRs) to visualize.
                        See compartments_to_pfrs or compartments_to_cstrs for an in-depth description.
        compartments:   The compartments that were converted into the model network.
                        See calculate_compartments for an in-depth description.
        mesh:           The OpenCCM mesh used to generate the reactor network.
        n:              Direction vector as a numpy array where the ith row represents the ith mesh element.
        config_parser:  OpenCCM ConfigParser used for parameters.
    """
    print("Visualizing model network")

    try:
        import networkx as nx
    except ModuleNotFoundError:
        print("The optional package `NetworkX` was not installed, cannot visualize the network. "
              "Install using `pip install networkx`.")
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("The optional package `Matplotlib` was not installed, cannot visualize the network. "
              "Install using `pip install matplotlib`.")
        return

    connection, _, _, compartment_to_model_map = model_network

    ####################################################################################################################
    # 1. Calculate the center for each compartment (projected onto the xy-plane)
    ####################################################################################################################
    center_of_compartments: Dict[int, np.ndarray] = {}
    for compartment, compartment_elements in compartments.items():
        pos = np.zeros(2)
        for element in compartment_elements:
            pos += np.sum(mesh.vertices[mesh.element_vertices[element], 0:2], 0) / len(mesh.element_vertices[element])
        center_of_compartments[compartment] = pos / len(compartment_elements)

    ####################################################################################################################
    # 2. Calculate the center of the network
    ####################################################################################################################
    network_centroid = np.zeros(2)
    for position_element in center_of_compartments.values():
        network_centroid += position_element
    network_centroid /= len(center_of_compartments)

    ####################################################################################################################
    # 3. Calculate position of each model in the compartment
    ####################################################################################################################
    # Calculate the average alignment of the compartment (projected onto the xy-plane)
    n_avg = np.zeros((len(compartments), 2))
    for compartment, elements_in_compartment in compartments.items():
        # Taking on the x and y components projects onto the xy-plane
        n_avg[compartment, :] = np.mean(n[list(elements_in_compartment), :], axis=0)[0:2]
        n_avg[compartment, :] /= np.linalg.norm(n_avg[compartment, :])

    positions: Dict[int, np.ndarray] = {}
    for compartment, models_in_compartment in compartment_to_model_map.items():
        if len(models_in_compartment) == 1:  # CSTR (or single PFR per compartment)
            positions[models_in_compartment[0]] = center_of_compartments[compartment]
        else:  #Multiple PFRs per compartment
            num_pfrs = len(models_in_compartment)
            n_compartment = n_avg[compartment]
            center_compartment = center_of_compartments[compartment]

            # Split the length of each compartment into n segments
            # Take average direction, point it through the center of the compartment
            # Find left and right bounds by seeing where that intersects with the surface of the compartment
            distance_min =  np.inf
            distance_max = -np.inf
            for element in compartments[compartment]:
                for facet in mesh.element_facets[element]:
                    distance_facet = n_compartment.dot(mesh.facet_centers[facet][0:2])
                    distance_min = min(distance_min, distance_facet)
                    distance_max = max(distance_max, distance_facet)

            length_compartment = distance_max - distance_min
            length_pfr = length_compartment / num_pfrs

            p = center_compartment + (distance_min - n_compartment.dot(center_compartment) + length_pfr/2) * n_compartment
            for pfr in models_in_compartment:
                positions[pfr] = p.copy()
                p += length_pfr * n_compartment

    ####################################################################################################################
    # 4. Calculate the center of each of the boundaries that need a node
    ####################################################################################################################
    facets_for_bcs: Dict[int, Set[int, ...]] = defaultdict(set)
    for facet, bc in enumerate(mesh.facet_to_bc_map):
        if bc < 0:
            facets_for_bcs[bc].add(facet)
    # Don't need to label the no-flux BC
    # Default of None so the code doesn't error if there aren't any no-flux BCs.
    facets_for_bcs.pop(mesh.grouped_bcs.no_flux, None)

    for bc, facets in facets_for_bcs.items():
        pos = np.zeros(2)
        for facet in facets:
            pos += mesh.facet_centers[facet][0:2]
        pos /= len(facets)
        # Move the position for the boundary slightly outside the boundary in order to spread them out better
        positions[bc] = pos + 0.1 * (pos - network_centroid)

    ####################################################################################################################
    # 5. Push all of the nodes slightly apart so that they don't overlap and so that you can see the arrowheads
    ####################################################################################################################
    min_distance_between_nodes = 0.1
    for id in positions:
        positions[id] *= 1 + 2*min_distance_between_nodes
    spread_out_nodes(positions, min_distance_between_nodes)

    ####################################################################################################################
    # 6. Calculate the aspect ratio for the plot by finding the binding box around the points
    ####################################################################################################################
    x_min =  np.inf
    x_max = -np.inf
    y_min =  np.inf
    y_max = -np.inf
    for point in positions.values():
        x_min = min(x_min, point[0])
        x_max = max(x_max, point[0])
        y_min = min(y_min, point[1])
        y_max = max(y_max, point[1])

    scale = 3
    width  = scale * (x_max - x_min)
    height = scale * (y_max - y_min)

    ####################################################################################################################
    # 7. Graph
    ####################################################################################################################
    graph = nx.MultiDiGraph()
    for compartment, inlets_outlets_info in connection.items():
        for neighbour_upstream in inlets_outlets_info[0].values():
            graph.add_edge(neighbour_upstream, compartment)
        for neighbour_downstream in inlets_outlets_info[1].values():
            graph.add_edge(compartment, neighbour_downstream)

    node_color = []
    node_color.extend(sorted(facets_for_bcs.keys()))
    for ignored_id in mesh.grouped_bcs.ignored:
        node_color.remove(ignored_id)
    for compartment, models_in_compartment in compartment_to_model_map.items():
        for _ in range(len(models_in_compartment)):
            node_color.append(compartment)

    plt.figure(figsize=(width, height))
    nx.draw(graph, pos=positions, with_labels=True, font_weight='bold', node_color=node_color, cmap=plt.cm.coolwarm)
    plt.savefig(config_parser.get_item(['SETUP', 'output_folder_path'], str) + 'compartment_network.pdf')

    print("Done visualizing model network")


def spread_out_nodes(positions: Dict[int, np.ndarray], node_min_distance: float) -> None:
    """
    This function tweaks to position of the passed in nodes to try to stop overlaps from happening.
    A minimum L2 distance between the nodes is enforced.

    NOTE: positions is modified in-place.

    The domain is split into a NxM grid of uniform size, the first and last row/column are semi-infinite in size
    to capture the growth of the bounding box of the nodes.
    The values for NxM are chosen such that any node within cell (i, j) is guaranteed to be at least min_distance
    away from any node in cell (i+/-2, j+/-2), thus only the 8 neighbouring cells must be considered for ensuring the
    minimum distance.
    Each node is then placed into one of the NxM grid cells.
    An optimization problem is set up where the objective function is ∑_i ||d_i||_2, where d is the displacement vector
    of a node.
    For each node (i) in this cell, a non-linear constraint is created of the form ||d_i - d_j||_2 >= min_distance
    where (j) is every other node being considered.

    The original problem would consider EVERY other cell for (j), however this problem is infeasible.
    Instead, the problem is relaxed and the position optimization problem is performed one cell at a time, beginning
    at one corner of the domain and covering the whole domain with the column index moving faster.

    Thus, (j) only need to include every node in the current cell and those in the adjacent 3 THREE 8 cell.
    Further, for each optimization problem ONLY the nodes in the current cell are allowed to move, all other nodes are
    fixed in place.
    This results in K displacement vectors, where K is the number of nodes in the current cell.

    Given the solution to the optimization problem, the positions of the nodes are adjusted, and moved to different
    cells if need be.
    Once a cell has been optimized, the nodes inside of it are fixed and will not be moved again.
    Note that if the optimization causes a node to move into a cell which has not yet been optimized, that node
    will be moved again.

    These choices have been made in order to significantly reduced the complexity of the problem.
    Attempting to do it in one go would be very computationally demanding as it would involve P unknowns and
    P^2 non-linear constraints where P is the number of nodes (# of PFRS), which can easily be O(10^2) if not O(10^3).

    Args:
        positions:          Mapping between node name (int) and 2D position.
        node_min_distance:  The minimum distance between any two nodes.

    Returns:
        ~: None. positions is modified in place.
    """
    x_min =  np.inf
    x_max = -np.inf
    y_min =  np.inf
    y_max = -np.inf
    for point in positions.values():
        x_min = min(x_min, point[0])
        x_max = max(x_max, point[0])
        y_min = min(y_min, point[1])
        y_max = max(y_max, point[1])

    l_x = x_max - x_min
    l_y = y_max - y_min
    shortest_side = min(l_x, l_y)

    num_splits_shortest = 10
    # The split size is chosen to be at least 2*node_radius so that only one layer of points need to be evaluated
    d = max(node_min_distance, shortest_side / num_splits_shortest)

    num_splits_x = int(np.ceil(l_x / d))
    num_splits_y = int(np.ceil(l_y / d))
    origin = np.array([x_min, y_min])

    # The first and last entries also capture all points outside the original bounds of the domain.
    # E.g. domain[0][0] would bound the region:
    #   x in [x_min, x_min + d)
    #   y in [y_min, y_min + d)
    # However it will actually store all points in the ranges:
    #   x in (-inf, x_min + d)
    #   y in (-inf, y_min + d)
    # Same idea for the last column/row.
    domain: List[List[Dict[int, np.ndarray]]] = [[{} for col in range(num_splits_x)] for row in range(num_splits_y)]

    def pos_to_row_col(x: np.ndarray) -> Tuple[int, int]:
        """
        Helper function to find which cell (row, col) to use for a node with a positon vector x.

        Args:
            x: The position vector to locate

        Returns:
            row, col: The row and column in the grid in which the position vector lands.
        """
        i_col_row = np.int64(np.ceil((x-origin) / d) - 1)

        # Handle the fact that the first and last split stretch out to +/-infinity
        col = min(num_splits_x-1,
                  max(0,
                      i_col_row[0]))
        row = min(num_splits_y-1,
                  max(0,
                      i_col_row[1]))

        return int(row), int(col)

    def constraint_func(delta: np.ndarray, dist_vec_0: np.array, num_movable: int, num_fixed: int) -> np.ndarray:
        """
        Calculate the L2 distance between each movable node and every fixed node as well as every other
        movable node.

        NOTE: This function captures many variables from the outside scope, they can't be passed in by argument
        since scipy will only call it with one argument.

        Args:
            delta:          2*num_movable vector of proposed displacements (in dx, dy ordering).
            dist_vec_0:     Vector representing the distances between paired nodex
                            The first num_movable*num_fixed entries are movable-to-fixed nodes distances
                            with the movable id moving slowly, i.e:
                                0:                            m_0 -> f_0
                                1:                            m_0 -> f_1
                                ...
                                num_fixed:                    m_1 -> f_0
                                num_fixed+1:                  m_1 -> f_1
                                ...
                                (num_movable*num_fixed) - 1:  m_num_movable -> f_num_fixed
                                ...
                                The subsequent num_movable*(num_movable-1)/2 are movable-to-movable node distances:
                                num_movable*num_fixed:                    m_0 -> m_1
                                num_movable*num_fixed+1:                  m_0 -> m_2
                                ...
                                num_movable*num_fixed + num_movable-1:    m_1 -> m_2
                                ...
            num_movable:    The number of nodes which can be moved.
            num_fixed:      The number of nodes whose position is fixed.

        Returns:
            ~:  Vector indicating the L2 distance between paired nodes. See documentation on `dist_vec_0`
                For the ordering of points within this vector.
        """
        delta = delta.reshape((len(delta) // 2, 2))
        dist_vec = dist_vec_0.copy()

        indx = 0
        # Movable-fixed connections
        for i in range(num_movable):
            dist_vec[indx:indx + num_fixed, :] += delta[i, :]
            indx += num_fixed

        # Movable-Movable connections
        for i in range(num_movable):
            for j in range(i + 1, num_movable):
                dist_vec[indx, :] += delta[i, :] - delta[j, :]
                indx += 1

        return np.linalg.norm(dist_vec, axis=1)

    # Load all points into the domain
    for point, pos in positions.items():
        row, col = pos_to_row_col(pos)
        domain[row][col][point] = pos

    # Move all of the points in a single pass
    for row in range(num_splits_y):
        for col in range(num_splits_x):
            nodes_movable = domain[row][col]
            num_movable = len(nodes_movable)
            if num_movable == 0:
                # Nothing needs optimizing
                continue

            # Find all of the neighbouring cells which are fixed in place
            nodes_fixed: Dict[int, np.ndarray] = {}
            if row != 0:
                nodes_fixed = {**nodes_fixed, **domain[row-1][col  ]}
            if col != 0:
                nodes_fixed = {**nodes_fixed, **domain[row  ][col-1]}
            if row != 0 and col != 0:
                nodes_fixed = {**nodes_fixed, **domain[row-1][col-1]}
            num_fixed = len(nodes_fixed)
            if num_fixed == 0 and num_movable == 1:
                # Automatically satisfied because of grid spacing
                continue

            # Array containing the initial distances between the points being compared
            dist_vec_0 = np.zeros((num_movable*num_fixed + num_movable*(num_movable-1)//2, 2))
            indx = 0
            # Calculate initial distances
            # Constraints between movable and fixed nodes num_movable*num_fixed of these
            keys_mov = list(nodes_movable.keys())
            for key in keys_mov:
                node_movable = nodes_movable[key]
                for node_fixed in nodes_fixed.values():
                    dist_vec_0[indx, :] = node_movable - node_fixed
                    indx += 1

            # Constraints between movable nodes num_movable * (num_movable-1)/2 of these
            for i, key_i in enumerate(keys_mov):
                for j in range(i+1, num_movable):
                    dist_vec_0[indx, :] = nodes_movable[key_i] - nodes_movable[keys_mov[j]]
                    indx += 1

            # Skip the rest of the optimization if the nodes are already spread apart enough
            if not np.any(np.linalg.norm(dist_vec_0, axis=1) < node_min_distance):
                continue

            # Create non-linear constraint
            optimization_constraints = optimize.NonlinearConstraint(
                fun=lambda delta: constraint_func(delta, dist_vec_0, num_movable, num_fixed),
                lb=node_min_distance,
                ub=np.inf
            )

            delta_0 = np.zeros(2 * len(nodes_movable))
            # If there are fixed nodes, find the distance between them and all movable points
            # If any fixed points are closer than the minimum to a movable point
            if num_fixed > 0:
                # distances_below_min is in the range of (-inf, node_min_distance]
                # How much further a given pair must be moved before it satisfied node_min_distance.
                # Negative values mean minimum distance is satisfied.
                distances_below_min = node_min_distance - np.linalg.norm(dist_vec_0, axis=1)[:num_movable*num_fixed]
                i_max = np.argmax(distances_below_min)
                if distances_below_min[i_max] > 0:
                    delta_0 = dist_vec_0[i_max].repeat(len(nodes_movable), axis=0).flatten()

            # Have to use this since:
            # 1. The distance constraint is either non-linear (L2-norm) or needs produces a MILP (L1-norm)
            # 2. If we use the MILP approach, the objective function would require us to bound the displacement to be
            #    non-negative.
            results = optimize.minimize(lambda delta: np.linalg.norm(delta), delta_0, method="SLSQP", constraints=optimization_constraints)

            if not results['success']:
                continue  # This is only for visualization, if a feasible solution is not found, carry on.
            else:
                deltas = results['x'].reshape((len(nodes_movable), 2))
                for i, key in enumerate(keys_mov):
                    # Update the position
                    nodes_movable[key] += deltas[i, :]

                    # Update the grid location
                    _row, _col = pos_to_row_col(nodes_movable[key])
                    domain[_row][_col][key] = domain[row][col].pop(key)
