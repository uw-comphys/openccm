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

from collections import defaultdict
from typing import Tuple, Dict, List, Set, Optional
from itertools import combinations
from .cmesh import CMesh, GroupedBCs
from ..config_functions import ConfigParser
from ..io import read_boundary_condition, read_mesh_data


def convert_mesh_openfoam(config_parser: ConfigParser, phase_frac: np.ndarray) -> CMesh:
    """
    Read OpenFOAM mesh information from file and convert it into OpenCCM's internal CMesh format.

    Parameters
    ----------
    * config_parser:    OpenCCM ConfigParser from which to get the required OpenFOAM data.
    * phase_frac:       Fraction of each mesh element taken up by the phase we wish to compartmentalize.

    Returns
    -------
    * cmesh: The internal CMesh representation of the OpenFOAM mesh.
    """
    print("Converting Mesh")
    # Note: These file paths are NOT manually specified by the user. They are automatically generated

    # The owner contains one line for each facet, and on that line is the ID of the element (cell) that owns that facet
    owner               = read_mesh_data(config_parser.get_item(['INPUT', 'owner_file_path'],       str), int)
    # The neighbour contains one line per facet, **but only for internal facets**.
    # Each facet in this file is shared by two elements (the owner and the neighbour)
    neighbour           = read_mesh_data(config_parser.get_item(['INPUT', 'neighbour_file_path'],   str), int)
    # Each line contains the vertex ID for the vertices that make up the facet
    facet_information   = read_mesh_data(config_parser.get_item(['INPUT', 'face_file_path'],        str), int)
    # File contains one value per line, implicitly indexed by element ID, containing the size (volume) of that element
    element_sizes       = read_mesh_data(config_parser.get_item(['INPUT', 'volume_file_path'],      str), float)
    # File contains a tuple of form (x y z) per line, indexed by vertex ID, specifying the 3D position of each vertex
    vertices            = read_mesh_data(config_parser.get_item(['INPUT', 'point_file_path'],       str), float)
    # This file specifies the names, types, and facets that make up each boundary condition.
    bc_names            = read_boundary_condition(config_parser.get_item(['INPUT', 'boundary_file_path'], str))

    facet_vertices                  = tuple(tuple(map(int, facet)) for facet in facet_information)
    facet_elements, element_facets  = _get_facet_element_info(owner, neighbour)
    element_vertices                = _get_element_vertices(element_facets, facet_vertices)

    facet_connectivity      = _create_facet_connectivity(facet_vertices)
    element_connectivity    = _create_element_connectivity(neighbour, owner)

    grouped_bcs = GroupedBCs(config_parser)

    facet_to_bc_map, bc_to_facet_map = _create_bc_mappings(len(facet_vertices), grouped_bcs, bc_names)

    cmesh = CMesh(vertices,
                 facet_elements, facet_vertices, facet_connectivity,
                 element_facets, element_vertices, element_connectivity, element_sizes,
                 grouped_bcs, facet_to_bc_map, bc_to_facet_map,
                 phase_frac)

    print("Done converting Mesh")
    return cmesh


def _create_bc_mappings(number_facets: int, grouped_bcs: GroupedBCs, bc_names: Dict[str, Tuple[int, int]])\
        -> Tuple[
            np.ndarray,
            Dict[str, Tuple[int, ...]]
        ]:
    """
    Create the facet_to_bc_map for looking up which BC ID (NOT NAME) belongs to which facet,
    and the bc_to_facet_map for looking up which facets belong to which BC NAME.

    bc_to_facet_map is currently only needed for outputting data back into OpenFOAM format.
    bc_to_facet_map will store all values, not just the start and number of facets incase different mesh formats.

    Parameters
    ----------
    * number_facets:    The number of facets that make up the mesh.
    * bc_names:         Dictionary where keys are the name of boundary condition and values are (startFace, nFace).

    Returns
    -------
    * facet_to_bc_map:  A dictionary mapping each facet to a BC index.
                        * 0 for a non-bc facet
                        * grouped_bcs.id(bc_name) for everything else.
    * bc_to_facet_map:  A dictionary mapping each BC NAME to the facets that make it up.
    """
    assert len(set(bc_names)) == grouped_bcs.num_bcs

    facet_to_bc_map = np.zeros(number_facets, dtype=np.min_scalar_type(-number_facets))
    bc_to_facet_map = {}
    for bc_name, start_and_num in bc_names.items():
        facet_to_bc_map[start_and_num[0]:(start_and_num[0] + start_and_num[1])] = grouped_bcs.id(bc_name)

        bc_to_facet_map[bc_name] = tuple(range(start_and_num[0], (start_and_num[0] + start_and_num[1])))

    return facet_to_bc_map, bc_to_facet_map


def _get_facet_element_info(owner: np.ndarray, neighbour: np.ndarray) -> Tuple[
                Tuple[Tuple[int, ...], ...],
                Tuple[Tuple[int, ...], ...]]:
    """
    Calculate the facets bounding each element and the elements which share each facet.

    Parameters
    ----------
    * owner:        Array indexed by facet ID containing the ID of the owner element.
    * neighbour:    Array indexed by facet ID containing the ID of the neighbouring element.
                    Guaranteed to be at most the size of owner since entries in this array do not include domain boundary facets.
                    Only facets shared by two facets

    Returns
    -------
    * facet_elements:  The elements which share a facet, indexed by its ID.
    * element_facets:  The facets that bound an element, indexed by its ID.
    """
    assert len(owner) >= len(neighbour)

    element_facets = defaultdict(set)
    facet_elements = []

    for i in range(len(neighbour)):
        owner_i: int     = owner[i]
        neighbour_i: int = neighbour[i]

        facet_elements.append((owner_i, neighbour_i))
        element_facets[owner_i].add(i)
        element_facets[neighbour_i].add(i)

    for j in range(len(neighbour), len(owner)):
        owner_j = owner[j]

        facet_elements.append((owner_j,))
        element_facets[owner_j].add(j)

    element_facets_tuple = tuple(tuple(element_facets[element]) for element in sorted(element_facets.keys()))

    assert len(element_facets_tuple) == max(element_facets.keys()) + 1  # +1 since counting from 0

    return tuple(facet_elements), element_facets_tuple


def _create_element_connectivity(neighbour: np.ndarray, owner: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    """
    Calculate the connectivity of mesh's elements.
    Two elements are considered connected, i.e. neighbours, if they share a facet.

    OpenFOAM stores its connectivity using two entries: `owner` and `neighbour`.
    Each facet has exactly one owner element, and will have a neighbour element unless the facet is on the domain boundary.
    Because of this, the length of the neighbour array is the number of facets which have more than one element, and
    represent the elements which have a neighbour.
    This function uses those two arrays to create an explicit connectivity of the elements.

    Parameters
    ----------
    * owner:        The owner element of each facet, indexed by facet ID. Size of N
    * neighbour:    The neighbour element of each facet, indexed by facet ID. Size M.

    Returns
    -------
    * element_connectivity: The neighbours of each element, indexed by its ID.
    """
    element_connectivity: Dict[int, Set[int]] = defaultdict(set)

    for i in range(len(neighbour)):
        element_connectivity[    owner[i]].add(neighbour[i])
        element_connectivity[neighbour[i]].add(owner[i])

    return tuple(tuple(sorted(element_connectivity[element])) for element in sorted(element_connectivity.keys()))


def _get_element_vertices(element_facets: Tuple[Tuple[int, ...], ...], facet_vertices: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    """
    Retrieve the vertex IDs that make up each element based on the element facet information and facet vertices.

    Parameters
    ----------
    * element_facets: The facets making up each element, indexed by its ID.
    * facet_vertices: The vertices making up each facet, indexed by its ID.

    Returns
    -------
    * element_vertices: The vertex IDs making up each element, indexed by its ID.
    """
    element_vertices = []

    for facet in element_facets:
        facet_vertices_all = []
        for vertex_id in facet:
            facet_vertices_all.extend([int(vertex) for vertex in facet_vertices[vertex_id]])
        element_vertices.append(facet_vertices_all)
    element_vertices = tuple(tuple(set(map(int, vertices_all))) for vertices_all in element_vertices)
    return element_vertices


def _create_facet_connectivity(facet_vertices_all: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    """
    Calculate the connectivity of mesh's facets.
    Two facets are considered connected, i.e. neighbours, if they share a vertex.

    Parameters
    ----------
    * facet_vertices_all: All neighbours A tuple of tuples representing the vertices of each facet.

    Returns
    -------
    * facet_connectivity: The neighbouring facets of each facet, indexed by its ID.
    """
    # Create a vertex -> facets lookup
    vertex_facets = defaultdict(set)
    for facet_id, vertices in enumerate(facet_vertices_all):
        for vertex in vertices:
            vertex_facets[vertex].add(facet_id)

    facet_connectivity = []
    for facet_id, vertices in enumerate(facet_vertices_all):
        # Get the facet IDs associated with the current vertex and remove the current facet ID from the set
        possible_connection = [vertex_facets[vertex_id] - {facet_id} for vertex_id in vertices]

        # Perform the intersection of facet ID sets for all combinations of possible connections
        intersected_elements = set()
        for combination in combinations(possible_connection, 2):
            intersected_elements.update(set.intersection(*combination))

        facet_connectivity.append(tuple(sorted(intersected_elements)))

    return tuple(facet_connectivity)
