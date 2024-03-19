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

from typing import Dict, List, Tuple, Callable

import numpy as np

from ngsolve import CoefficientFunction, FacetFESpace, GridFunction, Mesh
from ngsolve.comp import MeshNode, Ngs_Element, Integrate

from .cmesh import CMesh, GroupedBCs
from ..config_functions import ConfigParser


def convert_mesh_ngsolve(config_parser: ConfigParser, mesh: Mesh) -> CMesh:
    """
    Convert the NGSolve mesh into OpenCCM's internal format.

    Args:
        config_parser:  The OpenCCM ConfigParser
        mesh:           The NGSolve mesh to re-represent in a common format

    Returns:
        vertices:               N*M numpy array with each row is for one of the N vertices.
                                Each column is the coordinate along the Mth dimension.
        facet_elements:         Tuple of element IDs that share a facet.
        facet_vertices:         Tuple of vertex IDs that make up a facet.
        facet_connectivity:     Tuple of tuples containing the neighbours (sharing a vertex) of all facets in the mesh
        facet_sizes:            Size (area or length depending on mesh dimension) of each facet, indexed by its ID.
        facet_normals:          The unit normal to the facet pointing OUT of the first element in facet_elements[facet]
        facet_center:           NxM numpy array containing the center of each facet.
                                Each represents of the N facets (by id), each column represents each of the M dimensions.
        element_facets:         Tuple of facet IDs that bound an element.
        element_vertices:       Tuple of vertex IDs that bound and element.
        element_connectivity:   Dictionary indexed by element number which returns a list of element number representing
                                the neighbours of the key element.
                                NOTE: This value WILL be modified by the compartmentalization scheme. Elements will be
                                      Removed from here if they're deemed ineligible for compartmentalization, e.g.
                                      because they have a magnitude of 0.
    """
    print("Converting Mesh")

    _all_elements = tuple(mesh.Elements())
    vertices                                        = np.array([np.array(vertex.point) for vertex in mesh.vertices])
    facet_vertices: Tuple[Tuple[int, ...], ...]     = tuple(tuple(vertex.nr for vertex in facet.vertices) for facet in mesh.facets)
    facet_elements: Tuple[Tuple[int, ...], ...]     = tuple(tuple(sorted(element.nr for element in facet.elements)) for facet in mesh.facets)
    element_facets: Tuple[Tuple[int, ...], ...]     = tuple(tuple(facet.nr for facet in element.facets) for element in _all_elements)
    element_vertices: Tuple[Tuple[int, ...], ...]   = tuple(tuple(vertex.nr for vertex in element.vertices) for element in _all_elements)

    element_sizes               = np.array(Integrate(CoefficientFunction(1), mesh, element_wise=True))
    facet_connectivity          = _create_facet_connectivity(mesh)
    element_connectivity        = _create_element_connectivity(mesh, _all_elements)

    for element, _element_facets in enumerate(element_facets):
        for facet in _element_facets:
            assert element in facet_elements[facet]

    grouped_bcs = GroupedBCs(config_parser)

    facet_to_bc_map, bc_to_facet_map = _create_bc_mappings(mesh, grouped_bcs)

    cmesh = CMesh(vertices,
                 facet_elements, facet_vertices, facet_connectivity,
                 element_facets, element_vertices, element_connectivity, element_sizes,
                 grouped_bcs, facet_to_bc_map, bc_to_facet_map)

    print("Done converting Mesh")
    return cmesh


def _create_bc_mappings(mesh: Mesh, grouped_bcs: GroupedBCs) -> Tuple[np.ndarray, Dict[str, Tuple[int, ...]]]:
    """
    Create a dictionary for mapping mesh facets to the BC that they represent.

    All BCs which represent a no-flux BC, either explicitly in terms of the flux or with a Dirichlet condition, are
    grouped together into the same BC since they will not be treated as an inlet/outlet.

    Args:
        mesh:           The NGSolve mesh containing the compartments.
        grouped_bcs:    Helper class which handles the internal numbering of the BCs for the CMesh.

    Returns:
        facet_to_bc_map: A dictionary mapping each facet to a BC index.
                            * 0 for a non-bc facet
                            * grouped_bcs.id(bc_name) for everything else.
        bc_to_facet_map: A dictionary mapping each BC NAME to the facets that make it up.
    """
    assert len(set(mesh.GetBoundaries())) == grouped_bcs.num_bcs

    def label_facets_using_func(labeling_func: Callable) -> np.ndarray:
        cfs_4_bcs: List[CoefficientFunction] = []
        ordered_unique_bc_names: List[str] = []

        for bc_name in mesh.GetBoundaries():
            if bc_name not in ordered_unique_bc_names:
                ordered_unique_bc_names.append(bc_name)
            cfs_4_bcs.append(CoefficientFunction(labeling_func(bc_name)))

        marker_gfu.Set(CoefficientFunction(cfs_4_bcs), definedon=mesh.Boundaries('|'.join(ordered_unique_bc_names)))
        return np.array(np.round(marker_gfu.vec), dtype=int)

    fes         = FacetFESpace(mesh, order=0)
    marker_gfu  = GridFunction(fes)

    bc_name_to_facet_map: Dict[str, Tuple[int, ...]] = {}
    bc_name_to_index = {bc_name: index+1 for index, bc_name in enumerate(mesh.GetBoundaries())}
    facet_labels = label_facets_using_func(bc_name_to_index.get)
    for bc_name, i in bc_name_to_index.items():
        bc_name_to_facet_map[bc_name] = tuple(np.where(facet_labels == i)[0])

    facet_to_bc_id_map = label_facets_using_func(grouped_bcs.id)

    return facet_to_bc_id_map, bc_name_to_facet_map


def _create_element_connectivity(mesh: Mesh, all_elements: Tuple[Ngs_Element]) -> Dict[int, List[int]]:
    """
    Returns:
        neighbours_all: A tuple of the IDs of all neighbouring elements of a given element ID, indexed by that ID.
    """
    neighbours_all: Dict[int, List[int]] = dict()

    for element in all_elements:
        # Using a set, rather than a list, to make handling of duplicates trivial
        neighbours_of_element = set()

        # Iterate over all edges of the face, and add all faces connected to that edges to the list of neighbours
        for facet_id in element.facets:
            neighbours_of_element.update(e_neighbour.nr for e_neighbour in mesh.facets[facet_id.nr].elements)

        # Make sure that an element isn't listed as its own neighbour
        neighbours_of_element.remove(element.nr)

        neighbours_all[element.nr] = sorted(neighbours_of_element)

    return neighbours_all


def _create_facet_connectivity(mesh: Mesh) -> Tuple[Tuple[int, ...], ...]:
    """
    Returns:
        neighbours_all: A tuple of the IDs of all neighbouring facets of a given facet ID, indexed by that ID.
    """
    neighbours_all: Dict[int, Tuple[int, ...]] = dict()

    if mesh.dim == 2:  # Facets are edges in 2D
        edges: Tuple[MeshNode] = tuple(mesh.edges)
        vertex_edges = list(set(edge.nr for edge in vertex.edges) for vertex in mesh.vertices)
        for edge in edges:
            neighbours = set()
            for vertex in edge.vertices:
                neighbours.update(vertex_edges[vertex.nr])
            neighbours.remove(edge.nr)
            neighbours_all[edge.nr] = tuple(sorted(neighbours))
    elif mesh.dim == 3:  # Facets are faces in 3D
        edges = tuple(mesh.edges)
        faces = tuple(mesh.faces)
        for face in faces:
            neighbours = set()

            for edge in face.edges:
                neighbours.update(face.nr for face in edges[edge.nr].faces)

            neighbours.remove(face.nr)
            neighbours_all[face.nr] = tuple(sorted(neighbours))
    else:
        raise ValueError("Unexpected mesh of dimension {}".format(mesh.dim))
    return tuple(neighbours_all[id_facet] for id_facet in neighbours_all)
