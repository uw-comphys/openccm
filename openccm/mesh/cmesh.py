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

from typing import Tuple, Dict, List

import numpy as np

from openccm.config_functions import ConfigParser


class GroupedBCs:
    """
    Helper class which handles the internal numbering of the BCs for the mesh.
    """
    def __init__(self, config_parser: ConfigParser):
        no_flux_names       = config_parser.get_expression(['INPUT', 'no_flux_names'])
        ignored_names       = config_parser.get_expression(['INPUT', 'ignored_names'])
        domain_inlet_names  = config_parser.get_expression(['INPUT', 'domain_inlet_names'])
        domain_outlet_names = config_parser.get_expression(['INPUT', 'domain_outlet_names'])

        assert (not no_flux_names)       or len(no_flux_names)       == len(set(no_flux_names))
        assert (not ignored_names)       or len(ignored_names)       == len(set(ignored_names))
        assert (not domain_inlet_names)  or len(domain_inlet_names)  == len(set(domain_inlet_names))
        assert (not domain_outlet_names) or len(domain_outlet_names) == len(set(domain_outlet_names))

        self.no_flux_names:       Tuple[str, ...] = no_flux_names                      if no_flux_names       else tuple()
        self.ignored_names:       Tuple[str, ...] = ignored_names                      if ignored_names       else tuple()
        self.domain_inlet_names:  Tuple[str, ...] = tuple(sorted(domain_inlet_names))  if domain_inlet_names  else tuple()
        self.domain_outlet_names: Tuple[str, ...] = tuple(sorted(domain_outlet_names)) if domain_outlet_names else tuple()
        self.domain_in_out_names: Tuple[str, ...] = tuple(inout for inout in self.domain_inlet_names + self.domain_outlet_names)

        self.domain_inlets  = tuple(self.id(inlet_name)   for inlet_name   in self.domain_inlet_names)
        self.domain_outlets = tuple(self.id(outlet_name)  for outlet_name  in self.domain_outlet_names)
        # Ignored BCs may or may not have flux through them, but they represent BCs which we neither want for seeds
        # nor want to remove from the compartmentalization.
        # These can also include periodic BCs that are used to create a 2D domain,
        # but they will NOT be properly handled as periodic.
        self.ignored        = tuple(self.id(ignored_name) for ignored_name in self.ignored_names)
        self.no_flux        = -(1 + len(self.domain_in_out_names) + len(self.ignored_names))

        self.num_bcs = len(self.no_flux_names) + len(self.domain_in_out_names) + len(self.ignored_names)

    def id(self, label: str) -> int:
        """
        Helper function to map between the label (name) of a BC and it's ID.

        The negative values for the IDs are obtained as follows:
        - Domain inlet/outlets are labeled using their positional value (i.e. 1st, 2nd, etc.) in the domain_in_out tuple.
        - Ignored bcs are labeled as len(domain_in_out) + the positional value in self.ignored_names tuple (i.e. 1st, 2nd, etc.)
        - No flux BCs are all labeled as len(domain_in_out) + len(self.ignored_names) + 1

        Args:
            label: Name of the boundary to convert into an ID.

        Returns:
            id: ID for the passed in boundary label
        """
        if label in self.no_flux_names:
            return self.no_flux
        elif label in self.domain_in_out_names:
            return -(1 + self.domain_in_out_names.index(label))
        elif label in self.ignored_names:
            return -(1 + len(self.domain_in_out_names) + self.ignored_names.index(label))
        else:
            raise ValueError(f"The passed label ({label}) is not one of the specified bcs: {self.no_flux_names + self.domain_in_out_names + self.ignored_names}")


class CMesh:

    def __init__(self,
                 vertices: np.ndarray,
                 facet_elements: Tuple[Tuple[int, ...], ...],
                 facet_vertices: Tuple[Tuple[int, ...], ...],
                 facet_connectivity: Tuple[Tuple[int, ...], ...],
                 element_facets: Tuple[Tuple[int, ...], ...],
                 element_vertices: Tuple[Tuple[int, ...], ...],
                 element_connectivity: Dict[int, List[int]],
                 element_sizes: np.ndarray,
                 grouped_bcs: GroupedBCs,
                 facet_to_bc_map: np.ndarray,
                 bc_to_facet_map: Dict[str, Tuple[int, ...]]):

        # All of these must be calculated in different ways based on the simulation package they come from.
        # Some may not seem so (e.g. element connectivity), but it makes more sense to leave them
        # out of CMesh in order to be able to support fixing of hanging nodes.
        self.vertices               = vertices
        self.facet_elements         = facet_elements
        self.facet_vertices         = facet_vertices
        self.facet_connectivity     = facet_connectivity
        self.element_facets         = element_facets
        self.element_vertices       = element_vertices
        self.element_connectivity   = element_connectivity
        self.element_sizes          = element_sizes
        self.grouped_bcs            = grouped_bcs
        self.facet_to_bc_map        = facet_to_bc_map
        self.bc_to_facet_map        = bc_to_facet_map

        # These properties can be calculated all the same regardless of the source of the mesh.
        self.facet_centers          = self._calculate_facet_center()
        self.facet_normals          = self._calculate_facet_normal()
        self.facet_size             = self._calculate_facet_sizes()

    def _calculate_facet_center(self) -> np.ndarray:
        """
        Returns:
            centers: A numpy array representing the centers of each facet.
        """
        vertices            = self.vertices
        facet_vertices_all  = self.facet_vertices

        centers = np.zeros((len(facet_vertices_all), vertices.shape[1]))

        for facet, facet_vertices in enumerate(facet_vertices_all):
            centers[facet] = sum(vertices[vertex] for vertex in facet_vertices) / len(facet_vertices)

        return centers

    def _calculate_facet_normal(self) -> np.ndarray:
        """
        Each facet will be shared by *at most* 2 elements.
        The normal calculated by this function is pointing out of the first element and into the second.

        Returns:
            normals: Unit formal for each facet.
        """
        element_vertices    = self.element_vertices
        facet_vertices_all  = self.facet_vertices
        facet_elements      = self.facet_elements
        vertices            = self.vertices

        mesh_dim = vertices.shape[1]
        normals = np.zeros((len(facet_vertices_all), mesh_dim))

        for facet, facet_vertices in enumerate(facet_vertices_all):
            p1 = vertices[facet_vertices[0]]
            vector_facet_1 = vertices[facet_vertices[1]] - p1

            if mesh_dim == 2:
                normals[facet, :] = [vector_facet_1[1], -vector_facet_1[0]]
            elif mesh_dim == 3:
                normals[facet, :] = np.cross(vector_facet_1, vertices[facet_vertices[2]] - p1)
            else:
                raise ValueError("Unsupported mesh dimension of {}".format(mesh_dim))
            normals[facet, :] /= np.linalg.norm(normals[facet, :])

            other_points = set(element_vertices[facet_elements[facet][0]])
            other_points.difference_update(facet_vertices)

            # Figure out when direction is facing outwards by looking at the dot product between it and the vector between
            # one of this facet's points and all other facts in this cell
            for other_point in other_points:
                dot_value = np.dot(normals[facet, :], vertices[other_point] - p1)

                if dot_value > 0:  # The normal is facing outwards
                    normals[facet, :] *= -1
                    break
                elif dot_value < 0:  # The normal is facing inwards
                    break
                else:  # The normal is perpendicular to the edge we're testing with
                    pass

        return normals

    def _calculate_facet_sizes(self) -> np.ndarray:
        """
        Returns:
            sizes: The length (2D) or area (3D) of each facet.
        """
        vertices            = self.vertices
        facet_vertices_all  = self.facet_vertices

        mesh_dim = vertices.shape[1]
        sizes = np.zeros(len(self.facet_vertices))

        if mesh_dim == 2:
            for id_facet, facet_vertices in enumerate(facet_vertices_all):
                p0 = vertices[facet_vertices[0]]
                p1 = vertices[facet_vertices[1]]
                sizes[id_facet] = np.linalg.norm(p1 - p0)
        elif mesh_dim == 3:
            size = np.zeros(vertices[0].shape)
            # Iterate over each facet and its corresponding vertex indices
            for id_facet, facet_vertices in enumerate(facet_vertices_all):
                size[:] = 0

                # Iterate over the vertices of the facet
                for i in range(1, len(facet_vertices) - 1):
                    # Calculate the size of the facet
                    size += np.cross(vertices[facet_vertices[i], :] - vertices[facet_vertices[0], :],
                                     vertices[facet_vertices[i + 1], :] - vertices[facet_vertices[0], :])
                sizes[id_facet] = np.linalg.norm(size) / 2
        else:
            raise ValueError("Unsupported mesh of dimension {}".format(mesh_dim))

        return sizes

    def get_outward_facing_normal(self, facet: int, element: int):
        if element == self.facet_elements[facet][0]:
            return self.facet_normals[facet]
        else:
            return -self.facet_normals[facet]
