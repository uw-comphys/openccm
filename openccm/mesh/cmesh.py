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
The intermediary mesh format, CMesh, used by all internal OpenCCM calculations so that the compartmentalization and simulation
algorithms provided in this package are CFD-package agnostic.
"""

from typing import Tuple, Dict, List

import numpy as np

from openccm.config_functions import ConfigParser


class GroupedBCs:
    """
    Helper class which handles the internal numbering of the boundary conditions for the mesh.
    """
    def __init__(self, config_parser: ConfigParser):
        """
        Only initializer for  the class.

        Parameters
        ----------
        * config_parser: The OpenCCM configuration parser which contains the
        """
        domain_inlet_names  = config_parser.get_expression(['INPUT', 'domain_inlet_names'])
        domain_outlet_names = config_parser.get_expression(['INPUT', 'domain_outlet_names'])
        ignored_names       = config_parser.get_expression(['INPUT', 'ignored_names'])
        no_flux_names       = config_parser.get_expression(['INPUT', 'no_flux_names'])

        assert (not domain_inlet_names)  or (len(domain_inlet_names)  == len(set(domain_inlet_names)))
        assert (not domain_outlet_names) or (len(domain_outlet_names) == len(set(domain_outlet_names)))
        assert (not ignored_names)       or (len(ignored_names)       == len(set(ignored_names)))
        assert (not no_flux_names)       or (len(no_flux_names)       == len(set(no_flux_names)))

        self.domain_inlet_names:  Tuple[str, ...] = tuple(sorted(domain_inlet_names))  if domain_inlet_names  else tuple()
        """The names of all boundaries which are inlets to the domain."""
        self.domain_outlet_names: Tuple[str, ...] = tuple(
            sorted(domain_outlet_names)) if domain_outlet_names else tuple()
        """The names of all boundaries which are outlets to the domain."""
        self.domain_in_out_names: Tuple[str, ...] = tuple(
            inout for inout in self.domain_inlet_names + self.domain_outlet_names)
        """The name of all boundaries which are inlets or outlets to the domain."""
        self.ignored_names: Tuple[str, ...] = ignored_names if ignored_names else tuple()
        """
        The names of all boundaries which are ignored by the algorithm.
        An example of such a boundary are the ignored boundaries used in OpenFOAM for creating a quasi-2D domain by creating a mesh
        with a single layer of cells.

        These boundaries are not used for picking seeds, but they are also **not** removed from the compartmentalization domain  
        """
        self.no_flux_names: Tuple[str, ...] = no_flux_names if no_flux_names else tuple()
        """The names of all boundaries no flow through them; they are removed from the compartmentalization."""

        for reserved_name in ['point']:
            if (reserved_name in self.no_flux_names
                    or reserved_name in self.ignored_names
                    or 'point' in self.domain_in_out_names):
                raise ValueError("'point' is a reserved keyword and cannot be used in boundary conditions names.")

        self.domain_inlets = tuple(self.id(inlet_name) for inlet_name in self.domain_inlet_names)
        """The IDs of all inlet boundaries, in the same order as `domain_inlet_names`."""
        self.domain_outlets = tuple(self.id(outlet_name) for outlet_name in self.domain_outlet_names)
        """ The IDs of all outlet boundaries, in the same order as `domain_outlet_names`."""
        self.ignored = tuple(self.id(ignored_name) for ignored_name in self.ignored_names)
        """The IDs of all ignored boundaries, in the same order as `ignored_names`."""
        self.no_flux = -(1 + len(self.domain_in_out_names) + len(self.ignored_names))
        """The ID to which all no-flux boundaries are mapped to."""

        self.num_bcs = len(self.no_flux_names) + len(self.domain_in_out_names) + len(self.ignored_names)
        """The total number of named boundaries."""

    def id(self, label: str) -> int:
        """
        Helper function to map between the label (name) of a BC and it's integer ID.

        The negative values for the IDs are obtained as follows:
        - Domain inlet/outlets are labeled using their positional value (i.e. 1st, 2nd, etc.) in the `domain_in_out` tuple.
        - Ignored bcs are labeled as `len(domain_in_out)` + the positional value in `self.ignored_names` tuple (i.e. 1st, 2nd, etc.)
        - No flux BCs are all labeled as `len(domain_in_out) + len(self.ignored_names) + 1`

        Parameters
        ----------
        * label: Name of the boundary to convert into an ID.

        Returns
        ----------
        * id: Integer ID for the passed in boundary label
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
    """
    The CMesh class for storing the CFD-mesh in a source-agnostic format.
    """

    def __init__(self,
                 vertices:              np.ndarray,
                 facet_elements:        Tuple[Tuple[int, ...], ...],
                 facet_vertices:        Tuple[Tuple[int, ...], ...],
                 facet_connectivity:    Tuple[Tuple[int, ...], ...],
                 element_facets:        Tuple[Tuple[int, ...], ...],
                 element_vertices:      Tuple[Tuple[int, ...], ...],
                 element_connectivity:  Tuple[Tuple[int, ...], ...],
                 element_sizes:         np.ndarray,
                 grouped_bcs:           GroupedBCs,
                 facet_to_bc_map:       np.ndarray,
                 bc_to_facet_map:       Dict[str, Tuple[int, ...]],
                 phase_frac:            np.ndarray):
        """
        Parameters
        ----------
        * vertices:               N*M numpy array with each row is for one of the N vertices.
                                  Each column is the coordinate along the Mth dimension.
        * facet_elements:         Tuple of element IDs that share a facet.
        * facet_vertices:         Tuple of vertex IDs that make up a facet.
        * facet_connectivity:     Tuple of tuples containing the neighbours (sharing a vertex) of all facets in the mesh
        * element_facets:         Tuple of facet IDs that bound an element.
        * element_vertices:       Tuple of vertex IDs that bound and element.
        * element_connectivity:   Dictionary indexed by element number which returns a list of element number representing
                                  the neighbours of the key element
                                  - **NOTE:**   This value WILL be modified by the compartmentalization scheme. Elements will be
                                                Removed from here if they're deemed ineligible for compartmentalization, e.g.
                                                because they have a magnitude of 0.
        * phase_frac:             Phase fraction of the phase being compartmentalized, indexed by element ID. None if single phase.
        """

        # All of these must be calculated in different ways based on the simulation package they come from.
        # Some may not seem so (e.g. element connectivity), but it makes more sense to leave them
        # out of CMesh in order to be able to support fixing of hanging nodes.
        self.vertices               = vertices
        """The coordinates of each vertex, indexed by its ID."""

        self.facet_elements         = facet_elements
        """
        The elements which share this facet, indexed by its ID.
        Since the mesh cannot have hanging nodes, there will only ever be 1 or 2 elements.
        """

        self.facet_vertices         = facet_vertices
        """The vertices that make up each facet, indexed by its ID."""

        self.facet_connectivity     = facet_connectivity
        """The neighbours of each facet, indexed by its ID. Two facets are neighbours if they share a vertex."""

        self.element_facets         = element_facets
        """The facets that make up each element, indexed by its ID."""

        self.element_vertices       = element_vertices
        """The vertices that make up each element, indexed by its ID."""

        self.element_connectivity   = element_connectivity
        """The neighbours of each element, indexed by its ID. Two elements are neighbours if they share a facet."""

        self.element_sizes          = element_sizes
        """The size of each element, indexed by its ID. Represents area for 2D meshes and volume for 3D meshes."""
        self.element_sizes          *= phase_frac
        self.element_sizes.flags.writeable = False

        self.grouped_bcs            = grouped_bcs
        """The GroupedBCs object specified by the OpenCCM ConfigParser."""

        self.facet_to_bc_map        = facet_to_bc_map
        """Mapping between a facet ID and the ID of boundary on which it's on."""
        self.facet_to_bc_map.flags.writeable = False

        self.bc_to_facet_map        = bc_to_facet_map
        """Mapping between a the name of a boundary and facets which make it up."""

        # These properties can be calculated all the same regardless of the source of the mesh.
        self.facet_centers          = self._calculate_facet_center()
        """The position of each facet's center, indexed by its ID."""
        self.facet_centers.flags.writeable = False

        self.facet_normals          = self._calculate_facet_normal()
        """
        The unit normal to each facet, indexed by its ID. 
        The normal is pointing out of `self.facet_elements[facet][0]`.
        """
        self.facet_normals.flags.writeable = False

        self.facet_size             = self._calculate_facet_sizes()
        """The size of each facet, indexed by its ID. Represents length for 2D meshes and area for 3D meshes."""
        self.facet_size.flags.writeable = False
        self.element_centroids      = self._calculate_element_centroids()
        self.element_centroids.flags.writeable = False

    def _calculate_element_centroids(self) -> np.ndarray:
        """
        Returns:
            centroids: A numpy array representing the centroid of each element.
        """
        n_elements = len(self.element_vertices)
        n_dim      = len(self.vertices[0])

        centroids = np.zeros((n_elements, n_dim))
        for element, vertices in enumerate(self.element_vertices):
            for vertex in vertices:
                centroids[element, :] += self.vertices[vertex]
            centroids[element, :] /= len(vertices)

        return centroids

    def _calculate_facet_center(self) -> np.ndarray:
        """
        Calculate the center of each facet as the average position of each of its vertices.

        This method is only used in CMesh's initializer.

        Returns
        -------
        * centers: A numpy array, indexed by facet ID, containing the position, (X, Y) for 2D meshes and (x,y,z) for 3D,
                  of each facet's center.
        """
        vertices            = self.vertices
        facet_vertices_all  = self.facet_vertices

        centers = np.zeros((len(facet_vertices_all), vertices.shape[1]))

        for facet, facet_vertices in enumerate(facet_vertices_all):
            centers[facet] = sum(vertices[vertex] for vertex in facet_vertices) / len(facet_vertices)

        return centers

    def _calculate_facet_normal(self) -> np.ndarray:
        """
        Calculate the unit normal to each facet.

        Since the program requires no hanging nodes, each facet will be shared by *at most* 2 elements.
        The unit normal is calculated such for each `facet` it points out of the first element shared by this facet,
        as stored in `self.facet_elements[facet]`.

        This method is only used in CMesh's initializer.

        Returns
        -------
        * normals: Unit formal for each facet.
        """
        element_vertices    = self.element_vertices
        facet_vertices_all  = self.facet_vertices
        facet_elements      = self.facet_elements
        vertices            = self.vertices

        mesh_dim = vertices.shape[1]
        normals = np.zeros((len(facet_vertices_all), mesh_dim))

        for facet, facet_vertices in enumerate(facet_vertices_all):
            p1 = vertices[facet_vertices[0]]
            v1 = vertices[facet_vertices[1]] - p1

            if mesh_dim == 2:
                normals[facet, :] = [v1[1], -v1[0]]
            elif mesh_dim == 3:
                v2 = vertices[facet_vertices[2]] - p1
                normals[facet, :] = [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]
            else:
                raise ValueError("Unsupported mesh dimension of {}".format(mesh_dim))

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

        normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

        assert not np.any(np.isnan(normals))
        return normals

    def _calculate_facet_sizes(self) -> np.ndarray:
        """
        Calculate the size of each facet in the mesh. Size is either length, for 2D meshes, or area, for 3D meshes.

        This method is only used in CMesh's initializer.

        Returns
        -------
        * sizes: The length (2D) or area (3D) of each facet.
        """
        vertices            = self.vertices
        facet_vertices_all  = self.facet_vertices

        mesh_dim = vertices.shape[1]
        sizes = np.zeros(len(self.facet_vertices))
        _tmp_sizes = np.zeros((len(self.facet_vertices), 3))

        if mesh_dim == 2:
            for id_facet, facet_vertices in enumerate(facet_vertices_all):
                p0 = vertices[facet_vertices[0]]
                p1 = vertices[facet_vertices[1]]
                sizes[id_facet] = np.linalg.norm(p1 - p0)
        elif mesh_dim == 3:
            # Iterate over each facet and its corresponding vertex indices
            for id_facet, facet_vertices in enumerate(facet_vertices_all):
                # Iterate over the vertices of the facet
                for i in range(1, len(facet_vertices) - 1):
                    # Calculate the size of the facet
                    v1 = vertices[facet_vertices[i], :] - vertices[facet_vertices[0], :]
                    v2 = vertices[facet_vertices[i + 1], :] - vertices[facet_vertices[0], :]
                    _tmp_sizes[id_facet, 0] += v1[1]*v2[2] - v1[2]*v2[1]
                    _tmp_sizes[id_facet, 1] += v1[2]*v2[0] - v1[0]*v2[2]
                    _tmp_sizes[id_facet, 2] += v1[0]*v2[1] - v1[1]*v2[0]
        else:
            raise ValueError("Unsupported mesh of dimension {}".format(mesh_dim))
        sizes += np.linalg.norm(_tmp_sizes, axis=1) / 2
        return sizes

    def get_outward_facing_normal(self, facet: int, element: int) -> np.ndarray:
        """
        Get the unit normal to `facet` pointing out of the `element`.

        Parameters
        ----------
        * facet:    The facet whose unit normal to get.
        * element:  The element out of which the unit should point out of.

        Returns
        -------
        * normal: The unit formal to `facet` pointing out of `element`.
        """
        if element == self.facet_elements[facet][0]:
            return self.facet_normals[facet]
        else:
            return -self.facet_normals[facet]
