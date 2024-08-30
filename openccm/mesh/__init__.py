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
The mesh module contains the intermediate mesh representation, CMesh, and the functions required to convert OpenFOAM
and NGSolve meshes to CMesh format.
"""

from typing import Optional

import numpy as np

from .cmesh import CMesh, GroupedBCs

from .convert_openfoam import convert_mesh_openfoam
from ..config_functions import ConfigParser


def convert_mesh(config_parser: ConfigParser, ngsolve_mesh: Optional['ngsolve.Mesh']) -> CMesh:
    """


    Parameters
    ----------
    * config_parser:    The OpenCCM ConfigParser from which to get the required info for conversion.
    * ngsolve_mesh:     The NGSolve mesh object to convert if using OpenCMP.

    Returns
    -------
    * cmesh: Internal representation of the mesh inside ngsolve_mesh or pointed to by config_parser
    """
    if ngsolve_mesh is None:
        return convert_mesh_openfoam(config_parser)
    else:
        from .convert_ngsolve import convert_mesh_ngsolve  # Import here to NGSolve & OpenCMP dependencies optional
        return convert_mesh_ngsolve(config_parser, ngsolve_mesh)


def convert_velocities_to_flows(cmesh: CMesh, vel_vec: np.ndarray) -> np.ndarray:
    """
    Convert cell-centered velocities to flowrate through facets using upwinding.

    Parameters
    ----------
    * cmesh:      The CMesh to project on.
    * vel_vec:    Velocity vector indexed by element ID.

    Returns
    -------
    * flows_and_upwind: 2D object array indexed by facet ID.
                        - 1st column is volumetric flowrate through facet.
                        - 2nd column is a flag indicating which of a facet's elements are upwind of it.
                            - 0, and 1 represent the index into mesh.facet_elements[facet]
                            - -1 is used for boundary elements to represent
    """
    flow = cmesh.facet_size.copy()
    upwind_element = np.zeros(len(flow), dtype=int)
    for facet, facet_elements in enumerate(cmesh.facet_elements):
        if len(facet_elements) == 1:
            flux = cmesh.facet_normals[facet].dot(vel_vec[facet_elements[0]])
            flow[facet] = abs(flux) * cmesh.facet_size[facet]
            upwind_element[facet] = 0 if flux >= 0 else -1
        elif len(facet_elements) == 2:
            flux_0 = cmesh.get_outward_facing_normal(facet, facet_elements[0]).dot(vel_vec[facet_elements[0]])
            flux_1 = cmesh.get_outward_facing_normal(facet, facet_elements[1]).dot(vel_vec[facet_elements[1]])
            if flux_0 > 0 and flux_1 < 0:  # Element 0 is upwind of the facet since flow is coming out of it
                flow[facet] = abs(flux_0) * cmesh.facet_size[facet]
                upwind_element[facet] = 0
            elif flux_1 > 0 and flux_0 < 0:  # Element 1 is upwind of the facet since flow is coming out of it
                flow[facet] = abs(flux_1) * cmesh.facet_size[facet]
                upwind_element[facet] = 1
            else:  # Can happen due to too coarse of a mesh in regions where flow impinges on itself.
                flux_0, flux_1 = abs(flux_0), abs(flux_1)
                flow[facet] = (flux_0 if flux_0 > flux_1 else flux_1) * cmesh.facet_size[facet]
                upwind_element[facet] = int(not (flux_0 > flux_1))  # Want 1st entry when flux_0 is bigger
        else:
            raise ValueError(f'Facet {facet} has {len(facet_elements)} elements, but should only have 1 or 2.')

    return np.array([flow, upwind_element], dtype=object).transpose()
