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
from typing import List, Tuple, Optional

r"""
The mesh module contains the intermediate mesh representation, CMesh, and the functions required to convert OpenFOAM
and NGSolve meshes to CMesh format.
"""

from typing import Optional

import numpy as np

from .cmesh import CMesh, GroupedBCs

from .convert_openfoam import convert_mesh_openfoam
from ..config_functions import ConfigParser


def convert_mesh(config_parser: ConfigParser, phase_frac: np.ndarray, ngsolve_mesh: Optional['ngsolve.Mesh']) -> CMesh:
    """
    Helper function for cleaing up call site

    Parameters
    ----------
    * config_parser:    The OpenCCM ConfigParser from which to get the required info for conversion.
    * phase_frac:       Fraction of each mesh element taken up by the phase we wish to compartmentalize.
    * ngsolve_mesh:     The NGSolve mesh object to convert if using OpenCMP.

    Returns
    -------
    * cmesh: Internal representation of the mesh inside ngsolve_mesh or pointed to by config_parser
    """
    if ngsolve_mesh is None:
        return convert_mesh_openfoam(config_parser, phase_frac)
    else:
        from .convert_ngsolve import convert_mesh_ngsolve  # Import here to NGSolve & OpenCMP dependencies optional
        return convert_mesh_ngsolve(config_parser, phase_frac, ngsolve_mesh)


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


def create_dof_to_element_map(model_to_element_map: List[List[Tuple[float, int]]], points_per_model: int) -> List[List[Tuple[int, int, float]]]:
    """
    Convert the mapping between PFR and elements in it to a mapping from degree of freedom (dof) in the discretized system
    to the element.

    Args:
    * model_to_element_map:   Mapping between model ID and a list of ordered tuples (distance_in_model, element ID)
    * points_per_model:       Number of discretization points per model. 1 for CSTRs, >1 for PFRs.

    Returns:
    * dof_to_element_map:     Mapping between degree of freedom and the ordered lists of tuples representing the elements
                              that this dof maps to. Tuple contains (element ID, dof_other, weight_this).
                              dof_other and weight_this are used for a linear interpolation of value between the value of
                              this dof and the nearest (dof_other).
    """
    dof_dist = np.linspace(0.0, 1.0, num=points_per_model)
    # delta is not used for CSTR. Using a very small number for delta in order to push the weighing term to clip at 1.0
    # so that the same downstream processing path can be used for both cases.
    delta = 1.0 / (points_per_model - 1) / 2 if points_per_model > 1 else 1e20

    dof_to_element_map: List[List[Tuple[int, int, float]]] = []

    for distances_elements in model_to_element_map:
        distances_elements = distances_elements.copy()
        for i in range(points_per_model):
            dof = len(dof_to_element_map)
            if len(distances_elements) == 0:  # Remaining DOFs don't map to any element
                dof_to_element_map.extend([] for _ in range(i, points_per_model))
                break
            if i == points_per_model - 1:  # Last point in PFR (Must have this branch before the i == 0 for CSTRs)
                dof_other = dof - 1
                dists = distances_elements  # Whatever is left
                dof_to_element_map.append([(element, dof_other, np.clip(1.0 + (dist - 1.0) / (2*delta), 0.0, 1.0)) for dist, element in dists])
            elif i == 0:  # First point in PFR
                dof_other = dof + 1
                j = next((_i for _i, val in enumerate(distances_elements) if val[0] > delta), len(distances_elements))
                dists, distances_elements = distances_elements[:j], distances_elements[j:]

                dof_to_element_map.append([(element, dof_other, np.clip(1.0 - dist / (2*delta), 0.0, 1.0)) for dist, element in dists])
            else:  # Middle Points
                # Before discretization point
                dof_other_1 = dof - 1
                j = next((_i for _i, val in enumerate(distances_elements) if val[0] > dof_dist[i]),
                         len(distances_elements))
                dists, distances_elements = distances_elements[:j], distances_elements[j:]
                mapping_1 = [(element, dof_other_1, np.clip((dist - dof_dist[i-1]) / (2*delta), 0.0, 1.0)) for dist, element in dists]

                if len(distances_elements) == 0:  # Remaining DOFs don't map to any element
                    dof_to_element_map.append(mapping_1)
                    continue

                # After discretization point
                dof_other_2 = dof + 1
                j = next((_i for _i, val in enumerate(distances_elements) if val[0] > dof_dist[i] + delta), len(distances_elements))
                dists, distances_elements = distances_elements[:j], distances_elements[j:]
                dof_to_element_map.append(mapping_1 + [(element, dof_other_2, np.clip((dof_dist[i+1] - dist) / (2*delta), 0.0, 1.0)) for dist, element in dists])
    return dof_to_element_map
