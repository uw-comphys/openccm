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

from typing import Tuple, Any
from pathlib import Path

import numpy as np

from numpy import ndarray

from ..config_functions import ConfigParser


def load_opencmp_results(config_parser: ConfigParser) -> Tuple[Any, Any, ndarray, ndarray]:
    """
    Function to load the OpenCMP results and return them

    NOTE: All file paths are assumed to be relative to the working directory.

    Args:
        config_parser: Custom OpenCCM ConfigParser object from which to get values

    Returns:
        mesh:     NGsolve mesh representing the original mesh but refined {num_refinements} times.
        n_gfu:    Direction vector projected on the refined mesh using a 0th order fes (1 value per element)
        dir_vec:  Direction vector on the REFINED mesh.
        vel_vec:  Velocity vector using ORIGINAL fes but on the REFINED mesh.
    """
    print("Start LOAD")
    import ngsolve as ngs
    from netgen.read_gmsh import ReadGmsh

    # Number of times to (uniformly) refine the mesh by splitting.
    # This is done so that when projecting the *higher-order* velocity/director results onto
    # the 0th order fes, more of the information is retained.
    num_refinements         = config_parser.get_item(['INPUT',           'num_refinements'],         int)
    subdivisions            = config_parser.get_item(['POST-PROCESSING', 'subdivisions'],            int)
    min_magnitude_threshold = config_parser.get_item(['INPUT',           'min_magnitude_threshold'], float)

    # File paths
    opencmp_config_file_path    = config_parser.get_item(['INPUT', 'opencmp_config_file_path'], str)
    sol_file_path               = config_parser.get_item(['INPUT', 'opencmp_sol_file_path'],    str)
    # Folder paths
    output_folder_path  = config_parser.get_item(['SETUP', 'output_folder_path'],   str)
    tmp_folder_path     = config_parser.get_item(['SETUP', 'tmp_folder_path'],      str)

    mesh_file_path = _mesh_file_path_from_opencmp_config(config_parser)

    ngmesh_c = ReadGmsh(mesh_file_path)
    mesh_c = ngs.Mesh(ngmesh_c)
    fes_mixed_c = fes_from_opencmp_config(opencmp_config_file_path, mesh_c)

    gfu_mixed_c = ngs.GridFunction(fes_mixed_c)
    gfu_mixed_c.Load(sol_file_path)
    v_gfu_c = gfu_mixed_c.components[0]

    ngmesh_f = ReadGmsh(mesh_file_path)
    for _ in range(num_refinements):
        ngmesh_f.Refine()
    ngmesh_f.Save(tmp_folder_path + 'sim_fine.vol')
    mesh_f = ngs.Mesh(tmp_folder_path + 'sim_fine.vol')
    fes_f = fes_from_opencmp_config(opencmp_config_file_path, mesh_f).components[0]
    v_gfu = ngs.GridFunction(fes_f)
    v_gfu.Set(v_gfu_c)

    # Calculate n
    n_gfu = _calculate_n(v_gfu, min_magnitude_threshold, mesh_f)
    dir_vec: np.ndarray = np.array(n_gfu.vec).reshape(n_gfu.dim, mesh_f.ne).transpose()

    # Project the velocity vector onto a 0th order interpolant to have one value per mesh element
    v = ngs.GridFunction(ngs.L2(mesh_f, order=0, dgjumps=True) ** mesh_f.dim)
    v.Interpolate(v_gfu)
    vel_vec: np.ndarray = np.array(v.vec).reshape(v.dim, mesh_f.ne).transpose()

    # Output direction and magnitude of velocity
    ngs.VTKOutput(ma=mesh_f,
                  coefs=[v_gfu, n_gfu, v],
                  names=['velocity', 'direction', 'velocity 0th'],
                  filename=output_folder_path + 'velocity_info',
                  subdivision=subdivisions).Do()
    print("End LOAD")
    return mesh_f, n_gfu, dir_vec, vel_vec


def _calculate_n(velocity, min_magnitude_threshold: float, mesh):
    """
    Calculate the velocity direction field.
    Locations where the velocity magnitude is below the specified threshold are set to zero velocity.

    Args:
        velocity:                   The velocity field.
        min_magnitude_threshold:    Magnitude threshold below which a velocity is set to zero.
        mesh:                       The NGSolve mesh object on which the simulation was solved.

    Returns:
        ~: GridFunction: The velocity direction vector field.
    """
    import ngsolve as ngs
    # Finite element spaces for the direction vector and teh velocity magnitude.
    # Not using the same ones as the original velocity since we need specific properties from these spaces
    # E.g. for the direction vector we want one value per mesh element
    fes_n = ngs.L2(mesh, order=0) ** mesh.dim

    # Calculate magnitude of velocity
    velocity_mag = ngs.sqrt(ngs.InnerProduct(velocity, velocity))

    # Create orientation vector
    velocity_mag_normalized = ngs.IfPos(velocity_mag - min_magnitude_threshold, velocity_mag, 0)
    # Using IfPost to make sure we're not dividing by zero
    n_coef = ngs.IfPos(velocity_mag_normalized, velocity / velocity_mag_normalized, velocity.dim * (0,))
    n = ngs.GridFunction(fes_n)
    n.Interpolate(n_coef)

    return n


def _mesh_file_path_from_opencmp_config(openccm_config_parser: ConfigParser) -> str:
    """
    Function to get the path to the mesh file used by the OpenCMP simulation.

    Args:
        openccm_config_parser: The OpenCCM ConfigParser.

    Returns:
        str: The absolute path to the mesh file used by the OpenCMP simulation.
    """
    from opencmp.config_functions import ConfigParser as OpenCMPConfigParser

    opencmp_config_file_path    = openccm_config_parser.get_item(['INPUT', 'opencmp_config_file_path'], str)
    opencmp_config_parser       = OpenCMPConfigParser(opencmp_config_file_path)

    # This file path, per OpenCMP spec, is relative to the OpenCMP config file.
    mesh_file_path_rel = opencmp_config_parser.get_item(['MESH', 'filename'], str)

    # Convert path relative to OpenCMP config file to one that's relative to the working directory.
    mesh_file_path_abs = str(Path(Path(opencmp_config_file_path).parent.absolute(), mesh_file_path_rel))

    return mesh_file_path_abs


def fes_from_opencmp_config(opencmp_configfile_path: str, mesh):
    """
    Function to read the OpenCMP Configfile and create the necessary finite element spaces onto which
    the simulation results can be loaded.

    Args:
        opencmp_configfile_path:    Path to the OpenCMP ConfigFile used for the simulation
        mesh:                       NGSolve mesh object to create the element spaces on.

    Returns:
        ~:  The full finite element space.
    """
    import ngsolve as ngs
    from opencmp.config_functions import ConfigParser as OpenCMPConfigParser

    # Load OpenCMP ConfigParser
    config_parser = OpenCMPConfigParser(opencmp_configfile_path)

    # Load finite element spaces
    interp_order = config_parser.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)
    elements     = config_parser.get_dict(['FINITE ELEMENT SPACE', 'elements'], "", None, all_str=True)
    DG           = config_parser.get_item(['DG', 'DG'], bool)

    fes_u = getattr(ngs, elements.pop('u'))(mesh, order=interp_order, dgjumps=DG)
    fes_p = getattr(ngs, elements.pop('p'))(mesh, order=interp_order - 1, dgjumps=DG)
    _fes = [fes_u, fes_p]

    for element_space in elements.values():
        _fes.append(getattr(ngs, element_space)(mesh, order=interp_order, dgjumps=DG))

    return ngs.FESpace(_fes, dgjumps=DG)
