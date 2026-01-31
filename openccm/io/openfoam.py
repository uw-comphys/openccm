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
All functions related to file IO for OpenFOAM CFD results.
"""

import re
from typing import Tuple, Union, TypeVar, List, Type, Dict

import numpy as np

from openccm.config_functions import ConfigParser

T = TypeVar('T', int, float)


def load_velocity_and_direction_openfoam(config_parser: ConfigParser) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the velocity vector from file and calculate the direction vector.
    Both are indexed by element id.

    Parameters
    ----------
    * config_parser:  The OpenCCM ConfigParser for the simulation.

    Returns
    -------
    * dir_vec:    Numpy array of direction vectors where the ith row represents the ith mesh element.
    * vel_vec:    Numpy array of velocity vectors where the ith row represents the ith mesh element.
    """
    print("Start LOAD")
    min_magnitude_threshold = config_parser.get_item(['INPUT', 'min_magnitude_threshold'], float)
    min_alpha_threshold = config_parser.get_item(['INPUT', 'min_alpha_threshold'], float)

    vel_vec = read_mesh_data(config_parser.get_item(['INPUT', 'velocity_file_path'], str), float)
    if 'alpha_file_path' in config_parser['INPUT']:
        phase_frac = read_mesh_data(config_parser.get_item(['INPUT', 'alpha_file_path'], str), float)
        phase_frac[phase_frac < min_alpha_threshold] = 0

        if len(vel_vec) != len(phase_frac):
            raise ValueError(f"Velocity vector (n={len(vel_vec)}) and phase fraction (n={len(phase_frac)}) do not have the same number of values.")
        if len(phase_frac.shape) != 1:
            raise ValueError(f"The specified phase fraction not a scalar, shape is {phase_frac.shape}")

        vel_vec *= phase_frac.reshape(-1, 1)
    else:
        phase_frac = np.array([1])  # Use 1 to keep rest of the code the same.

    magnitude = np.linalg.norm(vel_vec, axis=1)
    # Any vectors that had a magnitude of 0 don't need to be zero'd out since they're already zero
    # Change their magnitude to 1 (chosen arbitrarily) so that the division can be done all at once.
    indices = magnitude < min_magnitude_threshold
    if np.any(indices):
        magnitude[indices] = 1
        vel_vec[indices, :] = 0
    dir_vec = vel_vec / magnitude.reshape((len(vel_vec), 1))

    print("End LOAD")
    return dir_vec, vel_vec, phase_frac


def read_boundary_condition(bc_file_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Read boundary conditions from a file.

    Parameters
    ----------
    * bc_file_path: The path to the file containing the boundary conditions.

    Returns
    -------
    * bc_names: Mapping between boundary names and a tuple of (start_face, n_faces).
    """
    with open(bc_file_path, "r") as file:
        content = file.read()

    pattern = r"(?<=\n)\s*([\w-]+)\s*{[^}]+nFaces\s+(\d+);\s*startFace\s+(\d+);(?:[^}]+inGroups[^}]+List<word>\s*\d+\(\w+\);)?\s*}"
    matches = re.findall(pattern, content)

    bc_names: Dict[str, Tuple[int, int]] = {}
    for match in matches:
        boundary_name = match[0]
        n_faces       = int(match[1])
        start_face    = int(match[2])
        bc_names[boundary_name] = (start_face, n_faces)
    return bc_names


def read_mesh_data(file_path: str, dataType: Type[T]) -> Union[List[List[T]], np.ndarray]:
    """
    Read mesh data from a file.

    Parameters
    ----------
    * file_path:    The path to the file containing the mesh data.
    * dataType:     The data type to use for parsing the data.
                    This value is only used if the entries as scalar.
                    If the entries as vectors, e.g. "(10 3 4 5)\\n", then floats will always be used.

    Returns
    -------
    * data: The mesh data as a List of Lists for handling ragged entries (e.g. number of vertices per element)
            or numpy array for uniformly sized entries.
    """
    if dataType not in (int, float):
        raise ValueError("Invalid dataType specified!")

    with open(file_path, 'r') as file:
        prev_line = ""
        while True:
            line = file.readline()

            if line == "(\n":
                num_entries = int(prev_line)
                break
            else:
                prev_line = line

        line = file.readline()
        if '(' in line and ')' in line:  # Vectors for each line
            data = ['' for _ in range(num_entries)]
            i_left = line.find('(')
            i_right = line.find(')')
            data[0] = np.fromstring(line[i_left + 1:i_right], sep=' ', dtype=dataType)

            ragged = False
            len_ref = len(data[0])

            for i in range(1, num_entries):  # Start at 1! Had to use the first line to see if scalar or vector
                line = file.readline()
                i_left = line.find('(')
                i_right = line.find(')')
                new_data = [float(val) for val in line[i_left + 1:i_right].split(' ')]
                data[i] = new_data

                ragged |= len_ref != len(new_data)

            if not ragged:
                data = np.array(data, dtype=dataType)
        else:
            data = np.zeros(num_entries, dtype=dataType)
            data[0] = dataType(line)
            for i in range(1, num_entries):  # Start at 1! First (0th) line to see if scalar or vector
                data[i] = dataType(file.readline())
    return data
