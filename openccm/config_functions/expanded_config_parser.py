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
This file contains that modified ConfigParser along with any relevent helper functions.
"""

import configparser
import ast
import os
from os.path import isfile
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union, cast, Optional, Tuple, Callable
from os import cpu_count


T = TypeVar('T', bool, str, int, float)
"""Used only for type hints."""

config_defaults: Dict = {
    'SETUP': {'num_cores': cpu_count()//2,
              'DEBUG': False,
              'working_directory': './',
              'output_folder_path': 'output_ccm/',
              'tmp_folder_path': 'cache/',
              'log_folder_path': 'log/'},
    'INPUT': {'min_magnitude_threshold': 0,
              'min_alpha_threshold': 0,
              'openfoam_velocity_file_name': 'U'},
    'COMPARTMENTALIZATION': {'angle_threshold': 5,
                             'flow_threshold': 45},
    'COMPARTMENT MODELLING': {'flow_threshold': 1e-15,
                              'flow_threshold_facet': 1e-15,
                              'dist_threshold': 5. / 100,
                              'atol_opt': 1e-2},
    'SIMULATION': {'run': True,
                   'points_per_pfr': 2,
                   'first_timestep': 0.0001,
                   'solver': 'LSODA',
                   'reactions_file_path': 'None',
                   'rtol': 1e-6,
                   'atol': 1e-6,
                   't_eval': 'all',
                   'boundary_conditions': ''},
    'POST-PROCESSING': {'save_to_file': True,
                        'plot_results': False,
                        'output_vtk': False,
                        'calculate_rtd': False,
                        'network_diagram': False,
                        'subdivisions': 0,
                        'interpolant_order': 1},
}
"""
Default values for the various options available in OpenCCM.
If a default is used, OpenCCM outputs a message in the terminal.
"""


class ConfigParser(configparser.ConfigParser):
    """
    OpenCCM's modified ConfigParser extended to have several useful functions added to it.
    """

    def __init__(self, config_file_path: str) -> None:
        """
        Only initializer to use.

        Parameters
        ----------
        config_file_path: The path to the config file to load, relative to run directory.
        """
        super().__init__()

        self.need_to_update_paths = True
        """
        Boolean indicating whether the paths need to be updated.
        Path are updated with `update_paths` which can be called manually or will be called automatically
        at the top of `openccm.run.run`.
        """

        if not isfile(config_file_path):
            raise FileNotFoundError('The given config file \"{}\" does not exist.'.format(config_file_path))

        self.read(config_file_path)

        # Validate model choice
        model = self.get_item(['COMPARTMENT MODELLING', 'model'], str).lower()
        if model not in ['cstr', 'pfr']:
            raise ValueError(f"Invalid model type ({model}) specified.")
        else:
            self['COMPARTMENT MODELLING']['model'] = model

            # Set points_per_pfr for CSTR as well to make post-processing functions be generic
            if model == 'cstr':
                if 'points_per_pfr' in self['SIMULATION'] and self['SIMULATION']['points_per_pfr'] != '1':
                    print(f"CSTR model chosen with points_per_pfr > 1 "
                          f"({self['SIMULATION']['points_per_pfr']} was chosen) "
                          f"this value is being changed to 1.")
                self['SIMULATION']['points_per_pfr'] = '1'

        # Validate OpenCMP vs OpenFOAM choice
        OpenCMP  = self.get('INPUT', 'opencmp_sol_file_path',    fallback=None) is not None
        OpenFOAM = self.get('INPUT', 'openfoam_sol_folder_path', fallback=None) is not None
        if not OpenCMP and not OpenFOAM:
            raise ValueError('Please specify either "opencmp_sol_file_path" or "openfoam_sol_folder_path".')
        elif OpenCMP and OpenFOAM:
            raise ValueError('Please specify only one of "opencmp_sol_file_path" or "openfoam_sol_folder_path".')

        if OpenCMP:
            try:
                from opencmp import run
            except ModuleNotFoundError:
                raise ValueError("OpenCMP and its dependencies are not installed but you specified an OpenCMP input file."
                                 "Please install OpenCMP and its dependencies.")

        # Load defaults for any default-able values that were not specified
        for key, sub_dict in config_defaults.items():
            if key not in self:
                self.add_section(key)
            for key_sub, val_sub in sub_dict.items():
                if key_sub not in self[key]:
                    print(f'Using the default value of {val_sub} for {key}, {key_sub}.')
                    self[key][key_sub] = str(val_sub)

        # Specify default vtu output directory
        # Done separately from the rest of the models since it's not known ahead of time
        if 'vtu_dir' not in self['POST-PROCESSING']:
            self['POST-PROCESSING']['vtu_dir'] = f"compartment_{self['COMPARTMENT MODELLING']['model'].lower()}_vtu/"
            print(f"Using the default value of {self['POST-PROCESSING']['vtu_dir']} for POST-PROCESSING, vtu_dir.")

        if self['SIMULATION']['run'] == 'True':
            # Validate species
            specie_names = [name for name in self.get_list(['SIMULATION', 'specie_names'], str)]
            if 'specie_names' not in self['SIMULATION'] or len(specie_names) == 0:
                raise ValueError('Need to specify at least 1 specie if running a simulation on the compartmental model.')
            for i, specie_name in enumerate(specie_names):
                if specie_name in ['x', 'y', 'z', 't', 'S']:
                    raise ValueError(f"{specie_name} is a reserved symbol and cannot be used, as a specie name.")

            # Validate BCs
            bc_string = self.get_item(['SIMULATION', 'boundary_conditions'], str)
            for reserved_name in ['point']:
                if reserved_name in bc_string:
                    raise ValueError("'point' is a reserved keyword and cannot be used in boundary conditions names.")

    def update_paths(self) -> None:
        """
        OpenCCM is run from a given directory (folder). This folder is referred to as the running directory.

        This running directory may not be the same as the directory in which all the files are stored.
        The config file specifies a path to the "working directory" (default is the running directory) which
        is the directory in which to look for inputs and where to place outputs.

        All paths in the OpenCCM config file are given as relative to that directory.
        This method converts those paths to be relative to the running directory.
        """
        working_directory = self['SETUP']['working_directory']

        self['SETUP']['log_folder_path']    = working_directory + self['SETUP']['log_folder_path']
        self['SETUP']['tmp_folder_path']    = working_directory + self['SETUP']['tmp_folder_path']
        self['SETUP']['output_folder_path'] = working_directory + self['SETUP']['output_folder_path']

        # Get path to reactions file containing reactions for system
        if self.get('SIMULATION', 'reactions_file_path', fallback=None) != 'None':
            self['SIMULATION']['reactions_file_path'] = working_directory + self.get_item(['SIMULATION', 'reactions_file_path'], str)

        # Convert OpenCMP/OpenFOAM specific paths to relative and save them into the ConfigParser
        if self.get('INPUT', 'opencmp_sol_file_path', fallback=None) is not None:  # OpenCMP
            self['INPUT']['opencmp_sol_file_path']      = working_directory + self['INPUT']['opencmp_sol_file_path']
            self['INPUT']['opencmp_config_file_path']   = working_directory + self['INPUT']['opencmp_config_file_path']
        else:  # OpenFOAM
            openfoam_sol_folder_path = working_directory + self.get_item(['INPUT', 'openfoam_sol_folder_path'], str)
            self['INPUT']['openfoam_sol_folder_path'] = openfoam_sol_folder_path

            sim_folder_to_use = self.get('INPUT', 'openfoam_sim_folder_to_use', fallback=self._find_highest_number())
            self['INPUT']['openfoam_sim_folder_to_use'] = sim_folder_to_use

            self['INPUT']['face_file_path']      = openfoam_sol_folder_path + "constant/polyMesh/faces"
            self['INPUT']['point_file_path']     = openfoam_sol_folder_path + "constant/polyMesh/points"
            self['INPUT']['owner_file_path']     = openfoam_sol_folder_path + "constant/polyMesh/owner"
            self['INPUT']['neighbour_file_path'] = openfoam_sol_folder_path + "constant/polyMesh/neighbour"
            self['INPUT']['boundary_file_path']  = openfoam_sol_folder_path + "constant/polyMesh/boundary"
            if Path(openfoam_sol_folder_path + sim_folder_to_use + "/V").exists():  # OpenFOAM 10
                self['INPUT']['volume_file_path'] = openfoam_sol_folder_path + sim_folder_to_use + "/V"
            elif Path(openfoam_sol_folder_path + sim_folder_to_use + "/Vc").exists():  # OpenFOAM 11
                self['INPUT']['volume_file_path'] = openfoam_sol_folder_path + sim_folder_to_use + "/Vc"
            else:
                raise FileNotFoundError("Could not find mesh cell size file V or Vc.")
            self['INPUT']['velocity_file_path']  = openfoam_sol_folder_path + sim_folder_to_use + "/" + self['INPUT']['openfoam_velocity_file_name']
            if 'openfoam_liquid_phase_fraction' in self['INPUT']:
                self['INPUT']['alpha_file_path']    = openfoam_sol_folder_path + sim_folder_to_use + "/" + self['INPUT']['openfoam_liquid_phase_fraction']

        # NOTE: This one is left as relative to `output_folder_path` because of its final use in the .PVD file
        vtu_folder_path = self.get_item(['POST-PROCESSING', 'vtu_dir'], str)

        # Ensure output folders exist, create them if they don't
        Path(self['SETUP']['log_folder_path']).mkdir(parents=True, exist_ok=True)
        Path(self['SETUP']['tmp_folder_path']).mkdir(parents=True, exist_ok=True)
        Path(self['SETUP']['output_folder_path'] + vtu_folder_path).mkdir(parents=True, exist_ok=True)

        self.need_to_update_paths = False

    def _find_highest_number(self) -> str:
        """
        Find the highest numeric value from directory names.

        If none can be found then a ValueError is raised.

        Returns
        -------
        * highest_name: The folder name corresponding to the largest time value.
        """
        directory_path = self['INPUT']['openfoam_sol_folder_path']

        highest_number = None
        highest_name = ''
        for name in os.listdir(directory_path):
            if name.isnumeric():
                number = float(name)
                if highest_number is None or number > highest_number:
                    highest_number = number
                    highest_name = name

        if highest_number is None:
            raise ValueError(f'Specified folder ({directory_path}) does not contain folders in the OpenFOAM time step format.')
        else:
            return highest_name

    def get_list(self, config_keys: List[str], val_type: Type[T]) -> List[T]:
        """
        Function to load a list of parameters from the config file.

        Parameters
        ----------
        * config_keys:  The keys needed to access the parameters from the config file.
        * val_type:     The type that each parameter is supposed to be, and to which it will be converted.

        Returns
        -------
        * List of the parameters from the config file, converted to the specified type.
        """
        section, key = config_keys
        try:
            params_tmp = self[section][key].split(', ')
        except KeyError:
            raise ValueError(f"Need to specify a value for {section}, {key}")

        ret_list = []
        for param in params_tmp:
            if val_type == bool:
                ret_list.append(param == 'True')
            else:
                ret_list.append(val_type(param))

        return ret_list

    def get_item(self, config_keys: List[str], val_type: Type[T]) -> T:
        """
        Function to load a parameter from the config file.

        Parameters
        ----------
        * config_keys:  The keys needed to access the parameters from the config file.
        * val_type:     The type that the parameter is supposed to be, and to which it will be converted.

        Returns
        -------
        * The parameter from the config file converted to the specified type.
        """
        section, key = config_keys
        try:
            param = self[section][key]
        except KeyError:
            raise ValueError(f"Need to specify a value for {section}, {key}")

        if val_type == bool:
            return param.lower() == 'true'
        else:
            return val_type(param)
    
    def get_expression(self, config_keys: List[str]):
        """
        Function to load and evaluate a Python expression from the config file.

        Parameters
        ----------
        * config_keys: The keys needed to access the expression from the config file.

        Returns
        -------
        *   The evaluated result of the Python expression as a single value of the specified type,
            or a list/tuple of the specified type.
        """
        section, key = config_keys
        return ast.literal_eval(self[section][key])

    def __hash__(self):
        """
        Hash function used for generating unique reaction and boundary condition files for each simulation run.
        NOT for putting the config parser into a dicitonary or similar.

        Returns
        -------
        * _hash: Hash value of the dict
        """
        hashes = []
        for section in self.sections():
            for key, value in self[section].items():
                hashes.append(hash(value))
        return hash(tuple(hashes))
