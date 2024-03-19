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

import pickle
from os.path import isfile
from time import perf_counter_ns
from typing import Dict, Union

import numpy as np

from .config_functions import ConfigParser
from .compartment_models import create_model_network
from .compartmentalize import calculate_compartments, create_compartment_network
from .io import load_velocity_and_direction_openfoam
from .mesh import CMesh, convert_mesh
from .postprocessing import convert_to_vtu_and_save, create_element_label_gfu, create_compartment_label_gfu, label_compartments_openfoam, \
                            network_to_rtd, plot_results, visualize_model_network
from .postprocessing.vtu_output import label_elements_openfoam
from .system_solvers import solve_system


def run(config_parser_or_file: Union[ConfigParser, str]) -> Dict[str, int]:
    """

    Args:
        config_parser_or_file:`

    Returns:
        ~: A dictionary containing mapping between the name of each step and the time, in ns, that the step took.
    """
    timing_dict = {}

    if type(config_parser_or_file) is ConfigParser:
        config_parser = config_parser_or_file
    else:
        config_parser = ConfigParser(config_parser_or_file)
    if config_parser.need_to_update_paths:
        config_parser.update_paths()

    cache_info    = CacheInfo(config_parser)
    OpenCMP             = config_parser.get('INPUT', 'opencmp_sol_file_path', fallback=None) is not None
    model               = config_parser.get_item(['COMPARTMENT MODELLING', 'model'], str)
    output_folder_path  = config_parser.get_item(['SETUP', 'output_folder_path'], str)
    visualize_network   = config_parser.get_item(['POST-PROCESSING', 'network_diagram'], bool)

    run_simulation      = config_parser.get_item(['SIMULATION', 'run'], bool)
    calculate_rtd       = run_simulation and config_parser.get_item(['POST-PROCESSING', 'calculate_rtd'], bool)
    should_plot_results = run_simulation and config_parser.get_item(['POST-PROCESSING', 'plot_results'], bool)
    output_VTK          = run_simulation and config_parser.get_item(['POST-PROCESSING', 'output_VTK'], bool)

    if OpenCMP:
        import pyngcore as ngcore
        from ngsolve import GridFunction, L2, Mesh, VTKOutput
        from .io import load_opencmp_results
        ngcore.SetNumThreads(config_parser.get_item(['SETUP', 'num_cores'], int))

    start = perf_counter_ns()
    if cache_info.already_made_cfd_processed_results:
        dir_vec = np.load(cache_info.name_direction_vector)
        vel_vec = np.load(cache_info.name_velocity_vector)

        if cache_info.need_opencmp_mesh:
            with ngcore.TaskManager():
                mesh = Mesh(cache_info.name_refined_mesh)
        if OpenCMP and output_VTK:
            with ngcore.TaskManager():
                n_gfu = GridFunction(L2(mesh, order=0) ** mesh.dim)
                n_gfu.Load(cache_info.name_direction_sol)
    else:
        if OpenCMP:
            with ngcore.TaskManager():
                mesh, n_gfu, dir_vec, vel_vec = load_opencmp_results(config_parser)
                n_gfu.Save(cache_info.name_direction_sol)
        else:
            dir_vec, vel_vec = load_velocity_and_direction_openfoam(config_parser)
        np.save(cache_info.name_direction_vector, dir_vec)
        np.save(cache_info.name_velocity_vector,  vel_vec)
    timing_dict["Load Solution"] = perf_counter_ns() - start

    # Calculate the compartments
    start = perf_counter_ns()
    if cache_info.already_made_compartments and cache_info.already_made_cfmesh_pruned:
        with open(cache_info.name_compartments, 'rb') as handle:
            compartments = pickle.load(handle)
        with open(cache_info.name_cmesh_pruned, 'rb') as handle:
            c_mesh = pickle.load(handle)
    else:
        if cache_info.already_made_cfmesh:
            with open(cache_info.name_cmesh, 'rb') as handle:
                c_mesh: CMesh = pickle.load(handle)
        else:
            c_mesh = convert_mesh(OpenCMP, config_parser, mesh=mesh if OpenCMP else None)
            with open(cache_info.name_cmesh, 'wb') as handle:
                pickle.dump(c_mesh, handle, protocol=pickle.HIGHEST_PROTOCOL)
        timing_dict['Create CMesh'] = perf_counter_ns() - start

        compartments, _ = calculate_compartments(dir_vec, c_mesh, config_parser)
        with open(cache_info.name_compartments, 'wb') as handle:
            pickle.dump(compartments, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_info.name_cmesh_pruned, 'wb') as handle:
            pickle.dump(c_mesh, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not cache_info.already_made_cm_info_vtu:
        if OpenCMP:
            compartment_labels_pre = create_compartment_label_gfu(mesh, compartments)
        else:
            label_elements_openfoam(c_mesh, config_parser)
            label_compartments_openfoam('compartments_pre', compartments, config_parser)

    # Turn the compartments into a network
    if cache_info.already_made_compartment_network:
        with open(cache_info.name_compartment_network, 'rb') as handle:
            compartment_network = pickle.load(handle)
        # Note: Don't need to load compartments since it was loaded earlier up and not changed since
    else:
        compartments, compartment_network = create_compartment_network(compartments, c_mesh, dir_vec, vel_vec, config_parser)
        with open(cache_info.name_compartments, 'wb') as handle:
            pickle.dump(compartments, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_info.name_compartment_network, 'wb') as handle:
            pickle.dump(compartment_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    timing_dict['Compartmentalize'] = perf_counter_ns() - start

    # Label each element with the compartment ID it belongs to after merging small compartments
    if not cache_info.already_made_cm_info_vtu:
        if OpenCMP:
            compartment_labels_post = create_compartment_label_gfu(mesh, compartments)
            # Create element labels
            element_labels_gfu = create_element_label_gfu(mesh)
            with ngcore.TaskManager():
                VTKOutput(ma=mesh,
                          coefs=[compartment_labels_pre, compartment_labels_post, element_labels_gfu],
                          names=['compartment # pre', 'compartment # post', 'element #'],
                          filename=cache_info.name_model_info,
                          subdivision=config_parser.get_item(['POST-PROCESSING', 'subdivisions'], int)
                          ).Do()
        else:
            label_compartments_openfoam('compartments_post', compartments, config_parser)
    print("End COMPARTMENTALIZE")

    # Convert the compartment network to a CSTR/PFR network
    start = perf_counter_ns()
    if cache_info.already_made_model_network:
        with open(cache_info.name_model_network, 'rb') as handle:
            model_network = pickle.load(handle)
    else:
        model_network = create_model_network(model, compartments, compartment_network, c_mesh, vel_vec, dir_vec, config_parser)
        with open(cache_info.name_model_network, 'wb') as handle:
            pickle.dump(model_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    timing_dict['Compartment Modelling'] = perf_counter_ns() - start

    # Visualize the CSTR/PFR network
    if visualize_network:
        visualize_model_network(model_network, compartments, c_mesh, dir_vec, config_parser)

    # Solve the CSTR/PFR network
    if run_simulation:
        start = perf_counter_ns()
        system_results = solve_system(model, model_network, config_parser, c_mesh.grouped_bcs)
        np.save(output_folder_path + model + '_concentrations.npy', system_results[0])
        np.save(output_folder_path + model + '_t.npy', system_results[1])
        timing_dict['Solve model'] = perf_counter_ns() - start

    # Calculate RTD
    if calculate_rtd:
        start = perf_counter_ns()
        rtd = network_to_rtd(system_results, c_mesh, config_parser, model_network)
        timing_dict['RTD Calcs'] = perf_counter_ns() - start

        np.save(output_folder_path + model + '_rtd.npy', rtd)
    else:
        rtd = None

    # Plot the results
    if should_plot_results:
        plot_results(system_results, c_mesh, config_parser, rtd)

    # Output results to vtu for visualization
    if output_VTK:
        start = perf_counter_ns()
        convert_to_vtu_and_save(OpenCMP, model,
                                system_results, model_network, compartments, config_parser, c_mesh,
                                mesh=mesh if OpenCMP else None,
                                n_vec=dir_vec if OpenCMP else None)
        timing_dict['Export to VTK'] = perf_counter_ns() - start

    return timing_dict


class CacheInfo:
    def __init__(self, config_parser: ConfigParser):
        model               = config_parser.get_item(['COMPARTMENT MODELLING',  'model'],               str)
        output_VTK          = config_parser.get_item(['POST-PROCESSING',        'output_VTK'],          bool)
        tmp_folder_path     = config_parser.get_item(['SETUP',                  'tmp_folder_path'],     str)
        output_folder_path  = config_parser.get_item(['SETUP',                  'output_folder_path'],  str)

        OpenCMP             = config_parser.get('INPUT', 'opencmp_sol_file_path', fallback=None) is not None

        self.name_cmesh                 = tmp_folder_path + 'cmesh.pickle'
        self.name_cmesh_pruned          = tmp_folder_path + 'cmesh_pruned.pickle'
        self.name_compartments          = tmp_folder_path + 'compartments.pickle'
        self.name_compartment_network   = tmp_folder_path + 'compartment_network.pickle'
        self.name_direction_sol         = tmp_folder_path + 'n_gfu.sol'
        self.name_direction_vector      = tmp_folder_path + 'dir_vec.npy'
        self.name_model_info            = output_folder_path + model + '_info'
        self.name_model_network         = tmp_folder_path + model + '_network.pickle'
        self.name_refined_mesh          = tmp_folder_path + 'sim_fine.vol'
        self.name_velocity_info         = output_folder_path + 'velocity_info.vtu'
        self.name_velocity_vector       = tmp_folder_path + 'vel_vec.npy'

        self.already_made_cfd_processed_results = isfile(self.name_direction_vector) \
                                                  and isfile(self.name_velocity_vector) \
                                                  and (not OpenCMP or isfile(self.name_direction_sol)
                                                       and isfile(self.name_refined_mesh)
                                                       and isfile(self.name_velocity_info))

        self.already_made_cfmesh                = isfile(self.name_cmesh)
        self.already_made_cfmesh_pruned         = isfile(self.name_cmesh_pruned)
        self.already_made_compartments          = isfile(self.name_compartments)
        self.already_made_compartment_network   = isfile(self.name_compartment_network)
        self.already_made_cm_info_vtu           = isfile(self.name_model_info + '.vtu')
        self.already_made_model_network         = isfile(self.name_model_network)

        self.need_opencmp_mesh = OpenCMP and (output_VTK
                                              or not (self.already_made_cfmesh and self.already_made_cm_info_vtu))
