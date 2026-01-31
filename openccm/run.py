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

import pickle
from os.path import isfile
from time import perf_counter_ns
from typing import Dict, Union

import numpy as np

from .config_functions import ConfigParser
from .compartment_models import create_model_network
from .compartmentalize import calculate_compartments, create_compartment_network
from .io import load_velocity_and_direction_openfoam
from .mesh import CMesh, convert_mesh, convert_velocities_to_flows
from .postprocessing import convert_to_vtu_and_save, create_element_label_gfu, create_compartment_label_gfu, label_compartments_openfoam, \
                            label_elements_openfoam, label_models_and_dof_openfoam, network_to_rtd, plot_results, visualize_model_network
from .postprocessing.vtu_output import output_vector_openfoam, output_compartment_average_direction_vector
from .system_solvers import solve_system


def run(config_parser_or_file: Union[ConfigParser, str]) -> Dict[str, int]:
    """
    The main function of the OpenCCM packages. Sequentially goes through each step required to
    create the compartmental model, solve a simulation on it, output results, and perform any specified post-processing.

    Parameters
    ----------
    * config_parser_or_file:

    Returns
    -------
    * timing_dict: Mapping between the name of each step and the time, in ns, that the step took.
    """
    timing_dict = {}

    if type(config_parser_or_file) is ConfigParser:
        config_parser = config_parser_or_file
    else:
        config_parser = ConfigParser(config_parser_or_file)
    if config_parser.need_to_update_paths:
        config_parser.update_paths()

    cache_info          = CacheInfo(config_parser)
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
        dir_vec     = np.load(cache_info.name_direction_vector)
        vel_vec     = np.load(cache_info.name_velocity_vector)
        phase_frac  = np.load(cache_info.name_phase_fraction)

        if cache_info.need_opencmp_mesh:
            with ngcore.TaskManager():
                mesh = Mesh(cache_info.name_refined_mesh_opencmp)
        if OpenCMP and output_VTK:
            with ngcore.TaskManager():
                n_gfu = GridFunction(L2(mesh, order=0) ** mesh.dim)
                n_gfu.Load(cache_info.name_direction_sol)
    else:
        if OpenCMP:
            with ngcore.TaskManager():
                mesh, n_gfu, dir_vec, vel_vec = load_opencmp_results(config_parser)
                phase_frac = np.array([1])  # Use 1 to keep rest of the code the same.
                n_gfu.Save(cache_info.name_direction_sol)
        else:
            dir_vec, vel_vec, phase_frac = load_velocity_and_direction_openfoam(config_parser)
        np.save(cache_info.name_direction_vector, dir_vec)
        np.save(cache_info.name_velocity_vector,  vel_vec)
        np.save(cache_info.name_phase_fraction,   phase_frac)
    timing_dict["Load Solution"] = perf_counter_ns() - start

    # Convert to CMesh
    start = perf_counter_ns()
    if cache_info.already_made_cmesh:
        with open(cache_info.name_cmesh, 'rb') as handle:
            c_mesh: CMesh = pickle.load(handle)
    else:
        c_mesh = convert_mesh(config_parser, phase_frac, ngsolve_mesh=mesh if OpenCMP else None)
        with open(cache_info.name_cmesh, 'wb') as handle:
            pickle.dump(c_mesh, handle, protocol=pickle.HIGHEST_PROTOCOL)
    timing_dict['Create CMesh'] = perf_counter_ns() - start

    if not cache_info.already_made_cfd_processed_results:
        output_vector_openfoam(c_mesh, config_parser, vel_vec, 'velocity')
        output_vector_openfoam(c_mesh, config_parser, dir_vec, 'direction')

    # Calculate facet flow_rates
    if cache_info.already_made_flows_and_upwind_file:
        flows_and_upwind: np.ndarray = np.load(cache_info.name_flows_and_upwind, allow_pickle=True)
    else:
        flows_and_upwind = convert_velocities_to_flows(c_mesh, vel_vec)
        np.save(cache_info.name_flows_and_upwind, flows_and_upwind)

    # Calculate the compartments
    start = perf_counter_ns()
    if cache_info.already_made_compartments:
        with open(cache_info.name_compartments_pre, 'rb') as handle:
            compartments_pre = pickle.load(handle)
    else:
        compartments_pre, _ = calculate_compartments(dir_vec, flows_and_upwind, c_mesh, config_parser)
        with open(cache_info.name_compartments_pre, 'wb') as handle:
            pickle.dump(compartments_pre, handle, protocol=pickle.HIGHEST_PROTOCOL)
    timing_dict['Compartmentalize'] = perf_counter_ns() - start

    if not cache_info.already_made_cm_info_vtu:
        if OpenCMP:
            compartment_labels_pre_gfu = create_compartment_label_gfu(mesh, compartments_pre)
        else:
            output_compartment_average_direction_vector(c_mesh, config_parser, compartments_pre, dir_vec, 'direction_avg_pre')
            label_compartments_openfoam('compartments_pre', compartments_pre, config_parser)

    # Turn the compartments into a network
    start = perf_counter_ns()
    if cache_info.already_made_compartment_network:
        with open(cache_info.name_compartment_network, 'rb') as handle:
            compartment_network = pickle.load(handle)
        with open(cache_info.name_compartments_post, 'rb') as handle:
            compartments_post = pickle.load(handle)
        del compartments_pre  # This will have been loaded only for visualization purposes and can be deleted here to save RAM
    else:
        compartments_post, compartment_network = create_compartment_network(compartments_pre, c_mesh, dir_vec, flows_and_upwind, config_parser)
        with open(cache_info.name_compartments_post, 'wb') as handle:
            pickle.dump(compartments_post, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(cache_info.name_compartment_network, 'wb') as handle:
            pickle.dump(compartment_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    timing_dict['Compartment Network'] = perf_counter_ns() - start
    print("End COMPARTMENTALIZE")

    # Convert the compartment network to a CSTR/PFR network
    start = perf_counter_ns()
    if cache_info.already_made_model_network:
        with open(cache_info.name_model_network, 'rb') as handle:
            model_network = pickle.load(handle)
    else:
        model_network = create_model_network(model, compartments_post, compartment_network, c_mesh, dir_vec, flows_and_upwind, config_parser)
        with open(cache_info.name_model_network, 'wb') as handle:
            pickle.dump(model_network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    timing_dict['Reactor Network'] = perf_counter_ns() - start

    del compartment_network  # Changed when creating the PFR network, prevents accidentally using it wrong.

    # Label each element with the compartment ID it belongs to after merging small compartments
    if not cache_info.already_made_cm_info_vtu:
        if OpenCMP:
            compartment_labels_post_gfu = create_compartment_label_gfu(mesh, compartments_post)
            # Create element labels
            element_labels_gfu = create_element_label_gfu(mesh)
            with ngcore.TaskManager():
                VTKOutput(ma=mesh,
                          coefs=[compartment_labels_pre_gfu, compartment_labels_post_gfu, element_labels_gfu],
                          names=['compartment # pre', 'compartment # post', 'element #'],
                          filename=cache_info.name_model_info_opencmp,
                          subdivision=config_parser.get_item(['POST-PROCESSING', 'subdivisions'], int)
                          ).Do()
        else:
            label_elements_openfoam(c_mesh, config_parser)
            output_compartment_average_direction_vector(c_mesh, config_parser, compartments_post, dir_vec,'direction_avg_post')
            label_compartments_openfoam('compartments_post', compartments_post, config_parser)
            label_models_and_dof_openfoam(c_mesh, model_network[-1], config_parser)

    # Visualize the CSTR/PFR network
    if visualize_network:
        visualize_model_network(model_network, compartments_post, c_mesh, dir_vec, config_parser)

    # Solve the CSTR/PFR network
    if run_simulation:
        start = perf_counter_ns()
        system_results = solve_system(model, model_network, config_parser, c_mesh)
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
                                system_results, model_network, compartments_post, config_parser, c_mesh,
                                OpenCMP_mesh=mesh if OpenCMP else None,
                                n_vec=dir_vec if OpenCMP else None)
        timing_dict['VTU Export'] = perf_counter_ns() - start

    return timing_dict


class CacheInfo:
    """
    Helper class to store caching related names and checks.
    """
    def __init__(self, config_parser: ConfigParser):
        model               = config_parser.get_item(['COMPARTMENT MODELLING',  'model'],               str)
        tmp_folder_path     = config_parser.get_item(['SETUP',                  'tmp_folder_path'],     str)
        output_folder_path  = config_parser.get_item(['SETUP',                  'output_folder_path'],  str)
        vtu_folder_path     = config_parser.get_item(['POST-PROCESSING',        'vtu_dir'],             str)
        output_VTK          = config_parser.get_item(['POST-PROCESSING',        'output_VTK'],          bool)
        OpenCMP             = config_parser.get('INPUT', 'opencmp_sol_file_path', fallback=None) is not None

        self.name_cmesh                 = tmp_folder_path + 'cmesh.pickle'
        """Filename for saving the CMesh."""
        self.name_compartments_pre      = tmp_folder_path + model + '_' + 'compartments_pre.pickle'
        """Filename for saving the compartments before merging."""
        self.name_compartments_post     = tmp_folder_path + model + '_' + 'compartments_post.pickle'
        """Filename for saving the compartments after merging."""
        self.name_compartment_network   = tmp_folder_path + model + '_' + 'compartment_network.pickle'
        """Filename for saving the compartment network."""
        self.name_direction_sol         = tmp_folder_path + 'n_gfu.sol'
        """Filename for saving the director in OpenCMP format."""
        self.name_direction_vector      = tmp_folder_path + 'dir_vec.npy'
        """Filename for saving the director in numpy format."""
        self.name_model_info_opencmp     = output_folder_path + model + '_info'
        """Filename for saving the model info for visualziation."""
        self.name_model_network         = tmp_folder_path + model + '_network.pickle'
        """Filename for saving the PFR/CSTR network."""
        self.name_refined_mesh_opencmp  = tmp_folder_path + 'sim_fine.vol'
        """Filename for saving the refined OpenCMP mesh."""
        self.name_velocity_vector       = tmp_folder_path + 'vel_vec.npy'
        """Filename for saving the velocity vector in numpy format."""
        self.name_phase_fraction        = tmp_folder_path + 'phase_frac.npy'
        """Filename for saving the phase fraction data in numpy format."""
        self.name_flows_and_upwind      = tmp_folder_path + 'flows_and_upwind.npy'
        """Filename for saving the facet flows in numpy format."""

        if OpenCMP:
            name_velocity_info          = output_folder_path + 'velocity_info.vtu'
        else:  # OpenFOAM
            t0 = str(config_parser.get_list(['SIMULATION', 't_span'], float)[0])
            openfoam_vtu_folder = f"{output_folder_path}/{vtu_folder_path}/{t0}/"

        self.already_made_cfd_processed_results =   isfile(self.name_direction_vector) \
                                                    and isfile(self.name_velocity_vector) \
                                                    and isfile(self.name_phase_fraction) \
                                                    and ((not OpenCMP or isfile(self.name_direction_sol))
                                                        and (not OpenCMP or isfile(self.name_refined_mesh_opencmp))
                                                        and ((OpenCMP and isfile(name_velocity_info))
                                                            or (not OpenCMP
                                                                and isfile(openfoam_vtu_folder + "velocity")
                                                                and isfile(openfoam_vtu_folder + "direction"))
                                                            )
                                                    )

        self.already_made_cmesh                 = isfile(self.name_cmesh)
        """Bool indicating if the CMesh has already been created and can be loaded instead of needing to be created."""
        self.already_made_compartments          = isfile(self.name_compartments_pre)
        """Bool indicating if the compartments have already been created and can be loaded rather than neeing to be created."""
        self.already_made_compartment_network   = isfile(self.name_compartment_network) and isfile(self.name_compartments_post)
        """Bool indicating if the compartment network has already been created and can be loaded rather than needing to be created."""
        if OpenCMP:
            self.already_made_cm_info_vtu       = isfile(self.name_model_info_opencmp + '.vtu')
            """Bool indicating if the model visualization has already been created."""
        else:
            self.already_made_cm_info_vtu       = (    isfile(openfoam_vtu_folder + 'compartments_post')
                                                   and isfile(openfoam_vtu_folder + 'compartments_pre')
                                                   and isfile(openfoam_vtu_folder + 'dof_labels')
                                                   and isfile(openfoam_vtu_folder + 'element_labels')
                                                   and isfile(openfoam_vtu_folder + 'direction_avg_pre')
                                                   and isfile(openfoam_vtu_folder + 'direction_avg_post'))
            """Bool indicating if the model visualization has already been created."""

        self.already_made_model_network         = isfile(self.name_model_network)
        """Bool indicating if the model network has already been created and can be loaded instead of needing to be created."""
        self.already_made_flows_and_upwind_file = isfile(self.name_flows_and_upwind)
        """Bool indicating if the facet flows have already been created and can be loaded instead of needing to be created."""

        self.need_opencmp_mesh = OpenCMP and (output_VTK
                                              or not (self.already_made_cmesh and self.already_made_cm_info_vtu))
        """Bool indicating if the OpenCMP mesh need to be loaded."""
