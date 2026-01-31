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
Functions related to outputting
"""

import os
import shutil
from collections import defaultdict
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple, Any, Optional

import numpy as np

from ..config_functions import ConfigParser
from ..mesh import CMesh, create_dof_to_element_map

OPENFOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {};
    location    "{}";
    object      {};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<{}> 
{}
(
"""
"""OpenFOAM header template used to output simulation results for visualization."""


def create_element_label_gfu(mesh: 'ngsolve.Mesh') -> 'ngsolve.GridFunction':
    """
    Function which labels each element based on its ID.

    Parameters
    ----------
    * mesh: The NGSolve Mesh to label.

    Returns
    -------
    * gfu: The GridFunction containing the labeled mesh elements.
    """
    from ngsolve import L2, GridFunction

    fes = L2(mesh, order=0, dgjumps=True)
    gfu = GridFunction(fes)

    for element in mesh.Elements():
        gfu.vec[element.nr] = element.nr

    return gfu


def create_compartment_label_gfu(mesh: 'ngsolve.Mesh', compartments: Dict[int, Set[int]]) -> 'ngsolve.GridFunction':
    """
    Create a GridFunction which labels each element with the index of its compartment.
    Elements not in a compartment are labelled with NaN.

    Used for visualization.

    Parameters
    ----------
    * mesh:           The NGSolve mesh used to create the labels.
    * compartments:   Mapping between compartment ID and the element IDs that belong to that compartment.

    Returns
    -------
    * gfu:  Gridfunction on a 0th order DG L2 space (i.e. 1 value per element) whose
            value on each element is the corresponds to the ID of the compartment to which
            that element belongs.
    """
    from ngsolve import L2, GridFunction

    fes = L2(mesh, order=0, dgjumps=True)
    gfu = GridFunction(fes)

    # Fill it with NaN to differentiate elements which are in no compartment from those in the first (0th) compartment
    gfu.vec.data = np.nan * np.empty((gfu.vec.size,))

    # Go through and the created compartments and label them on gfu_out
    for id_compartment in compartments:
        for id_element in compartments[id_compartment]:
            gfu.vec[id_element] = id_compartment

    return gfu

def output_compartment_average_direction_vector(cmesh: CMesh, config_parse: ConfigParser, compartments, dir_vec: np.ndarray, vector_name: str):
    dir_vec_avg = -np.ones_like(dir_vec)
    for compartment, element_IDs in compartments.items():
        dir_vec_avg[list(element_IDs), :] = np.mean(dir_vec[list(element_IDs), :], axis=0)
    magnitude = np.linalg.norm(dir_vec_avg, axis=1)[..., np.newaxis]
    dir_vec_avg /= magnitude

    output_vector_openfoam(cmesh, config_parse, dir_vec_avg, vector_name)

def output_vector_openfoam(cmesh: CMesh, config_parser: ConfigParser, vector: np.ndarray, vector_name: str) -> None:
    """
    Label each mesh element with the corresponding value in the velocity vector

    Args:
        cmesh:          CMesh to print values for.
        config_parser:  OpenCCM ConfigParser.
        vector:         Numpy array of size NxM where N is number of mesh elements and M is components of the array.
        vector_name:    Name to use as a filename and to show up in Paraview.
    """
    time_0 = config_parser.get_list(['SIMULATION', 't_span'], float)[0]
    output_file_name = str(time_0) + '/' + vector_name

    output_folder_path = config_parser.get_item(['SETUP', 'output_folder_path'], str)
    vtu_folder_path = config_parser.get_item(['POST-PROCESSING', 'vtu_dir'], str)
    output_folder = output_folder_path + vtu_folder_path + output_file_name

    # Create the output directory if it doesn't exist
    Path(output_folder_path + vtu_folder_path + str(time_0)).mkdir(parents=True, exist_ok=True)

    write_buffer_to_file_openfoam(vector, output_folder, time_0, vector_name, cmesh)

def label_elements_openfoam(cmesh: CMesh, config_parser: ConfigParser) -> None:
    """
    Label each mesh element with its ID to help with debugging.

    Parameters
    ----------
    * cmesh:          The CMesh to print out labels for
    * config_parser:  The OpenCCM ConfigParser.
    """
    # t_span has to be loaded as a float and then converted to str so that zero gets handled correctly.
    # Asking for t_span directly as str results in 0 being returned instead of 0.0, which will cause two zero folders
    # to be created. One being 0/ created here, and the other 0.0/ created by output the concentration profile to VTK.
    time_0 = config_parser.get_list(['SIMULATION', 't_span'], float)[0]
    output_file_name = str(time_0) + '/' + 'element_labels'

    output_folder_path = config_parser.get_item(['SETUP', 'output_folder_path'], str)
    vtu_folder_path = config_parser.get_item(['POST-PROCESSING', 'vtu_dir'], str)
    output_folder = output_folder_path + vtu_folder_path + output_file_name

    # Create the output directory if it doesn't exist
    Path(output_folder_path + vtu_folder_path + str(time_0)).mkdir(parents=True, exist_ok=True)

    write_buffer_to_file_openfoam(np.arange(len(cmesh.element_sizes)), output_folder, time_0, 'element ID', cmesh)


def label_models_and_dof_openfoam(cmesh: CMesh, model_to_element_map: List[List[Tuple[float, int]]], config_parser: ConfigParser) -> None:
    points_per_pfr      = config_parser.get_item(['SIMULATION',             'points_per_pfr'],      int)
    model_name          = config_parser.get_item(['COMPARTMENT MODELLING',  'model'],               str)
    output_folder_path  = config_parser.get_item(['SETUP',                  'output_folder_path'],  str)
    vtu_folder_path     = config_parser.get_item(['POST-PROCESSING',        'vtu_dir'],             str)

    t0 = config_parser.get_list(['SIMULATION', 't_span'], float)[0]
    output_file_name = str(t0) + '/' + model_name + '_labels'
    output_file_path = output_folder_path + vtu_folder_path + output_file_name

    element_to_model_map = -np.ones(len(cmesh.element_sizes), dtype=int)
    for model, distances_element in enumerate(model_to_element_map):
        for _, element in distances_element:
            element_to_model_map[element] = model
    write_buffer_to_file_openfoam(element_to_model_map, output_file_path, t0, model_name + " id", cmesh)

    output_file_name = str(t0) + '/dof_labels'
    output_file_path = output_folder_path + vtu_folder_path + output_file_name

    dof_to_element_map = create_dof_to_element_map(model_to_element_map, points_per_pfr)

    element_to_dof_map = -np.ones(len(cmesh.element_sizes), dtype=int)
    for dof, mapping in enumerate(dof_to_element_map):
        for element, _, _ in mapping:
            element_to_dof_map[element] = dof

    write_buffer_to_file_openfoam(element_to_dof_map, output_file_path, t0, "dof", cmesh)


def label_compartments_openfoam(output_file_name: str, compartments: Dict[int, Set[int]], config_parser: ConfigParser) -> None:
    """
    Label each mesh element with the ID of the compartment it belongs to.
    A value of -1 is used to represent elements which are not added to any compartment (e.g. those on a no-slip BC).

    Parameters
    ----------
    * output_file_name: The filename to save the labeled mesh value into.
    * compartments:     The compartments to visualize.
    * config_parser:    The OpenCCM ConfigPararser.
    """
    # t_span has to be loaded as a float and then converted to str so that zero gets handled correctly.
    # Asking for t_span directly as str results in 0 being returned instead of 0.0, which will cause two zero folders
    # to be created. One, 0/, created here and the other, 0.0/, created by saving the concentrations to VTK.
    output_file_name = str(config_parser.get_list(['SIMULATION', 't_span'], float)[0]) + '/' + output_file_name

    output_folder_path       = config_parser.get_item(['SETUP',           'output_folder_path'],        str)
    vtu_folder_path          = config_parser.get_item(['POST-PROCESSING', 'vtu_dir'],                   str)
    openfoam_sol_folder_path = config_parser.get_item(['INPUT',           'openfoam_sol_folder_path'],  str)

    # Create the output directory if it doesn't exist
    Path(output_folder_path + vtu_folder_path + output_file_name.split("/")[-2]).mkdir(parents=True, exist_ok=True)

    # Create an empty .FOAM file that can be dragged into Paraview for easy visualization
    # The leading underscore is important as it allows the file to be sorted to the top of directory
    with open(output_folder_path + vtu_folder_path + '_compartmental_model_results.FOAM', 'w'):
        pass

    shutil.copytree(openfoam_sol_folder_path + 'constant',
                    output_folder_path + vtu_folder_path + 'constant',
                    dirs_exist_ok=True)

    element_to_compartment: Dict[int, int] = defaultdict(lambda: -1)
    for compartment_id, elements in compartments.items():
        for element_id in elements:
            element_to_compartment[element_id] = compartment_id
    
    # Extract values from the output file path
    split_output_file_string = output_file_name.split("/")
    object_value = split_output_file_string[-1]
    location_value = split_output_file_string[-2]

    input_file  = config_parser.get_item(['INPUT', 'volume_file_path'], str)
    output_file = output_folder_path + vtu_folder_path + output_file_name

    # Process input and output files (overwriting output file)
    k = 0
    startWrite = False
    with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
        for line in input_file:
            # Modify object value in the output file
            if "object" in line:
                output_file.write(f'    object      {object_value};\n')
                continue
            # Modify location value in the output file
            if "location" in line:
                output_file.write(f'    location    "{location_value}";\n')
                continue
            # Start writing compartment labels
            if line == "(\n" and k == 0:
                startWrite = True
                output_file.write(line)
                continue
            # End writing compartment labels
            if line == ")\n":
                startWrite = False
                output_file.write(line)
                continue
            # Write compartment label for each element
            if startWrite:
                output_file.write(str(element_to_compartment[k]) + "\n")
                k += 1
            else:
                # Copy the original content from input file
                output_file.write(line)


def export_to_vtu_openfoam(
        concentrations_all_time:    np.ndarray,
        ts:                         np.ndarray,
        model_to_element_map:       Optional[List[List[Tuple[float, int]]]],
        config_parser:              ConfigParser,
        cmesh:                      CMesh,
        dof_to_element_map:         Optional[List[List[Tuple[int, int, float]]]] = None,
) -> None:
    """

    Parameters
    ----------
    concentrations_all_time:    Array containing the
    ts:
    model_to_element_map
    config_parser
    cmesh
    dof_to_element_map:
    """
    print("Exporting simulation visualization")

    output_folder_path  = config_parser.get_item(['SETUP',              'output_folder_path'],  str)
    vtu_folder_path     = config_parser.get_item(['POST-PROCESSING',    'vtu_dir'],             str)
    points_per_model    = config_parser.get_item(['SIMULATION',         'points_per_pfr'],      int)
    species_names       = config_parser.get_list(['SIMULATION',         'specie_names'],        str)
    t_span              = config_parser.get_list(['SIMULATION',         't_span'],              float)

    num_elements = len(cmesh.element_sizes)

    if not dof_to_element_map:
        dof_to_element_map = create_dof_to_element_map(model_to_element_map, points_per_model)

    assert len(dof_to_element_map) == concentrations_all_time.shape[1]

    # t0 will contain several visualization files that are constant in time, will symlink to them for speed and small file size
    t0_path = os.path.join(output_folder_path + vtu_folder_path + str(t_span[0]))
    contents_of_t0 = [(file_name, os.path.abspath(os.path.join(t0_path, file_name))) for file_name in os.listdir(t0_path)]

    # Initialize to -1 so that any non-compartmentalized elements can be filtered out (assuming only positive values are valid)
    buffer = -np.ones(num_elements)
    for i_t, t in enumerate(ts):
        output_folder = output_folder_path + vtu_folder_path + str(t)
        # Throw an error if the folder already exists and this is not the first timestep.
        # First timestep will have been previously created to store compartmentalization info.
        Path(output_folder).mkdir(parents=True, exist_ok=(i_t == 0))

        if t != t_span[0]:
            for file_name, original_path in contents_of_t0:
                os.symlink(original_path, os.path.join(output_folder, file_name))

        for i_specie, specie in enumerate(species_names):
            concentrations = concentrations_all_time[i_specie, :, i_t]
            for dof, mapping in enumerate(dof_to_element_map):
                for element, dof_other, weight_dof in mapping:
                    buffer[element] = concentrations[dof] * weight_dof + concentrations[dof_other] * (1 - weight_dof)

            write_buffer_to_file_openfoam(buffer, output_folder + '/c_' + specie, t, specie, cmesh)
    print("Done exporting simulation visualization")


def write_buffer_to_file_openfoam(buffer: np.ndarray, output_file_path: str, t: float, variable_name: str, cmesh: CMesh) -> None:
    if buffer.ndim not in [1, 2]:
        raise ValueError(f'Unexpected dimensionality ({buffer.ndim}) for buffer')
    def line_to_str(buffer_line: np.ndarray) -> str:
        if buffer_line.ndim == 0:
            return str(buffer_line)
        else:
            return '(' + ' '.join(str(val) for val in buffer_line)  + ')'

    value_class = 'volScalarField'  if buffer.ndim == 1 else 'volVectorField'
    value_type  = 'scalar'          if buffer.ndim == 1 else 'vector'

    with open(output_file_path, 'w') as output_file:
        output_file.write(OPENFOAM_HEADER.format(value_class, t, variable_name, value_type, len(buffer)))

        output_file.write('\n'.join(line_to_str(line) for line in buffer))

        output_file.write(
            ")\n"
            ";\n"
            "\n"
            "boundaryField\n"
            "{\n"
        )

        for bc_name, facets in cmesh.bc_to_facet_map.items():
            facet_concentration_values = []
            for facet in facets:
                facet_concentration_values.append(line_to_str(buffer[cmesh.facet_elements[facet][0]]) + "\n")

            facet_concentrations = '\t\t\t'.join(facet_concentration_values)

            output_file.write(f""
                              f"\t{bc_name}\n"
                              f"\t{{\n"
                              f"\t\ttype            calculated;\n"
                              f"\t\tvalue           nonuniform List<{value_type}>\n"
                              f"\t\t{len(facets)}\n"
                              f"\t\t(\n"
                              f"\t\t\t{facet_concentrations}"
                              f"\t\t)\n"
                              f"\t\t;\n"
                              f"\t}}\n")

        output_file.write(
            "}\n"
            "\n"
            "// ************************************************************************* //"
        )


def cstrs_to_vtu_and_save_opencmp(
        system_results: Tuple[
                            np.ndarray,
                            np.ndarray,
                            Dict[int, List[Tuple[int, int]]],
                            Dict[int, List[Tuple[int, int]]]
                        ],
        compartments:   Dict[int, Set[int]],
        config_parser:  ConfigParser,
        mesh: 'ngsolve.Mesh') -> None:
    """
    Takes a time series from running a simulation on the compartment network and outputs it to native OpenFOAM format.
    Each mesh cell is labelled based on which compartment it was a part of.
    Cells which were not part of any compartment are labelled with NaN.

    Parameters
    ----------
    * system_results:   Tuple containing the results of the simulation.
    * compartments:     Lookup between compartment ID and the ID of the elements making it up.
    * config_parser:    OpenCCM ConfigParser used for generating the compartments and running the simulation.
    * mesh:             Mesh from which the compartment network was generated and onto which to project results.
    """
    print("Exporting simulation visualization")
    from ngsolve import L2, GridFunction, VTKOutput

    output_folder_path  = config_parser.get_item(['SETUP',           'output_folder_path'], str)
    vtu_folder_path     = config_parser.get_item(['POST-PROCESSING', 'vtu_dir'],            str)
    subdivisions        = config_parser.get_item(['POST-PROCESSING', 'subdivisions'],       int)
    species_names       = config_parser.get_list(['SIMULATION',      'specie_names'],       str)

    y, ts, _, _ = system_results

    # Create the list containing data structure to hold .pvd file
    output_list = [['<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\"\n' +
                    'byte_order=\"LittleEndian\"\ncompressor=\"vtkZLibDataCompressor\">\n<Collection>\n'] for _ in range(len(species_names))]

    fes = L2(mesh, order=0)
    # The GridFunction which will store the final result
    gfu = GridFunction(fes)
    gfu.vec.data = np.full((gfu.vec.size,), np.nan)

    for t_idx, t in enumerate(ts):
        cs = y[:, :, t_idx]
        for id_specie, name_specie in enumerate(species_names):
            c_specie = cs[id_specie, :]
            for id_compartment, elements_in_compartment in compartments.items():
                c = c_specie[id_compartment]
                for element in elements_in_compartment:
                    gfu.vec[element] = c

            # This path needs to be relative `output_folder_path` since that's where the to the .PVD file will be saved
            filepath = vtu_folder_path + 'compartment_cstr_' + name_specie + '_' + str(t)
            # Must be relative to run directory so that VTKOuput will save the .VTU file in the right place
            filepath_full = output_folder_path + filepath

            VTKOutput(ma=mesh, coefs=[gfu], names=[name_specie],
                      filename=filepath_full, subdivision=subdivisions).Do()

            # Save info about .vtu file for .pvd
            output_list[id_specie].append(f"<DataSet timestep=\"{t}\" group=\"\" part=\"0\" file=\"{filepath+ '.vtu'}\"/>\n")

    for _list in output_list:
        _list.append('</Collection>\n</VTKFile>')

    # Write each line to the .pvd file
    for id_specie, name_specie in enumerate(species_names):
        with open(output_folder_path + f'compartment_cstr_{name_specie}_transient.pvd', 'w') as file:
            for line in output_list[id_specie]:
                file.write(line)

    print("Done exporting simulation visualization")


def pfrs_to_vtu_and_save_opencmp(system_results:    Tuple[
                                                        np.ndarray,
                                                        np.ndarray,
                                                        Dict[int, List[Tuple[int, int]]],
                                                        Dict[int, List[Tuple[int, int]]]],
                                 pfr_network:       Tuple[
                                                        Dict[int, Tuple[Dict[int, int], Dict[int, int]]],
                                                        np.ndarray,
                                                        np.ndarray,
                                                        Dict[int, List[int]],
                                                        List[List[Tuple[float, int]]]
                                                    ],
                                 compartments:      Dict[int, Set[int]],
                                 config_parser:     ConfigParser,
                                 mesh:              'ngsolve.Mesh',
                                 n_vec:             np.ndarray
                                 ) -> None:
    """
    Takes a time series from running a simulation on the compartment network and outputs it to native OpenFOAM format.
    Each mesh cell is labelled based on which compartment it was a part of.
    Cells which were not part of any compartment are labelled with NaN.

    Parameters
    ----------
    * system_results:   Tuple containing the results of the simulation.
    * pfr_network:      The PFR network on which the simulation was run.
    * compartments:     Lookup between compartment ID and the ID of the elements making it up.
    * config_parser:    OpenCCM ConfigParser used for generating the compartments and running the simulation.
    * mesh:             NGSolve ,esh from which the compartment network was generated and onto which to project results.
    * n_vec:            Direction vector over the mesh, indexed by element id.
    """
    print("Exporting simulation visualization")
    from ngsolve import L2, GridFunction

    num_cores           = config_parser.get_item(['SETUP',           'num_cores'],          int)
    output_folder_path  = config_parser.get_item(['SETUP',           'output_folder_path'], str)
    vtu_folder_path     = config_parser.get_item(['POST-PROCESSING', 'vtu_dir'],            str)
    subdivisions        = config_parser.get_item(['POST-PROCESSING', 'subdivisions'],       int)
    interpolant_order   = config_parser.get_item(['POST-PROCESSING', 'interpolant_order'],  int)
    points_per_pfr      = config_parser.get_item(['SIMULATION',      'points_per_pfr'],     int)
    species_names       = config_parser.get_list(['SIMULATION',      'specie_names'],       str)


    y, ts, _, _ = system_results
    connections, volume_pfrs, q_connections, compartment_to_pfr_map, pfr_to_element_map = pfr_network
    all_elements = [e for e in mesh.Elements()]

    fes = L2(mesh, order=0)
    gfus     = []  # GridFunctions for final results
    gfus_tmp = []  # GridFunctions we'll override over and over
    for _ in range(num_cores):
        g = GridFunction(fes)
        gt = GridFunction(fes)
        g.vec.data  = np.full((g.vec.size,),  np.nan)
        gt.vec.data = np.full((gt.vec.size,), np.nan)
        gfus.append(g)
        gfus_tmp.append(gt)

    # Used to project the position vectors of each cell in order to
    n_avg_compartment = np.zeros((len(compartments), n_vec.shape[1]))

    # Nx2 where N is number of compartments
    # 1st column is the minimum dot-product value for the compartment, 2nd is the maximum.
    # Used to scale all other dot-products between 0 and 100%
    d_min_max = np.zeros((len(compartments), 2))

    # Populate the list of min and max values for compartments
    # Calculate the average alignment of each compartment
    for id_compartment, ids_elements_in_compartment in compartments.items():
        n_avg_compartment[id_compartment, :] = np.mean(n_vec[list(ids_elements_in_compartment), :], axis=0)

        d_min =  np.inf
        d_max = -np.inf
        for id_element in ids_elements_in_compartment:
            for vertex in all_elements[id_element].vertices:
                dot = n_avg_compartment[id_compartment, :].dot(mesh.vertices[vertex.nr].point)
                d_min = min(d_min, dot)
                d_max = max(d_max, dot)

        assert d_max > d_min
        d_min_max[id_compartment] = [d_min, d_max]

    # Number of discretization points
    n_points = y.shape[1]
    # The distance of each discretization point, in volume %, from the start of the COMPARTMENT it's a part of
    distance_of_point_i = np.zeros(n_points)

    # Calculate what percentage of the distance of the PFR each compartment makes up
    for id_compartment, ids_pfrs_in_compartment in compartment_to_pfr_map.items():
        volume_compartment = sum(volume_pfrs[id_pfr] for id_pfr in ids_pfrs_in_compartment)

        total_distance = 0.0
        for id_pfr in ids_pfrs_in_compartment:
            id_start = points_per_pfr * id_pfr
            id_stop  = points_per_pfr * (id_pfr+1)-1
            delta_between_points = (100 * volume_pfrs[id_pfr] / volume_compartment) / (points_per_pfr-1)
            for id_point in range(id_start, id_stop+1):  # Needs to be inclusive of both ends
                distance_of_point_i[id_point] = total_distance
                if id_point != id_stop:
                    total_distance += delta_between_points
        assert np.isclose(100, total_distance)

    # Create a list to contain the .pvd entries
    output_list = ['' for _ in range(len(ts))]

    with Pool(processes=num_cores) as pool:
        results = []
        for i in range(len(ts)):
            results.append(pool.apply_async(
                _pfr_vtu_parallel_runner,
                (
                    y[:, :, i],
                    ts[i],
                    gfus[i % num_cores],
                    gfus_tmp[i % num_cores],
                    mesh,
                    compartment_to_pfr_map,
                    compartments,
                    n_avg_compartment,
                    distance_of_point_i,
                    d_min_max,
                    points_per_pfr,
                    vtu_folder_path,
                    output_folder_path,
                    subdivisions,
                    species_names,
                    interpolant_order
                )
            ))

        for i, result in enumerate(results):
            output_list[i] = result.get()

    # Add the header and footer
    output_list.insert(0, '<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\"\n' +
                       'byte_order=\"LittleEndian\"\ncompressor=\"vtkZLibDataCompressor\">\n<Collection>\n')
    output_list.append('</Collection>\n</VTKFile>')

    # Write each line to the .pvd file
    with open(output_folder_path + 'compartment_pfr_transient.pvd', 'a+') as file:
        for line in output_list:
            file.write(line)

    print("Done exporting simulation visualization")


def _pfr_vtu_parallel_runner(
        y:                      np.ndarray,
        t:                      float,
        gfu:                    'ngsolve.GridFunction',
        gfu_tmp:                'ngsolve.GridFunction',
        mesh:                   'ngsolve.Mesh',
        compartment_to_pfr_map: Dict[int, List[int]],
        compartments:           Dict[int, Set[int]],
        n_avg_compartment:      np.ndarray,
        distance_of_point_i:    np.ndarray,
        d_min_max:              np.ndarray,
        points_per_pfr:         int,
        vtu_folder_path:        str,
        output_folder_path:     str,
        subdivisions:           int,
        species_names:          List[str],
        interpolant_order:      int
) -> str:
    """
    Wrapper function used to perform the projection and exporting to VTU in parallel.

    Parameters
    ----------
    * y:                        The (num_species, num_points) vector of simulated values for the given time.
    * t:                        The time corresponding to y, used for the file name and for the VTU entry.
    * gfu:                      Final GridFunction used for output.
    * gfu_tmp:                  Temporary GridFrunction used as a buffer in order to sample the interpolant only over
                                mesh elements which are part of the compartment.
    * mesh:                     The NGSolve mesh object, needed by VTKOutput
    * compartment_to_pfr_map:   Mapping from compartment ID to PFR ID(s).
    * compartments:             The compartments from which the PFR network was created.
    * n_avg_compartment:        Average direction vector for each compartment, indexed by compartment ID.
    * distance_of_point_i:      The distance of each discretization point, in volume %,
                                from the start of the COMPARTMENT it's a part of.
    * d_min_max:                All of the minimum and maximum distance values, indexed by compartment ID.
    * points_per_pfr:           The number of discretization points per PFR, assumed constant.
    * vtu_folder_path:          The path for the
    * output_folder_path:       The path for the VTU output folder, relative to the run direction.
    * subdivisions:             When converting the GridFunction into a VTU file, how many times each element should be
                                subdivided. E.g. with a value of 1, each triangle becomes 4 triangles.
    * species_names:            The name of each species used in the simulation, in the same order as was used
                                in the simulation.
    * interpolant_order:        Order of the interpolant to use to interpolant between discretization points.

    Returns
    -------
    * Entry to put into the .pvd file relating for the processed time step.
    """
    from ngsolve import VTKOutput

    if interpolant_order == 0:
        interpolant = _nearest_point_interpolant
    elif interpolant_order == 1:
        interpolant = _linear_interpolant

    lines_for_pvd = []
    for id_specie, name_specie in enumerate(species_names):
        # Projection concentration onto the GridFunction
        for id_compartment, ids_pfrs_in_compartment in compartment_to_pfr_map.items():
            gfu_tmp.Set(
                _piecewise_projected_results(
                    interpolant,
                    d_min_max[id_compartment],
                    y[id_specie, :],
                    n_avg_compartment[id_compartment],
                    distance_of_point_i,
                    points_per_pfr,
                    ids_pfrs_in_compartment
                )
            )
            # The projection above set all of the mesh elements, not just the ones inside the compartment.
            # Pick out the values associated with the cells in the compartment.
            for id_element in compartments[id_compartment]:
                gfu.vec[id_element] = gfu_tmp.vec[id_element]

        # Must be relative to `output_folder_path` since that's where the to the .PVD file will be saved
        filepath = vtu_folder_path + 'compartment_pfr_' + name_specie + '_' + str(t)
        # Must be relative to the run directory so that VTKOuput will save the .VTU file in the right place
        filepath_full = output_folder_path + filepath

        VTKOutput(ma=mesh, coefs=[gfu], names=[name_specie], filename=filepath_full, subdivision=subdivisions).Do()

        # Save info about .vtu file for .pvd
        lines_for_pvd.append(f"<DataSet timestep=\"{t}\" group=\"\" part=\"0\" file=\"{filepath + '.vtu'}\"/>\n")
    return ''.join(lines_for_pvd)


def _linear_interpolant(d: 'ngsolve.CoefficientFunction', d1: float, d2: float, v1: float, v2: float) -> 'ngsolve.CoefficientFunction':
    """
    1st order interpolating CoefficientFunction between (d1, v1) and (d2, v2) as a function of d.

    Parameters
    ----------
    * d:  Distance CoefficientFunction.
    * d1: Left-side distance value.
    * d2: Right-side distance value.
    * v1: Value at d1.
    * v2: Value at d2

    Returns
    -------
    * Linear interpolating CoefficientFunction between (d1, v1) and (d2, v2) as a function of d.
    """
    return (v2 - v1) / (d2 - d1) * (d - d1) + v1


def _nearest_point_interpolant(d: 'ngsolve.CoefficientFunction', d1: float, d2: float, v1: float, v2: float) -> 'ngsolve.CoefficientFunction':
    """
    Returns a CoefficienFunction which evaluates to either v1 or v2 based on if d is closer to d1 or d2.
    This will result in the values from the first and last degree of freedom in each PFR taking up 1/2 as much
    space as internal nodes:
    | v0 |   v1   | v2 |
      ^1/4   ^1/2   ^ 1/4

    The alternative would have been to have them each equally sized (ignore that the diagram being bigger):
    |  v0  |  v1  |  v2  |
       ^1/3   ^1/3   ^1/3

    Parameters
    ----------
    * d:  Distance CoefficientFunction.
    * d1: Left-side distance value.
    * d2: Right-side distance value.
    * v1: Value at d1.
    * v2: Value at d2

    Returns
    -------
    * Nearest neighbour interpolating CoefficientFunction between (d1, v1) and (d2, v2) as a function of d.
    """
    from ngsolve import IfPos

    d_mid = (d1 + d2) / 2
    # There are 4 domains:
    # 1.            d < d1
    # 2. d1     <=  d < d_mid
    # 3. d_mid  <=  d < d2
    # 4. d2     <   d

    # Because restrict_function_domain will zero out anything outside the domain [d1, d2],
    # the four domains above can be simplified to the following two:
    # 5.          d < d_min     (simplified version of 2)
    # 6. d_min <= d             (simplified version of 3)
    return IfPos(
        d_mid - d,  # Condition
        v1,         # If condition positive, d in (-inf, d_min)
        v2          # Else,                  d in [d_min, inf)
    )


def _piecewise_projected_results(basis_function:         Callable[['ngsolve.CoefficientFunction', float, float, float, float], 'ngsolve.CoefficientFunction'],
                                 d_min_max_compartment:  np.array,
                                 values:                 np.ndarray,
                                 n_avg:                  np.ndarray,
                                 distance_for_points:    np.ndarray,
                                 points_per_pfr:         int,
                                 pfrs_in_compartment:    List[int]) -> 'ngsolve.CoefficientFunction':
    """
    Take the simulation results for a given compartment and project them onto the 1D space whose dimension is
    the % of the compartments' volume.
    This is then in turn mapped onto every mesh element.

    This projection will inherently lead to discontinuities in the projection between PFRs.
    The value at the last node of PFR {i} and that at the first node of PFR {i+1} is almost always not equal.

    Parameters
    ----------
    * basis_function:           The interpolant to use as the basis function.
    * d_min_max_compartment:    The minimum and maximum values from projecting positions vectors inside this compartment
                                onto the average direction of the compartment.
    * values:                   The 1D vector representing values from the simulation, indexed by discretization point ID.
    * n_avg:                    The average direction vector for the compartment.
    * distance_for_points:      The distance along the compartment
    * points_per_pfr:           The number of discretization points per pfr.
    * pfrs_in_compartment:      The IDs of the PFR(s) in this compartment.

    Returns
    -------
    * The piecewise restricted function.
    """
    from ngsolve import CoefficientFunction, x, y

    d_min, d_max = d_min_max_compartment

    # Create the coordinate space.
    # 1D model with the coordinate representing % of the total volume of the compartment.
    dot = x * n_avg[0] + y * n_avg[1]
    d = 100 * (dot - d_min) / (d_max - d_min)

    # Iterate over each PFR in the compartment, and each discretization point within the PFR, creating the interpolant
    # Between the current point and the next point, and add it to the global interpolant for the compartment.
    f = CoefficientFunction(0)
    for id_pfr in pfrs_in_compartment:
        id_start = points_per_pfr * id_pfr
        id_stop  = points_per_pfr * (id_pfr + 1) - 1
        points   = list(range(id_start, id_stop+1))
        for i in range(len(points)-1):
            if id_pfr == pfrs_in_compartment[-1] and i == len(points)-2:
                break  # Handle the final point separately at the end so that it's inclusive of the end point.

            p1 = points[i]
            p2 = p1 + 1
            d1 = distance_for_points[p1]
            d2 = distance_for_points[p2]
            v1 = values[p1]
            v2 = values[p2]
            f += restrict_function_domain(basis_function(d, d1, d2, v1, v2), d, d1, d2)

    # Handle the last point of the domain separately so that it is inclusive of the end point
    end_point = points_per_pfr * (pfrs_in_compartment[-1] + 1) - 1
    p2 = end_point
    p1 = p2 - 1
    d1 = distance_for_points[p1]
    d2 = distance_for_points[p2]
    v1 = values[p1]
    v2 = values[p2]
    f += restrict_function_domain(basis_function(d, d1, d2, v1, v2), d, d1, d2, both_ends=True)
    return f


def restrict_function_domain(f: 'ngsolve.CoefficientFunction', x: float, x_min: float, x_max: float, both_ends: bool = False) -> 'ngsolve.CoefficientFunction':
    """
    Given a CoefficientFunction `f` returns it wrapped in IfPos calls such that f is evaluated
    only over the interval [x_min, x_max).
    If `both_ends` is True, then the interval is changed to be inclusive of both ends, i.e. [x_min, x_max].

    Outside of this interval the function evaluates to 0.

    Parameters
    ----------
    * f:            Function to evaluate.
    * x:            Position to evaluate it at.
    * x_min:        Minimum position to evaluate it at.
    * x_max:        Maximum position to evaluate it at.
    * both_ends:    If True, the interval is change to be inclusive of both ends, i.e. [x_min, x_max].

    Returns
    -------
    * The restricted version of `f`.
    """
    from ngsolve import IfPos

    if both_ends:
        left_bound = IfPos(
            x - x_max,  # Condition
            0,  # If condition positive, x in (x_max, inf)
            f   # Else,                  x in (-inf, x_max]
        )
    else:
        left_bound = IfPos(
            x_max - x,  # Condition
            f,  # If condition positive, x in (-inf, x_max)
            0   # Else,                  x in [x_max, inf)
        )

    return IfPos(
        x_min - x,  # Condition
        0,          # If condition positive, x in (-inf, x_min)
        left_bound  # Else,                  x in [x_min, inf)
    )
