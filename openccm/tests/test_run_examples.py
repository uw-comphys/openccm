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

import importlib
import os
from typing import Optional

import pytest
import shutil

from pathlib import Path
from openccm import run, ConfigParser

rel_path_to_examples = '../../examples/'


def clean_previous_results(configparser: ConfigParser) -> None:
    def delete_and_recreate_folder(path: str) -> None:
        if Path(path).exists():
            shutil.rmtree(path)
            Path(path).mkdir(parents=True)

    if configparser.need_to_update_paths:
        configparser.update_paths()

    delete_and_recreate_folder(configparser['SETUP']['tmp_folder_path'])
    delete_and_recreate_folder(configparser['SETUP']['log_folder_path'])
    delete_and_recreate_folder(configparser['SETUP']['output_folder_path'])


def clean_and_run(working_directory: str, analysis_script: Optional[str] = None) -> None:
    os.chdir(Path(__file__).parents[0])
    configparser = ConfigParser(working_directory + 'CONFIG')
    configparser['SETUP']['working_directory'] = working_directory

    clean_previous_results(configparser)
    if analysis_script:
        analysis_script_path = os.path.abspath(working_directory + analysis_script)
        spec = importlib.util.spec_from_file_location("module.name", analysis_script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_analysis = getattr(module, 'run_analysis')
        run_analysis()
    else:
        run(configparser)


@pytest.mark.skipif(not importlib.util.find_spec("opencmp"), reason="requires OpenCMP")
def test_opencmp_cstr_reversible():
    clean_and_run(working_directory=rel_path_to_examples + 'simple_reactors/cstr/reversible/',
                  analysis_script='cstr_analysis.py')


@pytest.mark.skipif(not importlib.util.find_spec("opencmp"), reason="requires OpenCMP")
def test_opencmp_cstr_irreversible():
    clean_and_run(working_directory=rel_path_to_examples + 'simple_reactors/cstr/irreversible/',
                  analysis_script='cstr_analysis.py')


@pytest.mark.skipif(not importlib.util.find_spec("opencmp"), reason="requires OpenCMP")
def test_opencmp_pfr():
    clean_and_run(working_directory=rel_path_to_examples + 'simple_reactors/pfr/',
                  analysis_script='pfr_analysis.py')


@pytest.mark.skipif(not importlib.util.find_spec("opencmp"), reason="requires OpenCMP")
def test_opencmp_recirc():
    from opencmp.run import run as run_opencmp
    from opencmp.config_functions import ConfigParser as OpenCMPConfigParser
    os.chdir(Path(__file__).parents[0])

    path_to_folder = rel_path_to_examples + 'OpenCMP/pipe_with_recirc_2d/'

    configparser_ccm = ConfigParser(path_to_folder + 'CONFIG')
    configparser_ccm['SETUP']['working_directory'] = path_to_folder

    if not Path(path_to_folder + configparser_ccm['INPUT']['opencmp_sol_file_path']).exists():
        configparser = OpenCMPConfigParser(path_to_folder + 'config_Stokes')
        configparser['MESH']['filename'] = path_to_folder + configparser['MESH']['filename']
        configparser['OTHER']['run_dir'] = path_to_folder
        run_opencmp("", configparser)

        configparser = OpenCMPConfigParser(path_to_folder + 'config_INS')
        configparser['MESH']['filename'] = path_to_folder + configparser['MESH']['filename']
        configparser['OTHER']['run_dir'] = path_to_folder
        run_opencmp("", configparser)

    clean_and_run(path_to_folder)


def test_openfoam_2d_pipe():
    clean_and_run(rel_path_to_examples + 'OpenFOAM/pipe_with_recirc/')


def test_openfoam_3d_recirc():
    clean_and_run(rel_path_to_examples + 'OpenFOAM/pipe_with_recirc_3d/')


def test_openfoam_3d_pipe():
    clean_and_run(rel_path_to_examples + 'OpenFOAM/3d_pipe/')
