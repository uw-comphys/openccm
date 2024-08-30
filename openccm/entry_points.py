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
OpenCCM has two entry points:
1. `openccm` for creating and running the compartmental model, invoked as `openccm CONFIG_FILE_PATH.
2. `openccm-tests` for running the provided pytests, invoked as `openccm-tests`.
"""

import os
import subprocess
import sys
from pathlib import Path

from .run import run


def run_openccm():
    """
    Function for running OpenCCM using entry points.
    Invoked as `openccm CONFIG_FILE_PATH`.

    Parameters (from command line)
    ----------
    * config_file_path: Filename of the config file to load. Required parameter.
    """
    if len(sys.argv) == 1:
        print("ERROR: Provide configuration file path."
              "Usage: openccm CONFIG_FILE_PATH")
        exit(1)
    elif len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print("ERROR: More than one argument was provided."
              "Usage: openccm CONFIG_FILE_PATH")
        exit(1)


def run_tests():
    """
    Main function for running all unit tests.
    Should only be used for the `openccm-tests` entry-point.
    """
    print("Starting unit tests, this should take a few seconds.")

    pyinterp = sys.executable
    tests_dir = os.path.join(Path(__file__).parents[0], 'tests')

    # automatically find and run all unit tests using unittest built-in discovery feature
    subprocess.call([pyinterp, '-B', '-m', 'pytest', '-v'], cwd=tests_dir)
