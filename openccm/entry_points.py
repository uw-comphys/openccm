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

import sys
from .run import run


def run_openccm():
    """
    Main function that runs OpenCCM.

    Args (from command line):
        config_file_path: Filename of the config file to load. Required parameter.
    """
    if len(sys.argv) == 1:
        print("ERROR: Provide configuration file path.")
        exit(1)
    elif len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print('ERROR: More than one argument was provided.')
        print('OpenCCM requires only one (1) argument which must be a path to the configuration file.')
        exit(1)
