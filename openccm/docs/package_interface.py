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

# Package Interface

`openccm` can be interfaced with in two different ways:

1. Configuration file method
    *   Parameters are specified in a configuration file which is then given to the `openccm` package through either
        a command line entry point, or through importing `openccm.run.run` and passing the configuration file location
        to it.
2. Python interface method
    *   The more traditional approach of writing your own python scripts and calling the different functions
        that `openccm` provides.

All example are implemented using the configuration file method, and can be run by either running `openccm CONFIG` from
the terminal or by running the `run_compartment.py` file within each example's directory.

# Example Config Files

## Main Config File
Below is an example config file showing all of the available options for OpenCCM as well as a description of
each option's function.
    .. include:: ../../CONFIG

## Reaction Config File
Below is an example config file for specifying reactions.
It models the following two-step reaction involving species $\textrm{a}, \textrm{b}, \textrm{c}$:

$$ \textrm{a} \xrightarrow[]{10} 2\textrm{b} $$
$$ 3\textrm{b} \xrightarrow[]{5} \textrm{c} $$

    .. include:: ../../CONFIG_REACTIONS
"""