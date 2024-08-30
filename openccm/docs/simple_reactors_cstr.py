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
# Overview
Two examples are shown for solving reactions in a network of a single CSTR and comparing their results to an analytic solution.
The relevant code for these two examples can be found under `examples/simple_reactors/cstr` or online [here](https://github.com/uw-comphys/openccm/tree/main/examples/simple_reactors/cstr).
The first example shows a set of irreversible reactions, the second shows a reversible reaction.

Note that both of these examples require the optional [OpenCMP](https://opencmp.io/index.html) dependency installed,
see the [Installation Instructions](installation_guide) for help installing it.
The reproduce the results of the examples below run the `cstr_analysis.py` script in the corresponding folder.

.. include:: ../../examples/simple_reactors/cstr/irreversible/analysis/cstr_irreversible_example.md

.. include:: ../../examples/simple_reactors/cstr/reversible/analysis/cstr_reversible_example.md
"""