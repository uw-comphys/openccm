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
from .cmesh import CMesh, GroupedBCs

from .convert_openfoam import convert_mesh_openfoam
from ..config_functions import ConfigParser


def convert_mesh(OpenCMP: bool, config_parser: ConfigParser, **kwargs) -> CMesh:
    if OpenCMP:
        # Done in order to keep NGSolve & OpenCMP dependency optional
        from .convert_ngsolve import convert_mesh_ngsolve
        return convert_mesh_ngsolve(config_parser, kwargs['mesh'])
    else:
        return convert_mesh_openfoam(config_parser)
