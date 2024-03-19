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

from .vtu_output import cstrs_to_vtu_and_save_opencmp, pfrs_to_vtu_and_save_opencmp, cstrs_to_vtu_and_save_openfoam, \
                        create_element_label_gfu, create_compartment_label_gfu, label_compartments_openfoam
from .analysis import network_to_rtd, plot_results, visualize_model_network


def convert_to_vtu_and_save(OpenCMP: bool, model: str, system_results, model_network, compartments, config_parser, cmesh, **kwargs) -> None:
    if OpenCMP:
        if model == 'pfr':
            pfrs_to_vtu_and_save_opencmp(system_results, model_network, compartments, config_parser, kwargs['mesh'], kwargs['n_vec'])
        else:
            cstrs_to_vtu_and_save_opencmp(system_results, compartments, config_parser, kwargs['mesh'])
    else:
        if model == 'pfr':
            raise NotImplementedError("PFR visualization for OpenFOAM input not yet implemented")
        else:
            cstrs_to_vtu_and_save_openfoam(system_results, compartments, config_parser, cmesh)
