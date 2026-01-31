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
Functions related to parsing user-provided reaction mechanisms and create a numba compiled function which is then used
for the simulations.
"""

import inspect
from pathlib import Path

import numpy as np
from pyparsing import Combine, Group, Literal, ParseResults, Suppress, Word, ZeroOrMore, nums, alphas, OneOrMore
from typing import List, Dict, Optional, Tuple

from ..config_functions import ConfigParser
from configparser import ConfigParser as ConfigParserPlain

import pyparsing as pp
import sympy as sp

from ..io import read_mesh_data


def generate_reaction_system(config_parser: ConfigParser,
                             dof_to_element_map: List[List[Tuple[int, int, float]]],
                             _ddt_reshape_shape: Optional[Tuple[int, int, int]]
                             ) -> str:
    """
    Main function that handles support for systems of chemical reactions. In order, this function:
    1. Performs an initial parsing of the reactions configuration file based on the two main headers, [REACTIONS] and [RATE CONSTANTS].
    2. Parses reactions (and associated rates) in the same species order as they appear specified in the main configuration file.
    3. Creates a runtime-generated function containing the differential reaction system of equations for each specie which contributes to overall mass balance.

    Parameters
    ----------
    * config_parser:        OpenCCM ConfigParser which contains configuration file information and location (i.e. relative path).
    * dof_to_element_map:   Mapping between degree of freedom and the ordered lists of tuples representing the elements
                            that this dof maps to. Tuple contains (element ID, dof_other, weight_this).
                            dof_other and weight_this are used for a linear interpolation of value between the value of
                            this dof and the nearest (dof_other).
    * _ddt_reshape_shape:   Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction occurring at it. Used by `create_reaction_code`

    Returns
    -------
    * rxn_file_name:        The file name, without the .py, in which the reactions are saved.
    """
    input_file      = config_parser.get_item(['SIMULATION', 'reactions_file_path'], str)
    rxn_species     = config_parser.get_list(['SIMULATION', 'specie_names'],        str)
    tmp_folder_path = config_parser.get_item(['SETUP', 'tmp_folder_path'],          str)

    if input_file == 'None':
        # Create dummy file if no reactions specified
        rxn_file_name = 'reaction_code_gen_blank'
        rxn_file_path = tmp_folder_path + '/' + rxn_file_name + '.py'
        with open(rxn_file_path, 'w') as file:
            file.write("from numba import njit\n")
            file.write("from numpy import *\n\n\n")
            file.write("@njit\n")
            file.write("def _reactions(C, _ddt):\n")
            file.write("    return\n\n")
    else:
        rxn_config_parser = ConfigParserPlain()
        rxn_config_parser.optionxform = str  # Case-sensitive parsing of keys
        rxn_config_parser.read(input_file)

        all_reactions:      Dict[str, str]  = dict(rxn_config_parser['REACTIONS'])      if rxn_config_parser.has_section('REACTIONS')       else {}
        all_rate_constants: Dict[str, str]  = dict(rxn_config_parser['RATE CONSTANTS']) if rxn_config_parser.has_section('RATE CONSTANTS')  else {}
        extra_terms_str:    Dict[str, str]  = dict(rxn_config_parser['EXTRA TERMS'])    if rxn_config_parser.has_section('EXTRA TERMS')     else {}

        rxn_file_name = f'reaction_code_gen_{hash(config_parser)}_{[hash(tuple(all_reactions.items())), hash(tuple(all_rate_constants.items())), hash(tuple(extra_terms_str.items()))]}'
        rxn_file_path = tmp_folder_path + '/' + rxn_file_name + '.py'

        reaction_eqns           = parse_reactions(all_reactions, all_rate_constants) if all_reactions else {}
        extra_terms_for_file    = generate_extra_terms_for_reactions(config_parser, dof_to_element_map, extra_terms_str, _ddt_reshape_shape)

        # Generate runtime function containing differential reaction system of equations for mass balance.
        create_reaction_code(rxn_species, reaction_eqns, rxn_file_path, extra_terms_for_file, _ddt_reshape_shape)

    return rxn_file_name


def generate_extra_terms_for_reactions(config_parser:       ConfigParser,
                                       dof_to_element_map:  List[List[Tuple[int, int, float]]],
                                       extra_terms_str:     Dict[str, str],
                                       _ddt_reshape_shape:  Optional[Tuple[int, int, int]]) -> str:
    """
    Helper function for converting the expressions for the extra terms into runnable code for the reaction file.

    Three types of expressions are supported:

    1.  A closed form expression, potentially involving math operations For example:

            a: 3
            b: sin(2*pi/0.12)

    2.  A closed form expression involving **other extra terms**. For example:

            radius: 10
            area:   pi * radius**2

    **Note: The order of terms is important. A term must be first defined before being used.**
    The above example is valid, but the example below is **not** valid since `radius` is used before being defined.

            area:   pi * radius**2
            radius: 10

    3.  Load a result in the native format of the CFD software used for creating the hydrodynamics.
        The format is:

            CFD, file_name

        The CFD keyword specifies to read the specific file in the CFD's native format.
        The file is assumed to in the same folder as the hydrodynamics result.
        Those results are then mapped to a numpy array that is indexed by the global degree of freedom.

    Parameters
    ----------
    * config_parser:        OpenCMP config parser to read values from.
    * dof_to_element_map:   Mapping between degree of freedom and the ordered lists of tuples representing the elements
                            that this dof maps to. Tuple contains (element ID, dof_other, weight_this).
                            dof_other and weight_this are used for a linear interpolation of value between the value of
                            this dof and the nearest (dof_other).
    * extra_terms_str:      Mapping between the name of the extra terms and the string representation of their value.
    * _ddt_reshape_shape:   Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction occurring at it.

    Returns
    -------
    * A string representation for the python code required for loading these extra terms by the reaction file.
    """
    if _ddt_reshape_shape:
        _ddt_reshape_shape = _ddt_reshape_shape[1:]  # 1st index is number of species, letting broadcasting take care of that

    i_to_interpolate: List[int]         = []
    elements_for_dof: List[List[int]]   = []
    for i, mapping in enumerate(dof_to_element_map):
        if len(mapping) == 0:
            i_to_interpolate.append(i)
        elements_for_dof.append([element for element, _, _ in mapping])

    def partition_i(i_to_interpolate: List[int], elements_for_dof: List[List[int]]) -> Tuple[List[int], List[int], List[int]]:
        i_left, i_middle, i_right = [], [], []
        j = 0
        while j < len(i_to_interpolate):
            i = i_to_interpolate[j]
            if i == 0:
                if len(elements_for_dof[1]) > 0:
                    i_right.append(i_to_interpolate.pop(j))
                    continue
            elif i == len(elements_for_dof) - 1:
                if len(elements_for_dof[-2]) > 0:
                    i_left.append(i_to_interpolate.pop(j))
                    continue
            else:
                left = len(elements_for_dof[i-1]) > 0
                right = len(elements_for_dof[i+1]) > 0
                if left and right:
                    i_middle.append(i_to_interpolate.pop(j))
                    continue
                elif left:
                    i_left.append(i_to_interpolate.pop(j))
                    continue
                elif right:
                    i_right.append(i_to_interpolate.pop(j))
                    continue
            j += 1

        return i_left, i_middle, i_right

    while i_to_interpolate:
        # i_to_interpolate is shrunk by partition_i on each iteration, eventually it will be empty.
        left, middle, right = partition_i(i_to_interpolate, elements_for_dof)
        for i in middle:  # 1. Interpolate all entries which have two full neighbours
            elements_for_dof[i] = elements_for_dof[i - 1] + elements_for_dof[i + 1]
        for i in left:  # 2. Interpolate all entries which have a neighbour to their left
            elements_for_dof[i] = elements_for_dof[i - 1]
        for i in right:  # 3. Interpolate all entries which have a neighbour to their right
            elements_for_dof[i] = elements_for_dof[i + 1]

    OpenCMP = config_parser.get('INPUT', 'opencmp_sol_file_path', fallback=None) is not None

    if not OpenCMP:
        openfoam_sol_folder_path    = config_parser.get_item(['INPUT', 'openfoam_sol_folder_path'], str)
        sim_folder_to_use           = config_parser.get('INPUT', 'openfoam_sim_folder_to_use')
        openfoam_sim_folder_path    = openfoam_sol_folder_path + sim_folder_to_use

    tmp_folder_path = config_parser.get_item(['SETUP', 'tmp_folder_path'], str)

    points_per_pfr  = config_parser.get_item(['SIMULATION', 'points_per_pfr'], int)
    model           = config_parser.get_item(['COMPARTMENT MODELLING', 'model'], str)

    if 'CFD' in extra_terms_str:
        raise KeyError("'CFD' is reserved and cannot be used as the name for an extra variable, "
                       "please use a different name, or change the capitalization.")

    tmp_buffer = np.zeros(len(dof_to_element_map))
    extra_terms_for_file = []
    np.seterr(all='raise')
    for name, expression in extra_terms_str.items():
        if expression[:3] == 'CFD':
            if OpenCMP:
                raise NotImplementedError('Loading of arbitrary OpenCMP results not supported.')
            numpy_name = f"{name}_{model}_{points_per_pfr}.npy"

            file_name = expression.split(',')[1].replace(' ', '')
            if not Path(f"{tmp_folder_path}{numpy_name}").exists():
                data = read_mesh_data(openfoam_sim_folder_path + '/' + file_name, float)
                for i, mapping in enumerate(elements_for_dof):
                    tmp_buffer[i] = np.mean(data[mapping])
                if _ddt_reshape_shape:
                    buffer_view_to_save = tmp_buffer.reshape(_ddt_reshape_shape)[..., 1:]  # [..., 1:] to remove first DOF since those are handled seperately
                else:
                    buffer_view_to_save = tmp_buffer
                np.save(f"{tmp_folder_path}{numpy_name}", buffer_view_to_save)
            extra_terms_for_file.append(f"{name} = np.load('{tmp_folder_path}{numpy_name}')")
        else:
            extra_terms_for_file.append(f"{name} = {expression}")
    return '\n'.join(extra_terms_for_file)


def _organize_reactions_input(all_reactions:      Dict[str, str],
                              all_rate_constants: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    This function first performs several checks to ensure that the reactions config file was handled correctly by the user. 
    If successful, it combines the reactions and rate constants into a list of paired values.

    Parameters
    ----------
    * all_reactions:        The partially-parsed reactions as extracted from reactions configuration file.
    * all_rate_constants:   The partially-parsed reaction rates as extracted from reactions configuration file.

    Returns
    -------
    * rxn_book:     List of (rxn equation, rate constant) pairings.
    """
    # Validity checks on the input
    rxn_ids = list(all_reactions.keys())
    rate_ids = list(all_rate_constants.keys())
    # 1. Duplicate reaction or rate labels are present (i.e., 2 or more definitions for a reaction or rate expression). These definitions must be unique
    if (len(rxn_ids) != len(set(rxn_ids))
            or len(rate_ids) != len(set(rate_ids))):
        raise RuntimeError("Duplicate reaction and/or rate labels detected. Check reactions configuration file.")
    # 2. If the number of unique reaction IDs does not equal rate IDs, then there are missing reactions or rates
    elif len(rxn_ids) != len(rate_ids):
        raise RuntimeError("Number of reactions does not match number of rates (or their reaction IDs are different). Check reactions configuration file.")
    # 3. Each reaction does not have a related rate
    elif sorted(rxn_ids) != sorted(rate_ids):
        raise RuntimeError("Cannot find rate for one or more reactions. Check that reaction and rate IDs align.")

    rxn_book: List[Tuple[str, str]] = [(rxn_eqn, all_rate_constants[rxn_id]) for rxn_id, rxn_eqn in all_reactions.items()]
    return rxn_book


def parse_reactions(all_reactions:      Dict[str, str],
                    all_rate_constants: Dict[str, str]) -> Dict[str, str]:
    """
    Parses all reactions and rates and creates the differential reaction system of equations.
    See the Notes section below for further description of expectations and limitations of the reactions configuration parser.

    Notes
    -----
    -   The reaction parser only supports forward (written) reactions (i.e. use of ->).
        Reversible reactions must be written as two independent reactions with their associated rate constants.
    -   The reactions parser can parse the general reaction: aA1 + bB2 + [...] -> gG7 + hH8 + [...].
    -   All chemical species must be alphanumeric.
        -   We recommend using "m" or "p" for + and - ions, respectively.
            I.e.: Na+ should be written as Nap in the reactions and main configuration file.

    An appropriate reactions configuration file for the reversible reaction N2O4 <-> 2NO2 with k_f = 1e-2 and k_r = 3e-4 (these are hypothetical rates) would be:
        [REACTIONS]
        R1: N2O4 -> 2NO2
        R2: 2NO2 -> N2O4

        [RATE CONSTANTS]
        R1: 1e-2
        R2: 3e-4

    Parameters
    ----------
    * all_reactions:        Mapping between reaction number and the unparsed reaction.
    * all_rate_constants:   Mapping between reaction number and the unparsed rate constant.

    Returns
    -------
    * de_eqns:  Mapping between specie name and the net reaction rate.
    """

    def to_dict(rxn_parsed: ParseResults) -> Dict[str, str]:
        """
        Convert the pyparsing result into a dictionary.
        Supports one of two formats for each item in rxn_parsed:
        1. Explicit coefficient: ['a', 'A']
        2. Implicit coefficient: ['A']

        where:
        - A: A chemical specie in the reaction
        - a: coefficient for specie A

        When converting to the dictionary version all implicit coefficients will be made explicit using a value of '1'.

        Parameters
        ----------
        * rxn_parsed: Pyparsing result with potential implicit coefficients

        Returns
        -------
        * rxn_dict: Mapping between specie and its coefficient.
        """

        rxn_dict = {}
        for term in rxn_parsed:
            if len(term) == 1:  # Implicit coefficient
                rxn_dict[term[0]] = '1'
            else:  # Explicit coefficient
                rxn_dict[term[1]] = term[0]

        return rxn_dict

    def update_lhs_rhs(de_eqn:          Dict[str, str],
                       side_dict:       Dict[str, str],
                       reactants_dict:  Dict[str, str],
                       rate_const:      str,
                       sign:            str) -> None:
        """
        Helper function to convert the species, coefficients, and rate constants into net reaction rate for each specie,
        one side of each reaction at a time.

        Parameters
        ----------
        * de_eqn:           Mapping between species and their net reaction rate.
        * side_dict:        Mapping between species and their stoichiometric coefficients for this half of the reaction.
        * reactants_dict:   Mapping between species and their stoichiometric coefficients for the reactants of this reaction.
        * rate_const:       The rate constant for this reaction.
        * sign:             Whether a negative sign needs to be added to the rate, such as for products.

        Returns
        -------
        * Nothing. `lhs` and `rhs` are updated in-place.
        """
        for specie, coeff in side_dict.items():
            if specie not in de_eqn.keys():
                de_eqn[specie] = ''

            if rate_const[:2] == 'z=':  # 0th order reaction
                de_eqn[specie] += sign + coeff + '*' + rate_const[2:]
            else:
                rxn_rate = '*'.join([_s + '**' + _c for _s, _c in reactants_dict.items()])
                de_eqn[specie] += sign + coeff + '*' + rate_const + '*' + rxn_rate

    # The following pyparsing structure outlines the expected reaction structure from reactions config file.
    rxn_arrow               = Literal('->')
    coefficient             = ZeroOrMore(Word(nums, '.' + nums))  # Match integers & floats
    element                 = Word(alphas, alphas)
    subscript               = ZeroOrMore(Word(nums, '.' + nums))
    molecule_w_coefficient  = Group(
                                pp.Optional(coefficient) +
                                Combine(OneOrMore(
                                    Combine(element + pp.Optional(subscript))
                                ))
                            )
    all_molecules           = ZeroOrMore(molecule_w_coefficient + Suppress('+') | molecule_w_coefficient)
    reactants               = all_molecules + Suppress(rxn_arrow)
    products                = Suppress(reactants) + all_molecules

    de_eqns: Dict[str, str] = {}
    rxn_book = _organize_reactions_input(all_reactions, all_rate_constants)
    for rxn_eqn, rate_constant in rxn_book:
        # Parse, convert to dict, and make implicit coefficient explicit for the products and reactants of each reaction.
        reactants_dict  = to_dict(reactants.parseString(rxn_eqn))
        products_dict   = to_dict(products.parseString(rxn_eqn))

        # Update the lhs and rhs of differential equation reaction system for the reactants:
        update_lhs_rhs(de_eqns, reactants_dict, reactants_dict, rate_constant, sign="-")
        update_lhs_rhs(de_eqns, products_dict, reactants_dict, rate_constant, sign="+")

    return de_eqns


def create_reaction_code(rxn_species:           List[str],
                         reaction_eqns:         Dict[str, str],
                         rxn_file_path:         str,
                         extra_terms:           str,
                         _ddt_reshape_shape:    Optional[Tuple[int, int, int]] = None
                         ) -> None:
    """
    Symbolically creates and simplifies the rhs of the system of differential equations for reaction environment using sympy.
    Then, this rhs is translated into source code and written into a separate .py file at runtime.

    Parameters
    ----------
    * rxn_species:          Name of species in the order that they appear in the main config file.
    * reaction_eqns:        Mapping between specie name and the net reaction for that specie.
    * rxn_file_path:        The path to the file in which to save the generated reactions.
    * extra_terms:          The extra terms, if any, in string format ready to write to the reactions file.
    * _ddt_reshape_shape:   Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction occurring at it.
    """
    # If an empty dict was passed for reaction_eqns, it means no reactions are involved in simulation (i.e. tracer experiment).
    # Therefore take species given and add dummy *0 so that the reaction terms are not contributing to mass balance as desired.
    if reaction_eqns:
        reaction_list = [reaction_eqns[specie] for specie in rxn_species]
    else:
        reaction_list = [specie + '*0' for specie in rxn_species]

    # Convert string ODE system to sympy expression type ODE system
    rhs_symb = [sp.parse_expr(eqn) for eqn in reaction_list]

    # Convert into function and then to string (i.e. source code)
    the_source = inspect.getsource(sp.lambdify([rxn_species], rhs_symb, 'numpy'))

    # Change function name to reactions
    the_source = the_source.replace("_lambdifygenerated","_reactions")

    # Change Dummy variable name
    dummy_name = the_source[the_source.find('(')+1:the_source.find(')')]
    the_source = the_source.replace(dummy_name, 'specie_concentrations')

    # Add _ddt parameter
    close_bracket = the_source.find(')')
    source_left, source_right = the_source[:close_bracket], the_source[close_bracket:]
    if _ddt_reshape_shape:
        source_right = source_right.replace('specie_concentrations', f'specie_concentrations.reshape({_ddt_reshape_shape})[..., 1:]')
    the_source = source_left + ', _ddt' + source_right

    # Remove the unneeded return statement
    the_source = the_source[0:the_source.find('return')].strip() + '\n\n'

    # Create a view into _ddt that we can easily work with
    # (n_species, n_pfr * points_per_pfr) -> (n_species, n_pfr, points_per_pfr)
    if _ddt_reshape_shape:  # Only for PFRs
        the_source +=  '    # Reshape concentrations and _ddt to apply reaction only to non-inlet nodes\n'
        the_source += f'    _ddt = _ddt.reshape({_ddt_reshape_shape})[..., 1:]\n'
        the_source += '    \n'

    # Unroll _ddt update for each specie
    rhs_strs = [str(eqn) for eqn in rhs_symb]

    for id_specie, rhs in enumerate(rhs_strs):
        the_source += f'    # Update for specie {rxn_species[id_specie]}\n'
        the_source += f'    _ddt[{id_specie}, :] += {rhs} \n\n'

    # Write system of differential reaction equations to runtime-generated file
    with open(rxn_file_path, 'w') as file:
        file.write("from numba import njit\n")
        file.write("import numpy as np\n")
        file.write("\n")
        file.write(extra_terms)
        file.write("\n")
        file.write("\n")
        file.write("\n")
        file.write('@njit\n')
        file.write(the_source)
