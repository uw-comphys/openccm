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

import inspect
from typing import List, Optional, Tuple

import numpy as np
import pyparsing as pp
import sympy as sp


def generate_reaction_system(config_parser: 'ConfigParser', _ddt_reshape_shape: Optional[Tuple[int, int, int]]) -> None:
    """
    Main function that handles support for systems of chemical reactions. In order, this function:
        (1) Performs an initial parsing of the reactions configuration file based on the two main headers, [REACTIONS] and [RATES].
        (2) Parses reactions (and associated rates) in the same species order as they appear specified in the main configuration file.
        (3) Creates a runtime-generated function containing the differential reaction system of equations for each specie which contributes to overall mass balance.

    Args:
        config_parser:      OpenCCM ConfigParser which contains configuration file information and location (i.e. relative path).
        _ddt_reshape_shape: Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction occurring at it. Used by `create_reaction_code`
    """
    input_file    = config_parser.get_item(['SIMULATION', 'reactions_file_path'], str)
    rxn_species   = config_parser.get_list(['SIMULATION', 'specie_names'],        str)
    rxn_file_path = config_parser.get_item(['SETUP', 'working_directory'],        str) + '/reaction_code_gen.py'

    # Create dummy file if no reactions specified
    if input_file == 'None':
        with open(rxn_file_path, 'w') as file:
            file.write("from numba import njit\n\n\n")
            file.write("@njit\n")
            file.write("def reactions(C, _ddt):\n")
            file.write("    return\n\n")
        return

    # If reactions are specified, read each line in reactions configuration file
    with open(input_file, 'r') as file:
        all_lines = file.readlines()
    
    # Get [REACTIONS] and [RATES] header locations (since this can be unordered)
    reactions_header = all_lines.index('[REACTIONS]\n')
    rates_header     = all_lines.index('[RATES]\n')

    # Do not assume that reactions come before rates (as provided in the examples)
    if reactions_header < rates_header:
        all_reactions = all_lines[reactions_header+1:rates_header]
    else: 
        all_reactions = all_lines[reactions_header+1:]
    # Strip any preceeding, middle, and trailing whitespaces
    all_reactions = [rxn.strip() for rxn in all_reactions if rxn.strip()]
        
    # Generate reaction system if applicable (i.e. if any reactions are given)
    if all_reactions:
        # Again handle ordering of reactions config file headers
        if reactions_header < rates_header:
            all_rates = all_lines[rates_header+1:]
        else: 
            all_rates = all_lines[rates_header+1:reactions_header]
        # Strip any preceeding, middle, and trailing whitespaces
        all_rates = [rate.strip() for rate in all_rates if rate.strip()]

        # Send all reaction and rate information into parser
        reaction_eqns = parse_reactions(rxn_species, all_reactions, all_rates)
    # Otherwise, run a tracer experiment (species present but no chemical reactions occur)
    else:
        reaction_eqns = []

    # Generate runtime function containing differential reaction system of equations for mass balance.
    create_reaction_code(rxn_species, reaction_eqns, rxn_file_path, _ddt_reshape_shape)


def parse_reactions(specie_order: List[str], all_reactions: List[str], all_rates: List[str]) -> List:
    """
    Parses all reactions and rates and creates the differential reaction system of equations.
    Also enforces specific ordering of system of equations based on species in specie_order (i.e. from main config file).
    The parser expects (and requires) a specific format for the reactions and in general it can parse the general reaction: aA + bB + [...] -> gG + hH + [...].
    An appropriate reactions configuration file for the reversible reaction 2a <-> 3b with k_f = 1e-2 and k_r = 3e-4 might be:

        [REACTIONS] 
        R1: 2a -> 3b
        R2: 3b -> 2a

        [RATES]
        R1: 1e-2
        R2: 3e-4

    See the Notes section below for further description of the expectations and limitations of this custom reactions configuration parser.

    Only forward WRITTEN reactions are supported (use of ->). I.e. reversible reactions should be written as 2 separate forward reactions. Thus use of <-> is not supported.
    Molecules with standard chemical superscripts (i.e. ions) and subscripts (i.e. compounds) are not supported by this parser. I.e., if you want to write 5O2 (i.e. 5 dioxygen molecules), you can use 5A instead, where A (simulation) = O2 (reality).

    Args:
        specie_order:       The ordered species as written in the main configuration file. Must match the same types of species present in reactions config file.
        all_reactions:      The partially-parsed reactions as extracted from reactions configuration file.
        all_rates:          The partially-parsed reaction rates as extracted from reactions configuration file.

    Returns:
        reaction_eqns:      List containing the system of differential equations in the same order of species as they appear in specie_order. I.e. if specie_order = [a, b] --> reaction_eqns = [da/dt, db/dt].
    
    Notes:
        The reaction parser only supports forward (written) reactions (i.e. use of ->). Reversible reactions must be written as two independent reactions with their associated rate constants (see example above).
        Furthermore, all chemical species must be written as letter characters (and not alphanumeric). I.e., standard chemical species with superscripts (ions) or subscripts (compounds) are not supported.
        For example, if a reaction contains 5O2 (5 dioxygen molecules), it must be written as 5A or 5OTWO in the reactions config (i.e. no trailing numbers or underscores).
    """ 
    # Parse the reaction ids (i.e. R1, R2 etc) from the actual reactions and associated rates. These are needed for proper input checking.
    rate_ids = [rate.split(':')[0].strip() for rate in all_rates]
    rxn_ids = [rxn.split(':')[0].strip() for rxn in all_reactions]
    rxn_rxns = [rxn.split(':')[1].replace(" ", "") for rxn in all_reactions]

    # This section enforces proper input of reactions configuration file.
    # 1. Duplicate reaction or rate labels are present (i.e., 2 or more definitions for a reaction or rate expression). These definitions must be unique
    if (len(rxn_ids) != np.unique(rxn_ids).size) or (len(rate_ids) != np.unique(rate_ids).size):
        raise RuntimeError("Duplicate reaction and/or rate labels detected. Check reactions configuration file.")
    # 2. If the number of unique reaction IDs does not equal rate IDs, then there are missing reactions or rates
    elif np.unique(rxn_ids).size != np.unique(rate_ids).size:
        raise RuntimeError("Number of reactions does not match number of rates (or their reaction IDs are different). Check reactions configuration file.")
    # 3. Catches duplicate reactions listed in the reactions config file
    elif len(rxn_rxns) != np.unique(rxn_rxns).size:
        raise RuntimeError("Duplicate reactions detected. Check reactions configuration file.")
    # 4. Each reaction does not have a related rate
    elif sorted(rxn_ids) != sorted(rate_ids):
        raise RuntimeError("Cannot find rate for one or more reactions. Check that reaction and rate IDs align.")
    # If no issues, delete the reaction ids and temp reactions as they are no longer needed
    else:
        del rxn_ids, rxn_rxns

    # Prepare rate constants by separating from reaction IDs (i.e. 'R1: 5' --> '5')
    rate_vals = [rate.split(':')[1].strip() for rate in all_rates]

    # Create rate dictionary with appropriate reaction IDs
    rate_book = dict(zip((rate_ids), (rate_vals)))

    # Get dictionary for reactions with rates which will complete the parsing of reactions config file
    rxn_book = {} 
    for _, rxn in enumerate(all_reactions):
        curr_key = rxn.split(':')[0]
        rxn_book.setdefault(curr_key, [])
        
        # Append actual reaction to book
        rxn_book[curr_key].append(rxn.split(':')[1].strip())

        # Check #5 - rate constant must be a number. Note that since we also allow 'AeB' format (A x 10^B) then we need a try/except since isnumeric() and isdigit() will not work.
        try: 
            float(rate_book.get(curr_key))
        except Exception as e:
            raise ValueError(f"{e}. Please check that rate constants are numeric in reactions config.")
        # If rate constant is numeric, then append to reaction book
        rxn_book[curr_key].append(rate_book.get(curr_key))

    # The following pyparsing structure outlines the expected reaction structure from reactions config file.
    rxnType = pp.ZeroOrMore(pp.Literal('->')) 
    molecule_with_coeff = pp.ZeroOrMore(pp.Word(pp.nums, '.'+pp.nums))
    all_molecules = molecule_with_coeff + pp.Combine(pp.Word(pp.alphas)) + pp.ZeroOrMore(pp.Suppress('+') + molecule_with_coeff + pp.Combine(pp.Word(pp.alphas))) 
    reactantsExpr = all_molecules + pp.Suppress(rxnType) # automatically suppresses remainder of text
    productsExpr = pp.Suppress(reactantsExpr) + all_molecules 

    # Initialize lhs and rhs system of equation arrays
    DEsysLHS = np.array([]) # left of equal sign, the dA/dt, dB/dt, etc. (which will be written as dA, dB, etc) 
    DEsysRHS = np.array([]) # right of equal sign, the rate expression based on chemical kinetic theory    
    
    # Iterate through rxn_book and get system of equations
    for _, (rxn_id, rxn_info) in enumerate(rxn_book.items()): 

        # Parse the current reaction split by '->', i.e., separate reactants and products.
        reactantString = reactantsExpr.parseString(rxn_info[0]) # reactants
        productString = productsExpr.parseString(rxn_info[0]) # products

        # If the first reactant has a stoich. coefficient of 1 (but implicit in reactions config file), make this explicit.
        if reactantString[0].isalpha():
            reactantString.insert(0, '1')
        # If the first product has a stoich. coefficient of 1 (but implicit in reactions config file), make this explicit.
        if productString[0].isalpha():
            productString.insert(0, '1')
        
        # If any subsequent reactants have stoich. coefficient of 1 (but implicit in reactions config file), make this explicit.
        reactant_bool = np.asarray([r.isalpha() for r in reactantString])
        indices = np.where(reactant_bool == True)[0]
        for ind in indices:
            if reactantString[int(ind-1)].isalpha():
                reactantString.insert(ind, '1')
                # account for shift in original indice locations due to recent insert
                indices += 1 
        
        # If any subsequent products have stoich. coefficient of 1 (but implicit in reactions config file), make this explicit.
        product_bool = np.asarray([r.isalpha() for r in productString])
        indices = np.where(product_bool == True)[0]
        for ind in indices:
            if productString[int(ind-1)].isalpha():
                productString.insert(ind, '1')
                # account for shift in original indice locations due to recent insert
                indices += 1

        # Create both the lhs and rhs of differential equation reaction system for the reactants:
        counter = 0
        while counter < len(reactantString[0::2]):
            DEsysLHS = np.append(DEsysLHS, 'd' + reactantString[1::2][counter])
            DEsysRHS = np.append(DEsysRHS, '-' + reactantString[0::2][counter] + f'*{rxn_info[1]}*' + '*'.join([i + '**' + j for i, j in zip(reactantString[1::2], reactantString[0::2]) ]))
            counter += 1

        # Create both the lhs and rhs of differential equation reaction system for the products:
        counter = 0
        while counter < len(productString[0::2]):
            DEsysLHS = np.append(DEsysLHS, 'd' + productString[1::2][counter])
            DEsysRHS = np.append(DEsysRHS, productString[0::2][counter] + f'*{rxn_info[1]}*' + '*'.join([i + '**' + j for i, j in zip(reactantString[1::2], reactantString[0::2]) ]))
            counter += 1

    # Combine duplicate entries (i.e. 2 statements for dA, should be added to get total differential equation).
    # Duplicates occur because the parser digests reactions independently, thus duplicates arise because a species may be present in more than one reaction.
    checked = []
    tempLHS = np.array([])
    tempRHS = np.array([])
    for currMolec in DEsysLHS:
        if currMolec not in checked:
            dupIndices = [i for i, x in enumerate(DEsysLHS) if x == currMolec]
            tempLHS = np.append(tempLHS, currMolec)
            tempRHS = np.append(tempRHS, '+'.join(DEsysRHS[dupIndices]))
            checked.append(currMolec)

    # Until this point, the LHS was written as dA (i.e. for dA/dt). shorten this notation further such that dA --> A
    DEsysLHS = [LHS[1:] for LHS in tempLHS]
    DEsysRHS = tempRHS.tolist()

    # Check #6 - check if the species listed in main config file EXACTLY matches that found in reactions config file 
    # note that use of sorted() does not modify lists in place.
    if sorted(specie_order) != sorted(DEsysLHS):
        raise ValueError(f"The unique species found in the main config file does not match the unique species found after parsing the reactions. Ensure these two files contain the same species.")
    
    # Enforce order of differential equations in same species appearance as given in main configuration file
    # i.e. if a,b given in main config, then should return [da/dt, db/dt] etc.
    reaction_eqns = DEsysRHS
    if specie_order != DEsysLHS:
        reaction_eqns = []
        for _, specie in enumerate(specie_order):
            where_specie = DEsysLHS.index(specie)
            reaction_eqns.append(DEsysRHS[where_specie])
    
    return reaction_eqns


def create_reaction_code(rxn_species: List[str], reaction_eqns: List[str], rxn_file_path: str, _ddt_reshape_shape: Optional[Tuple[int, int, int]] = None) -> None:
    """
    Symbolically creates and simplifies the rhs of the system of differential equations for reaction environment using sympy.
    Then, this rhs is translated into source code and written into a separate .py file at runtime.

    Args:
        rxn_species:        Name of species in the order that they appear in the main config file.
        reaction_eqns:      Output of `parse_reactions` which gives system of differential reaction equations appearing in the same specie order as rxn_species.
        rxn_file_path:      The path to the file in which to save the generated reactions.
        _ddt_reshape_shape: Shape needed by _ddt for PFR systems so that the inlet node does not have a reaction occurring at it.
    """
    # If an empty list was passed for reaction_eqns, it means no reactions are involved in simulation (i.e. tracer experiment). 
    # Therefore take species given and add dummy *0 so that the reaction terms are not contributing to mass balance as desired.
    if not reaction_eqns:
        reaction_eqns = [specie + '*0' for specie in rxn_species]

    # Convert string ODE system to sympy expression type ODE system
    rhs_symb = [sp.parse_expr(eqn) for eqn in reaction_eqns]

    # Convert into function and then to string (i.e. source code)
    the_source = inspect.getsource(sp.lambdify([rxn_species], rhs_symb, 'numpy'))

    # Change function name to reactions
    the_source = the_source.replace("_lambdifygenerated","reactions")

    # Change Dummy variable name
    dummy_name = the_source[the_source.find('(')+1:the_source.find(')')]
    the_source = the_source.replace(dummy_name, 'specie_concentrations')

    # Add _ddt parameter
    close_bracket = the_source.find(')')
    source_left, source_right = the_source[:close_bracket], the_source[close_bracket:]
    if _ddt_reshape_shape:
        source_right = source_right.replace('specie_concentrations', f'specie_concentrations.reshape({_ddt_reshape_shape})')
    the_source = source_left + ', _ddt' + source_right

    # Remove the unneeded return statement
    the_source = the_source[0:the_source.find('return')].strip() + '\n\n'

    # Create a view into _ddt that we can easily work with
    # (n_species, n_pfr * points_per_pfr) -> (n_species, n_pfr, points_per_pfr)
    if _ddt_reshape_shape:  # Only for PFRs
        the_source +=  '    # Reshape concentrations and _ddt to apply reaction only to non-inlet nodes\n'
        the_source += f'    _ddt = _ddt.reshape({_ddt_reshape_shape})\n'
        the_source += '    \n'

    # Unroll _ddt update for each specie
    rhs_strs = [str(eqn) for eqn in rhs_symb]
    if _ddt_reshape_shape:
        for specie in rxn_species:
            for i, rhs in enumerate(rhs_strs):
                rhs_strs[i] = rhs.replace(specie, specie + '[..., 1:]')

    for id_specie, rhs in enumerate(rhs_strs):
        the_source += f'    # Update for specie {rxn_species[id_specie]}\n'
        if _ddt_reshape_shape:
            the_source += f'    _ddt[{id_specie}, :, 1:] += {rhs} \n\n'  # RXN does not apply to inlet node.
        else:
            the_source += f'    _ddt[{id_specie}, :] += {rhs} \n\n'

    # Write system of differential reaction equations to runtime-generated file
    with open(rxn_file_path, 'w') as file:
        file.write("from numba import njit\n")
        file.write("import numpy as np\n")
        file.write("\n")
        file.write("\n")
        file.write('@njit\n')
        file.write(the_source)
