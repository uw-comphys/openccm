---
title: '`OpenCCM`: An Open-Source Continuous Compartment Modelling Package'
tags:
 - Python
 - compartmental modelling
 - reduced order modelling
 - model order reduction
 - computational fluid dynamics
authors:
 - name: Alexandru Andrei Vasile
   orcid: 0000-0002-0233-0172
   affiliation: 1
 - name: Matthew Peres Tino
   orcid: 0009-0005-6832-1761
   affiliation: 1
 - name: Yuvraj Aseri
   orcid: 0009-0008-4703-3711
   affiliation: 1
 - name: Nasser Mohieddin Abukhdeir^[Corresponding author]
   orcid: 0000-0002-1772-0376
   affiliation: "1, 2, 3" # Need quotes for multiple affiliations
affiliations:
 - name: Department of Chemical Engineering, University of Waterloo, Ontario, Canada
   index: 1
 - name: Department of Physics and Astronomy, University of Waterloo, Ontario, Canada
   index: 2
 - name: Waterloo Institute for Nanotechnology, University of Waterloo, Ontario, Canada
   index: 3
date:
bibliography: paper.bib
---

# Summary

`OpenCCM` is a compartmental modelling [@Jourdan2019] software package which is based on recently-developed fully automated flow alignment compartmentalization methods [@Vasile2024]. It is primarily intended for large-scale flow-based processes where there is weak coupling between composition changes, e.g. through (bio)chemical reactions, and convective mass transport in the system. Compartmental modelling is an important approach used to developed reduced-order models [@Chinesta2017][@Benner2020] using a priori knowledge
of process hydrodynamics [@Jourdan2019]. The computational cost of large-scale reacting flow problems, such as in industrial processes with stirred-tank (bio)chemical reactors, makes direct fully-coupled simulations infeasible. Compartmental modelling methods, such as those implemented in `OpenCCM`, enable simulations of these processes to be performed with far less computational complexity while still capturing the key aspects of their performance.

`OpenCCM` integrates with two multiphysics simulation software packages, `OpenCMP` [@Monte2022] and `OpenFOAM` [@OpenFOAM], allowing for ease of transferring simulation data for compartmentalization. Additionally, it provides users with built-in functionality for calculating residence times, exporting to transfer data to simulation software, and for exporting results for visualization using `ParaView` [@Paraview].

`OpenCCM` development follows the principles of ease of use, performance, and extensibility. The configuration file-based user interface is intended to be concise, readable, and intuitive. Furthermore, the code base is structured and documented [@Vasile2024] and uses an internal mesh representation such that experienced users can add input-output bindings for their packages (e.g. FEniCS or ANSYS) with no modifications required to the main compartmental modelling code.

While compartmental modelling is advantageous compared to direct simulations, traditional approaches suffer from either lack of generality, too much generality, or lack of physical interpretability of results.
`OpenCCM` addresses these issues by providing a fully automated flow-informed compartmental modelling method [@Vasile2024] which uses both flow-informed compartmentalization methods and high-order (spatial) models for each compartment.
Furthermore, constructed compartmental models retain one-to-one correspondence between spatial regions within the original simulation domain and compartments used in the reduced-order model.
Methods are included for mapping simulation results from compartment representations to the original simulation domain, useful for both visualization purposes and for further approximations (i.e. multi-scale modelling).


# Statement of Need

Simulation-based design and analysis continues to be widely applied in research and development of physicochemical processes.
Processes with large characteristic time- and length-scale result in the infeasibility of direct multiphysics simulations due to computational limitations. For example, many processes in the pharmaceutical industry, such as bioreactors and crystalizers, contain dynamic phenomena with disparate time-scales. Typically short timescale hydrodynamics and long time scale (bio)chemical reactions necessitate that simulations be performed with for long (reaction) times but with small (hydrodynamic) time integration steps. This imposes a significant computation costs which severely reduces the utility of these simulations for entire classes of processes. Compartment modelling is well-suited for such applications in that it decouples or weakly couples short and long time-scale phenomena which results in simulations requiring several orders-of-magnitude less computational resources compared to direct simulation.

However, there are several barriers to the more widespread use of compartment models. The largest of these barriers is the lack of open-source software for automated generation of compartment models. There exists closed-source software, specifically `AMBER` [@Amber], for manually creating and solving well-mixed compartments networks.
However, this software is cost-prohibitive for much of the research community and lacks automated compartmentalization.
These attributes severely limit its applicability for many researchers and use-cases, specifically engineering design and geometry iterations. There also exists open-source software, Cantera [@Cantera], for solving compartment networks. However it does not incorporate flow information, when available, either from direct observation or multiphysics simulation. Furthermore, neither of these software allow for usage and direct transfer of flow information from multiphysics simulations, such as computational fluid dynamics (CFD) simulations, which are typically feasible over short timescales. Flow-informed compartmentalization methods are specifically developed for this use case, where compartments and interconnections are created based on detailed flow information, instead relying on expert knowledge/insight and manual compartmentalization. The only option for the research community to date, supported by practice as evidenced by the compartment modelling literature
[@Jourdan2019], is to resort to manual compartmentalization and manual development of the coupled ODE/PDEs corresponding to mass and energy balances for each compartment.

The overall aim of `OpenCCM` is to fill this need for an open-source compartment modelling package which is user-friendly, compatible with a variety of simulation package back-ends (`OpenFOAM`, `OpenCMP`), and which fits into the users existing simulation and post-processing software toolchain. `OpenCCM` was developed using an intermediary mesh and flow-field format, allowing for the compartment modelling algorithm and spatial mapping to be simulation package agnostic. This enables users to import flow information from direct simulation packages of their choice and export compartmental model simulation results back into that native format.

`OpenCCM` provides pre-implemented finite-difference based solvers for the resulting models and a configuration file-based user interface to allow the general simulation community to immediately take advantage of the benefits of such models, not just simulation experts. The user interface is designed to be intuitive, readable, and requires no programming experience - solely knowledge of the command line interface (CLI). Users may choose between either well-mixed (CSTR) or spatially-varying (PFR) compartment models, and any reactions occurring the system but need no experience with the actual numerical implementation of the models or the mathematical derivation of net reaction rates.


# Features

| Feature                 | Description                                                                       |
|-------------------------|-----------------------------------------------------------------------------------|
| Model support           | Accepts `OpenCMP` [@Monte2021] and `OpenFOAM` [@`OpenFOAM`] results                     |
| Compartmentalization    | Single-phase flow-based compartment identification                                |
| Compartmental Modelling | PFR-in-series-based model                                                         |
|                         | Previous SotA CSTR-based models                                                   |
| CM Simulations          | Linear, non-linear, and reversible arbitrary reactions.                           |
|                         | 1st Order upwinding finite-difference-based                                       |
|                         | Adaptive time-stepping                                                            |
| Post-Processing         | Residence time distribution                                                       |
| Output                  | Intermediary mesh format                                                          |
|                         | Labeled compartments in Paraview format                                           |
|                         | Concentrations from CM simulations in both Paraview and simulation package format |
| Performance             | Multi-threading                                                                   |
|                         | Caching of intermediary results to speed-up subsequent runs                       |

Testing of the package is accomplished using a variety of methods. The core functions of the package all feature assert statements on their output to ensure that expected properties are met and invariants are preserved. The method identifies compartments such that they satisfy certain properties, mainly that they represent unidirectional flow, and so while it is deterministic, it is sensitive to the mesh and to the initial seeds. Thus, several different, but interchangeable, compartment networks can be produced from the same CFD simulation. Aside from trivial flow profiles and meshes, it is not feasible to predict the resulting compartmentalization to the degree needed for traditional unit testing. Instead, examples are included which compare quantitatively measurable predictions, e.g. residence time distribution, between the CFD and compartmental model. However, since this method is inherently a coarse-grained approximation of the CFD, approximate and not exact results are  expected.  

# User Interface

The `OpenCCM` Python package can be used via text-based configuration files centered around the CLI (command line interface). The software was designed such that each run would be in its own self-contained directory. This can either be a new directory created for the compartment model. In addition to the `OpenCCM` config files, the required contents of the directory depends on the simulation package being used. The package currently supports two open source simulation packages: ``OpenCMP`` and ``OpenFOAM``.

For ``OpenCMP`` three files are required:

1) the ``OpenCMP`` config file,
2) the mesh on which the simulation was run, and
3) the .sol solution file that contains the velocity profile to use for creating the compartment model.

The `OpenCCM` config file will specify the path of the ``OpenCMP`` config file and the `.sol` file, the location of the mesh file will be read from the ``OpenCMP`` config file.

For ``OpenFOAM``, two directories are required:

1) the `constant` directory which contains the mesh information in ASCII format,
2) a directory containing the simulation results to be used for creating the compartment model saved in ASCII format.

The path to the solution directory is specified in the `OpenCCM` config, and the `constant` directory is assumed to be in the same parent folder.

After running, `OpenCCM` will create several directories:

* `log` directory which contains detailed debugging information, if debug logging is enabled, about each step of the compartment modelling process.
* `cache` directory which contains intermediary files used for speeding up subsequent runs of the model. This includes the mesh and velocity vector fields converted to the intermediary format as well as the network of PFRs/CSTRs.
* `ouput_ccm` directory which contains ParaView files for visualizing the compartments as well as the simulation results from the compartment model, both in numpy format for further analysis and in either ParaView format or the native format of the simulation package that was originally used.

# Reaction Configuration File

The reactions parser developed for `OpenCCM` reads and parses the reactions configuration files and can handle general reaction equations of the form,

`aA + bB + [...] -> cC + dD + [...]`

with associated numeric rate constants. It intentionally does not support the standard `<->` symbol for reversible chemical reactions, so that each independent reaction has an explicit rate constant clearly defined in the same file. Therefore, a reversible reaction must be written as two independent forward reactions (with separate rate constants).

Additionally, specie superscripts (i.e. ions) or subscripts (i.e. compounds) in traditional chemistry notation are not supported by the parser. Instead, each specie must solely contain letter characters (except for stoichiometric coefficients which may precede these characters). For example., if the user wishes to use `O2`, it must be written as `O` or a dummy name such as `a` in the reactions configuration file.

The kinetic rate constants must be expressed as positive real numbers and the parser does support scientific notation for these values. Additionally, each reaction/rate pair must also have a unique *identifier*.
This identifier allows the parser to correctly associate a reaction/rate pair, and is allowed to be any alpanumeric value (i.e. R1, R2, etc.).

Additionally, the parser is robust such that it will flag any inappropriate use of the configuration setup. This includes:

* duplicate reactions,
* reactions with missing rates,
* rates with missing reactions,
* non-numeric rate values, and
* alphanumeric species (aside from stoichiometric prefixes)

The parser does not have a preference for the ordering of the configuration file (either [RATES] then [REACTIONS] or vice versa). Also, the specific reactions and rates themselves do not need to be in a specific order as long the identifiers are correct for each reaction/rate pair.

## Example Configuration

Suppose the reversible reaction,

    `2NaCl + CaCO3 <-> Na2CO3 + CaCl2`

with `k_f = 5e-2` and `k_r = 2` is used for simulations. These species must first be redefined in simple terms in agreement with the reactions parser, i.e. a = NaCl, b = CaCO3, c = Na2CO3, and d = CaCl2. A configuration file for this reversible reaction may then be:

    [REACTIONS]
    R1: 2a + b -> c + d
    R2: c + d -> 2a + b

    [RATES]
    R1: 5e-2
    R2: 2

where **R1** and **R2** are the reaction *identifiers* for the forward and reverse reactions respectively.

Multiple examples with different reactions have been developed for `OpenCCM`.

# Examples of Usage

Several examples are provided, which for the time being are all in 2D. These examples show the handling of both `OpenCMP` and `OpenFOAM` simulation results, as well as creating both PFR-based and CSTR-based compartmental models.
The geometry used in [@Vasile2024] is reproduced in both `OpenCMP` and `OpenFOAM` under the ``OpenCMP`/pipe_with_recirc_2d` and ``OpenFOAM`/pipe_with_recic` directories, respectively.

Also included in the `examples/simple_reactors` folder are the files needed to run `OpenCMP`-based single CSTR and single PFR models for various reaction systems.
These two examples act as tutorials for how to input reactions and a convergence/error analysis for linear, non-linear, coupled, and reversible reaction systems.

Below is an in-depth example, following along with [@Vasile2024], of the example in `examples/`OpenCMP`/pipe_with_recirc_2d` on how to run the CFD simulation, create and visualize the compartmental model, compare the RTD of the CFD and compartmental model, and finally demonstrate simulating a set of reactions on the compartment network.

## Hydrodynamics and Compartmental Model
The steady-state hydrodynamic flow-profile is obtained by running the `OpenCMP` simulation through the `run_`OpenCMP`.py` script in the folder. The resulting flow profile was opened in ParaView and the line integral convolution of the velocity field is shown below, colored by velocity magnitude.

![Surface LIC of CFD hydrodynamics](images/lic_domain.png){ width=98% }

The underlying velocity field data was then processed by OpenCCM to produce a network of compartments by using the `run_compartment.py` script. The figure below, again visualized with ParaView, shows each element of the original mesh colored according to the compartment it belongs to.

![Labelled Compartments.](images/labelled_compartments.png){ width=98% }

That network of compartments is further processed as each compartment is represented by a series of plug flow reactors (PFRs). The resulting network (graph) of PFRs is shown in the figure below; nodes are the centers of the PFRs and edges are connections (flows) between PFRs.

![Network of PFRs.](images/compartment_network.pdf){ width=98% }

## RTD Curves
The Residence Time Distribution (RTD) curve for both the CFD and Compartmental Model (CM) are calculated using the script in the supplementary material of [@Vasile2024].

![Residence time distribution curves between CFD and CM.](images/e(t)_for_cfd_vs_pfr.pdf){ width=60% }

## Reactions
Finally, to demonstrate how to use the reaction system we will implement the reversible reaction system mentioned above:

    `2NaCl + CaCO3 <-> Na2CO3 + CaCl2`

with `k_f = 5e-2` and `k_r = 2` with a = NaCl, b = CaCO3, c = Na2CO3, and d = CaCl2. The initial conditions are 0 for all species and the boundary conditions at the inlet are `[NaCl] = [CaCO3] = 1` and `[Na2CO3] = [CaCl2] = 0`. The equations and conditions have already been specified, enable the reactions by uncommenting the `;reactions_file_path = reactions` line by removing the ';' at the start of the line. Note that when you re-run the compartmentalization it will finish much faster than the first time, this is because the compartmental model does not have to be re-created, instead it is loaded from disk.

To analyze the results, the equilibrium values for this reversible system are calculated as follows:

$$ k_f [a]^2[b] = k_r [c][d] $$
$$ \frac{k_f}{k_r} = \frac{[c][d]}{[a]^2[b]} $$
$$ \frac{5 \times 10^{-2}}{2} = \frac{(x)(x)}{x(1-2x)^2} $$
$$ x \approx 0.1147 $$

where `x` is the number of moles produced of each product.

The expected equilibrium concentrations for the four species are: `[NaCl]_{ss} = 0.7706`, `[CaCO3] = 0.8853`, `[Na2CO3] = 0.1147`, and `[CaCl2] = 0.1147`. Based on the figures below, and from opening up the results, it can be seen that these steady state values are obtained at the outlet of the reactor.

![Input/Output Concentrations for 'a'.](images/system_response_a.pdf){ width=49% } ![Input/Output Concentrations for 'b'.](images/system_response_b.pdf){ width=49% }
![Input/Output Concentrations for 'c'.](images/system_response_c.pdf){ width=49% } ![Input/Output Concentrations for 'd'.](images/system_response_d.pdf){ width=49% }

To output the ParaView visualizations, change `output_VTK` to `True` in the `CONFIG` file and re-run the simulation.

# Acknowledgements

This research was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Digital Research Alliance of Canada.

# References
