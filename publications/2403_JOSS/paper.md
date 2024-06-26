---
title: '`OpenCCM`: An Open-Source Continuous Compartmental Modelling Package'
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

`OpenCCM` is a compartmental modelling [@Jourdan2019] software package which is based on recently-developed fully automated flow alignment compartmentalization methods [@Vasile2024]. It is primarily intended for large-scale flow-based processes where there is weak coupling between composition changes, e.g. through (bio)chemical reactions, and convective mass transport in the system. Compartmental modelling is an important approach used to developed reduced-order models [@Chinesta2017] [@Benner2020] using a priori knowledge of process hydrodynamics [@Jourdan2019]. Compartmental modelling methods, such as those implemented in `OpenCCM`, enable simulations of these processes to be performed with far less computational complexity while still capturing the key aspects of their performance.

`OpenCCM` integrates with two multiphysics simulation software packages, `OpenCMP` [@Monte2022] and `OpenFOAM` [@OpenFOAM], allowing for ease of transferring simulation data for compartmentalization. Additionally, it provides users with built-in functionality for calculating residence times, exporting to transfer data to simulation software, and for exporting results for visualization using `ParaView` [@Paraview]. Post-processing methods are included for mapping simulation results from compartment domains to the original simulation domain, useful for both visualization purposes and for further simulations in other software (e.g. multi-scale modelling).


# Statement of Need

Simulation-based design and analysis continues to be widely applied in research and development of physicochemical processes.
Processes with large differences in characteristic time- and length-scale result in the infeasibility of direct multiphysics simulations due to computational limitations. This imposes a significant computation costs which severely reduces the utility of these simulations for entire classes of processes. Compartmental modelling is well-suited for such applications because it produces reduced-order models which are orders-of-magnitude less computationally demanding than direct simulation by taking advantage of the weak coupling between the short and long time-scale phenomena.

However, there are several barriers to the more widespread use of compartmental models. The largest of these is the lack of software for automatically generating compartmental models. There exists closed-source software, specifically `AMBER` [@Amber], for manually creating and solving well-mixed compartment networks.
However, this software is cost-prohibitive for much of the research community and lacks automated compartmentalization. There also exists open-source software, `Cantera` [@Cantera], for solving compartment networks. However, it does not incorporate flow information, when available, either from direct observation or multiphysics simulation. Furthermore, neither of these software allow for usage and direct transfer of flow information from multiphysics simulations, such as computational fluid dynamics (CFD) simulations, which are typically feasible over short times-scales.

The overall aim of `OpenCCM` is to fill this need for an open-source compartmental modelling package which is user-friendly, compatible with a variety of simulation package back-ends (e.g. `OpenFOAM` and `OpenCMP`), and which fits into the users existing simulation and post-processing software toolchain, i.e. `ParaView`.


# Features

| Feature                 | Description                                                                         |
|-------------------------|-------------------------------------------------------------------------------------|
| Model support           | Accepts `OpenCMP` [@Monte2022] and `OpenFOAM` [@OpenFOAM] results                   |
| Compartmentalization    | Single-phase flow-based compartment identification                                  |
| Compartmental Modelling | PFR-in-series-based model                                                           |
|                         | Previous SotA CSTR-based models                                                     |
| CM Simulations          | Linear, non-linear, and reversible arbitrary reactions.                             |
|                         | 1st Order upwinding finite-difference-based                                         |
|                         | Adaptive time-stepping                                                              |
| Post-Processing         | Residence time distribution                                                         |
| Output                  | Intermediary mesh format                                                            |
|                         | Labeled compartments in `Paraview` format                                           |
|                         | Concentrations from CM simulations in both `Paraview` and simulation package format |
| Performance             | Multi-threading                                                                     |
|                         | Caching of intermediary results to speed-up subsequent runs                         |


# User Interface

The `OpenCCM` Python package can be used via text-based configuration files centered around the CLI (command line interface) where each simulation run/project is in a self-contained directory. In addition to the `OpenCCM` configuration files, the required contents include flow information from one of two open source simulation packages: ``OpenCMP`` and ``OpenFOAM``. For ``OpenCMP`` three files are required:

1) the ``OpenCMP`` config file,
2) the mesh on which the simulation was run, and
3) the .sol solution file that contains the velocity profile to use for creating the compartmental model.

For ``OpenFOAM``, two directories are required:

1) the `constant` directory which contains the mesh information in ASCII format,
2) a directory containing the simulation results to be used for creating the compartmental model saved in ASCII format.

The path to the solution directory is specified in the `OpenCCM` config, and the `constant` directory is assumed to be in the same parent folder. The `OpenCCM` software will create several output directories: `log/` which contains detailed debugging information (if enabled), `cache/` which contains intermediary files, and `output_ccm/` both simulation results of the compartmental model (in various user-specified formats) and `ParaView` files for visualization.

# Reaction Configuration File

The chemical reaction parser in `OpenCCM` reads and parses the reactions configuration files and can handle general reaction equations of the form,

`aA + bB + [...] -> cC + dD + [...]`

with associated numeric rate constants. It intentionally does not support the standard `<->` symbol for reversible chemical reactions, so that each independent reaction has an explicit rate constant clearly defined in the same file. Therefore, a reversible reaction must be written as two independent forward reactions (with separate rate constants). Each specie *label* must solely contain letter characters, e.g. `O` instead of `O2` for oxygen. Kinetic rate constants must be expressed as positive real numbers in standard or scientific notation. Additionally, each reaction/rate pair must also have a unique *identifier* (i.e. R1, R2). For example, take the reversible reaction,

$$2NaCl + CaCO3 <-> Na2CO3 + CaCl2$$

with `k_f = 5e-2` and `k_r = 2` the species must first be redefined in simple terms in agreement with the reactions parser, i.e. a = NaCl, b = CaCO3, c = Na2CO3, and d = CaCl2. A configuration file for this reversible reaction may then be:

    [REACTIONS]
    R1: 2a + b -> c + d
    R2: c + d -> 2a + b

    [RATES]
    R1: 5e-2
    R2: 2

where **R1** and **R2** are the reaction *identifiers* for the forward and reverse reactions respectively.

# Examples of Usage

Several examples are provided in the `OpenCCM` documentation which demonstrate the usage of both `OpenCMP` and `OpenFOAM` simulation flow information for compartmentalization. One example uses the geometry from [@Vasile2024] and shows how to execute the needed CFD simulation for flow information (both using `OpenCMP` and `OpenFOAM`), create/visualize the compartmental model results, and compare the predicted RTD  to the reference result directly from CFD simulation.

For this illustrative example, the steady-state hydrodynamic flow-profile is obtained by running the `OpenCMP` simulation through the `run_OpenCMP.py` script in the folder. The resulting flow profile was opened in `ParaView` and the line integral convolution of the velocity field is shown below, colored by velocity magnitude.

![Visualization of hydrodynamics from CFD simulation with line integral convolutions indicating local flow direction and color corresponding to velocity magnitude.](images/lic_domain.png){ width=98% }

The underlying velocity field data is then processed using `OpenCCM` to produce a network of compartments by executing the `run_compartment.py` script. The figure below shows each element of the original mesh colored according to the compartment it belongs to.

![Visualization of flow-informed compartmentalization with coloring corresponding to compartment number.](images/labelled_compartments.png){ width=98% }

That network of compartments is further processed as each compartment is represented by a series of plug flow reactors (PFRs). The resulting network (graph) of PFRs is shown in the figure below; nodes are the centers of the PFRs and edges are connections (flows) between PFRs.

![Undirectly graph of the compartment network resulting from both (i) flow-information compartmentalization and (ii) the use of spatially-varying compartment approximations (PFRs).](images/compartment_network.pdf){ width=98% }

## RTD Curves

The Residence Time Distribution (RTD) curve for both the CFD and Compartmental Model (CM) are calculated using the script in the supplementary material of [@Vasile2024].

![Residence time distribution curves for both CFD and CM simulations.](images/e(t)_for_cfd_vs_pfr.pdf){ width=60% }

## Reactions

Finally, to demonstrate how to use the reaction system we will implement the reversible reaction system mentioned above:

$$2NaCl + CaCO3 <-> Na2CO3 + CaCl2$$

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

To output the `ParaView` visualizations, change `output_VTK` to `True` in the `CONFIG` file and re-run the simulation.

# Acknowledgements

This research was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Digital Research Alliance of Canada.

# References
