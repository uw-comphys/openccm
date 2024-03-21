---
title: 'OpenCCM: An Open-Source Continuous Compartment Modeling Package'
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

OpenCCM is a compartmental modelling software package based on flow alignment [@Vasile2024]. It is primarily intended for flow-based processes where there is a weak coupling between any reactions and the flow in the system. OpenCCM 
integrates with both OpenCMP [@Monte2022] and OpenFOAM [@OpenFOAM] allowing for easy movement of data between this package and either of those. Additionally, it provides users with built-in functionality for calculating residence times, exporting to re-import into simulation software, and for exporting results for visualization using ParaView [@Ahrens2005].

OpenCCM development follows the principles of ease of use, performance, and extensibility. The configuration file-based user interface is intended to be concise, readable, and intuitive. Furthermore, the code base is structured and documented [@Vasile2024] and uses an internal mesh representation such that experienced users can add input-output bindings for their packages (e.g. FEniCS or ANSYS) with no modifications required to the main compartmental modelling code.

Compartmental modelling allow for significant computational speedups compared to full CFD simulations while still providing some information about the spatial variation within the domain. However, traditional approaches suffer from lack of generality, too much generality, lack of physical interpretability of results,
OpenCCM addresses these issues by providing a fully automated compartmental modelling method which handles the general class of flow problems through the use of a flow-based compartmentaliztion scheme and a higher-order PFR-in-series-based model for each compartment.
Further, the constructed model retains a one-to-one correspondence between locations within the compartment model and the original physical domain as well as built-in methods for mapping the results back onto the physical space, for both visualization purposes and for further simulations, i.e. allowing for multi-scale modelling.
rather than a well-mixed CSTR-based model.

# Statement of Need

While simulation-based design and analysis continues to provide valuable insights and revolutionize the engineering design process, many processes, especially in chemical engineering, are inherently challenging to simulate. For example, many processes in the pharmaceutical industry, bioreactors and crystalizers, contain very disparate time-scales necessitating simulations be performed with very small time-steps over very long time horizons. This imposes a significant computation cost which hampers or even limits the ability to use simulations for entire classes of problems. Compartment modelling is a form of reduced order modelling which is especially well-suited for such applications as they take advantage of those disparate time-scales in order to greatly simplify the system in the spatial domain; replacing the full CFD-domain by a network of compartments, each with a simplified set of governing equations.

However, there are several barrier to the more widespread us of compartment models. The biggest of which is the lack of software, open-source or otherwise, for automatically generating compartment models. There exists closed-source, AMBER [@Amber], for manually creating and solving well-mixed compartments networks, however it is cost-prohibitive to use and the lack of automatice compartmentalization severely limits its applicability for the use-cases of interest, i.e. engineering design and geometry iterations. There also exist open-source software, Cantera [@Cantera], for solving compartment networks, however it lacks the ability to build such a network given a simulation result. Further, neither of these software allow for importing CFD results and creating compartments based on those results, instead relying on expert knowledge to manually create and connect the compartments.
The only alternative up until now, as evidenced by the compartment modelling literature, has been to manually construct the compartments and manually write out the coupled ODE/PDEs representing the mass balances over the compartments.

The goal of OpenCCM is to fill this need for an open-source compartment modelling package which is user-friendly, compatible with a variety of simulation package backends, and which fits into the users existing simulation and post-processing pipeline. OpenCCM is built using the idea of a intermediary mesh and flow-field format allowing for the compartment modelling algorithm and spatial mapping to be simulation package agnostic, allowing users to import simulation results from their package of choice and exporting the results back into that same format. Currently support for OpenCMP and OpenFOAM is included. OpenCCM provides pre-implemented finite-difference based solvers for the resulting models and a configuration file-based user interface to allow the general simulation community to immediately take advantage of the benefits of such models, not just simulation experts. The user interface is designed to be intuitive, readable, and requires no programming experience - solely knowledge of the CLI. Users must choose between the CSTR- and PFR-based models, certain tolerances, and any reactions occurring the system but need no experience with the actual numerical implementation of the models or the mathematical derivation of net reaction rates.


# Features

| Feature                 | Description                                                                       |
|-------------------------|-----------------------------------------------------------------------------------|
| Model support           | Accepts OpenCMP [@Monte2021] and OpenFOAM [@OpenFOAM] results                     |
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

Writing unit tests against an exact output is very difficult for a variety of reasons. First the method is inherently an approximation of the CFD simulation leading to expected differences between the original CFD result and the approximation. Further, while the method is automated it is nevertheless quite descriptive, providing criteria that the final result will meet and is sensitive to initial conditions, though the differing final results all have comparable performance. Thus, the majority of the software testing is provided through the use of extensive use of in-line asserts checking and enforcing of invariants and required properties at the beginning and end of most functions. The reaction system is tested against several analytic solutions, see the examples sections for more information.

# User Interface

The `OpenCCM` python package can be used via text-based configuration files centered around the CLI (command line interface). The software was designed such that each run would be in its own self-contained directory. This can either be a new directory created for the compartment model. In addition to the `OpenCCM` config files, the required contents of the directory depends on the simulation package being used. The package currently supports two open source simulation packages: `OpenCMP` and `OpenFOAM`.

For `OpenCMP` three files are required: 1) the `OpenCMP` config file, 2) the mesh on which the simulation was run, and 3) the .sol solution file that contains the velocity profile to use for creating the compartment model. The `OpenCCM` config file will specify the path of the `OpenCMP` config file and the `.sol` file, the location of the mesh file will be read from the `OpenCMP` config file.

For `OpenFOAM`, two directories are required: 1) the `constant` directory which contains the mesh information in ASCII format, and 2) a directory containing the simulation results to be used for creating the compartment model saved in ASCII format. The path to the solution directory is specified in the `OpenCCM` config, and the `constant` directory is assumed to be in the same parent folder.

After running, `OpenCCM` will create several directories:

The `.log` directory which contains detailed debugging information, if debug logging is enabled, about each step of the compartment modelling process.

The `.tmp` directory which contains intermediary files used for speeding up subsequent runs of the model. This includes the mesh and velocity vector fields converted to the intermediary format as well as the network of PFRs/CSTRs.

The `ouput_ccm` directory which contains `ParaView` files for visualizing the compartments as well as the simulation results from the compartment model, both in numpy format for further analysis and in either ParaView format or the native format of the simulation package that was originally used.

# Examples of Usage

Several examples are provided, which for the time being are all in 2D. These examples show the handling of both OpenCMP and OpenFOAM simulation results, as well as creating both PFR-based and CSTR-based compartmental models.
The geometry used in [@Vasile2024] is reproduced in both OpenCMP and OpenFOAM under the `OpenCMP/pipe_with_recirc_2d` and `OpenFOAM/pipe_with_recic` directories, respectively.

Also included in the `examples/simple_reactors` folder are two the files needed to run OpenCMP-simulation based single CSTR and single PFR models for various reaction systems.
These two example act as tutorials for how to imput reactions and a convergence/error analysis for linear, non-linear, coupled, and reversible reaction systems in CSTRs and/or PFRs.

# Acknowledgements

This research was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Digital Research Alliance of Canada.

# References
