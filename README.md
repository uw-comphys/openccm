# OpenCCM

TODO: DOI

[**Getting Started**](#getting-started)
| [**Issues**](#issues)
| [**Contribute**](#contribute)
| [**Citation**](#citation)
| [**Authors**](#authors)

## What is OpenCCM?

OpenCCM is a CFD-based compartment modelling software package.
It is primarily intended for convection dominated reactive flows which feature a weak or one-way coupling between
the reactive species and the carrier fluid, i.e. the reaction does not substantially influence the fluid flow over
the course of the simulation.

OpenCCM supports CFD results from OpenCMP and OpenFOAM, and has been written in such a way that support for other
simulation packages can be easily added.
It provides built-in post-processing for calculating residence time distributions and can output compartment model
simulation results in either ParaView format or the original simulation packages' native format
for use as input in subsequent CFD simulations.

Compartmental models have been researched in several chemical engineering fields, including:
* Bioreactors   ([J. Morchain (2024)](https://doi.org/10.1016/j.cherd.2024.04.014),
                [A. Delafosse (2010)](https://popups.uliege.be/1780-4507/index.php?id=6139),
                [P. Vrábel (1999)](https://www.sciencedirect.com/science/article/pii/S0263876299717892))
* Wastewater treatment facilities   ([M.C. Sadino-Riquelme (2023)](https://doi.org/10.1016/j.cej.2023.143180),
                                    [A. Alvarado (2012)](https://doi.org/10.1016/j.watres.2011.11.038),
                                    [Y. Le Moullec (2010)](https://doi.org/10.1016/j.ces.2009.06.035))
* Combustion engines    ([M. Savarese (2024)](https://doi.org/10.1016/j.ijhydene.2023.08.275),
                        [M.A. Agizza (2022)](https://doi.org/10.3390/en15010252),
                        [A. Innocenti (2018)](https://doi.org/10.1016/j.fuel.2017.11.097))
* Multiphase unit operations    ([Y. Du (2023)](https://doi.org/10.1016/j.ces.2023.118470),
                                [J. Darand (2022)](https://doi.org/10.1016/j.desal.2022.115743),
                                [E.K. Nauha (2015)](http://doi.org/10.1016/j.cej.2014.08.073))

The OpenCCM package provides reference code and documentation for the methods described in:
* [A. Vasile (2024)](https://doi.org/10.1016/j.compchemeng.2024.108650)

OpenCCM development follows the principles of ease of use, performance, and extensibility.
The configuration file-based user interface is intended to be concise, readable, and intuitive.
Similarly, the code base is structured such that experienced users can support for their simulation package of choice with minimal modifications to existing code.
OpenCCM comes with built-in support for performing reactive flow simulations with the compartment model.
Reactions are specified by the user on a per-reaction basis in the form for (A + B -> C + D) and are automatically parsed into ODEs.
Spatial discretization, if needed, is performed using the finite difference scheme.

## Getting Started

1. Ensure you have Python 3.10+ installed and then install OpenCCM using either pip (`pip install openccm`) or see the INSTALL.md file for complete installation instructions.
2. Several examples are provided inside the `examples/` folder for specifics of how to create the compartmental model, run simulations, and output visualizations.

If you plan to use the OpenCCM package for your own work, please cite appropriately using the [citation](#citation) below.

## Issues

If you encounter any **bugs** or **problems** with OpenCCM, please create a post using our package [issue tracker](https://github.com/uw-comphys/openccm/issues). Please provide a clear and concise description of the problem, with images or code-snippets where appropriate. We will do our best to address these problems as fast and efficiently as possible.

## Contribute

We welcome external contributions to the source code. This process will be easiest if users adhere to the contribution policy:

* Open an issue on the package [issue tracker](https://github.com/uw-comphys/openccm/issues) clearly describing your intentions on code modifications or additions
* Ensure your modifications or additions adhere to the existing standard of the OpenCCM package, specifically detailed documentation for new methods (see existing methods for example documentation)
* Test your modifications to ensure that the core functionality of the package has not been altered by running the unit tests.
* Once the issue has been discussed with a package author, you may open a pull request containing your modifications


## Citation

If you plan to use OpenCCm in your own work, please cite using the following Bibtex citation:
TODO: Fill in after having DOI
```Bibtex
@article{VasileOpenCCM202X,
    author  = {Vasile, Alexandru Andrei and Tino, Matthew Peres an Aseri, Yuvraj and Abukhdeir, Nasser Mohieddin},
    title   = {},
    doi     = {},
    journal = {Journal of Open Source Software},
    number  = {},
    pages   = {},
    volume  = {},
    year    = {202X},
    url     = {}
}
```

## Authors

* Alexandru Andrei Vasile
* Nasser Mohieddin Abukhdeir
* Matthew Peres Tino
* Yuvraj Aseri
