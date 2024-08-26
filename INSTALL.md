# Installation Instructions

openccm is available on all operating systems, but is predominantly tested on Linux and macOS.
Please ensure that Python 3.10+ is installed on your machine before installing openccm.

It's recommended that most people install openccm through pip using

`pip install openccm`.

If you wish to use all of the optional functionality, e.g. network visualization or plot generation,
or wish to use it with [OpenCMP](https://opencmp.io), the command becomes

`pip install 'openccm[all]'`

Users who wish to install openccm from source must:
1. Download or clone the source from the [github repo](https://github.com/uw-comphys/openccm).
2. Open a terminal and navigate into the downloaded folder.
3. Run `pip install -e .` for the minimal install
4. Or `pip install -e '.[all]'` for all dependencies.