from shutil import rmtree
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import pickle

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


import ngsolve as ngs
import pyngcore as ngcore
from netgen.read_gmsh import ReadGmsh
from ngsolve import GridFunction, Mesh, Integrate

from openccm.config_functions import ConfigParser
from openccm.io import fes_from_opencmp_config
from openccm import run

fig_width_pt = 390.0 * 1.7  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = 2.0 / (np.sqrt(5) + 1.0)  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {'axes.labelsize': 18,
          'font.size': 18,
          'legend.fontsize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

linewidth = 4

xlim = [0, 300]


def sort_folders_by_name(folders: List[Path], all_int: bool = False) -> List[Path]:
    """
    Helper function to deal with sorting folders by name in numerical order when they are floats
    """
    # If running on a mac, need to make sure to delete and .DS_Store files
    i = 0
    while i < len(folders):
        folder = folders[i]
        if folder.name == '.DS_Store':
            folders.pop(i)
        else:
            if all_int:
                assert folder.name.isalnum()
            i += 1
    if all_int:
        folders_info = [(int(folder.name), folder) for folder in folders]
    else:
        folders_info = [(float(folder.name.split("_")[1][:-4]), folder) for folder in folders]
    folders_info.sort(key=lambda tup: tup[0])

    return [folder_info[1] for folder_info in folders_info]


def rtd_cfd() -> np.ndarray:
    """
    Function to calculate E(t) and F(t) for the CFD tracer experiments without converting to .vtu.
    """
    # Get all the results folders
    folder = Path(".")

    # Set parameters for ngsolve
    ngcore.SetNumThreads(4)

    # Run the model.
    with (ngcore.TaskManager()):
        # Get data info from file
        msh_files = [file for file in folder.glob('*.msh')]
        assert len(msh_files) == 1
        config_files = [file for file in folder.glob('config_MCINS')]
        assert len(config_files) == 1
        config_file_path = str(config_files[0])

        # Load mesh and create grid function
        data_path               = Path(str(folder.absolute()) + "/cfd_rtd.npy")
        if data_path.exists():
            rtd_cfd = np.load(str(data_path.absolute()))
        else:
            mesh = Mesh(ReadGmsh(str(msh_files[0])))
            fes = fes_from_opencmp_config(config_file_path, mesh)
            gfu = GridFunction(fes)
            u: GridFunction = gfu.components[0]  # Hardcoded
            tracer: GridFunction = gfu.components[2]  # Hardcoded

            # Get list of .sol files
            sol_folder = Path(str(folder.absolute()) + "/output/multicomponentins_sol/")
            sols = sort_folders_by_name([s for s in sol_folder.glob("*.sol")])

            # Load the velocity field
            velocity_folder = Path(str(folder.absolute()) + "/output/components_sol/")
            velocity_sol: List[Path] = [s for s in velocity_folder.glob("u.sol")]
            assert len(velocity_sol) == 1
            u.Load(str(velocity_sol[0]))

            # t, F(t), E(t)
            rtd_cfd  = np.zeros((len(sols), 3))\

            n = ngs.specialcf.normal(mesh.dim)

            for i, sol in enumerate(sols):
                tracer.Load(str(sol))
                # Load time
                rtd_cfd[i, 0] = float(sol.name.split("_")[1][:-4])
                # Calculate tracer flow out of the domain
                rtd_cfd[i, 2] = Integrate(cf=n*u*tracer,
                                       mesh=mesh,
                                       definedon=mesh.Boundaries("outlet"))

            # Note: Using the last time-step to avoid any ramping at the beginning.
            mass_flow_in = Integrate(cf=-n*u*tracer,
                                     mesh=mesh,
                                     definedon=mesh.Boundaries("inlet"))
            rtd_cfd[:, 2] /= mass_flow_in

            # Ensure that none of the data for the concentration is negative. Some very small negative values (1e-15)
            # Show up, but those are essentially ==0.
            # assert np.min(data[:, 2]) > -1e-10
            rtd_cfd[:, 2] = np.abs(rtd_cfd[:, 2])

            rtd_cfd[:, 1] = np.gradient(rtd_cfd[:, 2], rtd_cfd[:, 0], edge_order=2)

            # Because of numerical issues, np.gradient will sometimes return a small (e.g. -1e-40) negative number
            # when it should really return 0
            # assert np.min(data[:, 1]) > -1e-20
            rtd_cfd[:, 1] = np.abs(rtd_cfd[:, 1])

            # Save data to file
            np.save(str(data_path.absolute()), rtd_cfd)

    return rtd_cfd


def rtd_cm() -> Tuple[np.ndarray, np.ndarray]:
    data: Dict[str, np.ndarray] = dict()
    for model in ['cstr', 'pfr']:
        data_path = Path(f"./output_ccm/{model}_rtd.npy")
        if not data_path.exists():
            config_parser = ConfigParser("CONFIG")
            config_parser['COMPARTMENT MODELLING']['model'] = model
            if model == 'cstr':
                config_parser['SIMULATION']['points_per_pfr'] = '1'
            config_parser.update_paths()

            run(config_parser)

            if Path("./cache/").exists():
                rmtree("./cache/")
            if Path("./log/").exists():
                rmtree("./log/")

        data[model] = np.load(str(data_path.absolute())).squeeze()

    return data['pfr'], data['cstr']


def plot_results(data_cfd:      np.ndarray,
                 data_pfr:      np.ndarray,
                 data_cstr:     np.ndarray) -> None:
    # Calculate absolute max so that everything is on the same scale.
    e_max = 0
    f_max = 0

    e_max = max(e_max, np.max(data_cfd[:, 1]))
    f_max = max(f_max, np.max(data_cfd[:, 2]))

    e_max = max(e_max, np.max(data_pfr[:, 1]))
    f_max = max(f_max, np.max(data_pfr[:, 2]))

    e_max = max(e_max, np.max(data_cstr[:, 1]))
    f_max = max(f_max, np.max(data_cstr[:, 2]))

    e_max *= 1.1
    f_max *= 1.1

    e_min = -0.005

    # Plot CFD vs PFR vs CSTR
    plt.figure()
    cmap = plt.get_cmap("tab10")
    plt.plot(data_cfd[:, 0], data_cfd[:, 1], color=cmap(1), linewidth=linewidth)
    plt.plot(data_pfr[:, 0], data_pfr[:, 1], color=cmap(1),
             linestyle='--', linewidth=linewidth)
    plt.plot(data_cstr[:, 0], data_cstr[:, 1], color=cmap(1),
             linestyle=':', linewidth=linewidth)
    plt.legend(["CFD", "CM-PFR", "CM-CSTR"], loc="upper right")
    plt.xlim(xlim)
    plt.ylim([e_min, e_max])
    plt.axhline(y=0, color='grey', linestyle=':')
    plt.xlabel("Time")
    plt.ylabel("Tracer Concentration")
    Path('figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/CFD vs PFR vs CSTR.pdf')


if __name__ == "__main__":
    data_pfr, data_cstr  = rtd_cm()
    data_cfd             = rtd_cfd()

    plot_results(data_cfd, data_pfr, data_cstr)
