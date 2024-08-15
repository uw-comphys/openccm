from shutil import rmtree
from typing import List, Tuple
from pathlib import Path

import pickle

import numpy as np
import matplotlib.pyplot as plt

from openccm.config_functions import ConfigParser
from openccm.io import read_mesh_data
from openccm import run
from openccm.mesh import CMesh

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

    # Run the model.
    data_path = Path(str(folder.absolute()) + "/cfd_rtd.npy")
    if data_path.exists():
        rtd_cfd = np.load(str(data_path.absolute()))
    else:
        with open("./cache/cmesh.pickle", 'rb') as handle:
            c_mesh: CMesh = pickle.load(handle)
        flow_indices = list(c_mesh.bc_to_facet_map['outlet'])
        conc_indices = [c_mesh.facet_elements[facet][0] for facet in flow_indices]

        # Get subset of flows that we need
        flows: np.ndarray = np.load("./cache/flows_and_upwind.npy", allow_pickle=True)[:, 0]
        flows_outlet = flows[flow_indices]

        timestep_folders = []
        for folder in folder.iterdir():
            if folder.is_dir():
                try:
                    float(folder.name)
                    timestep_folders.append(folder.name)
                except:
                    pass
        timestep_folders.sort(key=float)
        # t, F(t), E(t)
        rtd_cfd  = np.zeros((len(timestep_folders), 3))

        for i, folder in enumerate(timestep_folders):
            if i == 0:  # Skip first timestep since it's the IC for the simulation and has zero on the outlet
                continue
            rtd_cfd[i, 0] = float(folder)
            # Calculate tracer flow out of the domain
            conc = read_mesh_data(f"./{folder}/T", float)
            rtd_cfd[i, 2] = flows_outlet @ conc[conc_indices]

        # Note: Using the last time-step to avoid any ramping at the beginning.
        flow_indices_inlet = list(c_mesh.bc_to_facet_map['inlet'])
        conc_indices_inlet = [c_mesh.facet_elements[facet][0] for facet in flow_indices_inlet]
        rtd_cfd[:, 2] /= flows[flow_indices_inlet] @ conc[conc_indices_inlet]

        # Ensure that none of the data for the concentration is negative. Some very small negative values (1e-15)
        # Show up, but those are essentially ==0.
        rtd_cfd[:, 2] = np.abs(rtd_cfd[:, 2])

        rtd_cfd[:, 1] = np.gradient(rtd_cfd[:, 2], rtd_cfd[:, 0], edge_order=2)

        # Because of numerical issues, np.gradient will sometimes return a small (e.g. -1e-40) negative number
        # which is really zero
        rtd_cfd[:, 1] = np.abs(rtd_cfd[:, 1])

        # Save data to file
        np.save(str(data_path.absolute()), rtd_cfd)

    return rtd_cfd


def rtd_cm() -> Tuple[np.ndarray, np.ndarray]:
    # Get all the results folders
    sim_folder = Path(".")

    cm_output_folder = Path(str(sim_folder) + "/output_ccm/")
    rtd_pfr_path     = Path(str(cm_output_folder) + "/pfr_rtd.npy")
    rtd_cstr_path    = Path(str(cm_output_folder) + "/cstr_rtd.npy")

    if not cm_output_folder.exists():
        working_directory = str(sim_folder) + "/"
        for i, model in enumerate(['cstr', 'pfr']):
            config_parser = ConfigParser("CONFIG")
            config_parser['SETUP']['working_directory'] = working_directory
            config_parser['COMPARTMENT MODELLING']['model'] = model
            if model == 'cstr':
                config_parser['SIMULATION']['points_per_pfr'] = '1'
            config_parser.update_paths()

            run(config_parser)
            if i == 0:
                if Path(working_directory + "cache/").exists():
                    rmtree(working_directory + "cache/")
                if Path(working_directory + "log/").exists():
                    rmtree(working_directory + "log/")

    if rtd_pfr_path.exists():
        data_pfr = np.load(str(rtd_pfr_path)).squeeze()
    else:
        raise FileNotFoundError("PFR data not found")

    if rtd_cstr_path.exists():
        data_cstr = np.load(str(rtd_cstr_path)).squeeze()
    else:
        raise FileNotFoundError("CSTR data not found")

    return data_pfr, data_cstr


def plot_results(data_cfd:      np.ndarray,
                 data_pfr:      np.ndarray,
                 data_cstr:     np.ndarray) -> None:
    # Calculate absolute max so that everything is on the same scale.
    e_max = 0

    e_max = max(e_max, np.max(data_cfd[:, 1]))
    e_max = max(e_max, np.max(data_pfr[:, 1]))
    e_max = max(e_max, np.max(data_cstr[:, 1]))
    e_max *= 1.1

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
    # plt.ylim([e_min, e_max])
    plt.axhline(y=0, color='grey', linestyle=':')
    plt.xlabel("Time")
    plt.ylabel("Tracer Concentration")
    Path('figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/CFD vs PFR vs CSTR.pdf')

if __name__ == "__main__":
    data_pfr, data_cstr  = rtd_cm()
    data_cfd             = rtd_cfd()

    plot_results(data_cfd, data_pfr, data_cstr)
