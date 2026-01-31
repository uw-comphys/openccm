import pickle
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile
from pathlib import Path

from openccm import ConfigParser, run
from openccm.mesh import CMesh


def elements_and_flow_for(bc_name: str, cmesh: CMesh, vel_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    facets = np.array(np.where(cmesh.facet_to_bc_map == cmesh.grouped_bcs.id(bc_name))[0], dtype=int)
    elements = np.empty(facets.shape, dtype=int)
    for i, facet in enumerate(facets):
        elements[i] = cmesh.facet_elements[facet][0]

    Q_4_element = np.empty(facets.shape)
    for i, facet in enumerate(facets):
        Q_4_element[i] = cmesh.facet_size[facet] * abs(vel_vec[elements[i]].dot(cmesh.facet_normals[facet]))

    return elements, Q_4_element


def read_in_concentrations(filename: str, mass_flowrate: np.ndarray, elements_to_read: np.ndarray, Q_4_elements: np.ndarray) -> None:
    """Return implicitly by storing data inside mass_flowrate."""
    with open('scalar_transport/' + filename + '/T', 'r') as file:
        lines = file.readlines()

    offset = lines.index('(\n') + 1
    for j, element in enumerate(elements_to_read):
        mass_flowrate[j] = float(lines[offset + element].strip())

    mass_flowrate *= Q_4_elements


if __name__ == '__main__':
    # Generate
    if not isfile('output_ccm/pfr_rtd.npy'):
        run('./CONFIG')

    cache_folder = 'cache'
    config_parser = ConfigParser('CONFIG')
    config_parser.update_paths()

    # Load mesh
    with open(cache_folder + '/cmesh.pickle', 'rb') as handle:
        cmesh: CMesh = pickle.load(handle)

    # Load velocity field
    vel_vec = np.load(cache_folder + '/vel_vec.npy')

    # Iterate over folder and read the outlet concentrations
    dir_folder = 'analysis'
    Path(dir_folder).mkdir(parents=True, exist_ok=True)
    if isfile(dir_folder + '/mass_out.npy') and isfile(dir_folder + '/mass_in.npy') and isfile(dir_folder + '/times_foam.npy'):
        print('Loading previously generated mass flowrates')
        mass_in = np.load(dir_folder + '/mass_in.npy')
        mass_out = np.load(dir_folder + '/mass_out.npy')
        times_foam    = np.load(dir_folder + '/times_foam.npy')
    else:
        dirs = []
        for _dir in listdir('scalar_transport'):
            try:
                float(_dir)
                dirs.append(_dir)
            except:
                pass
        dirs.sort(key=float)
        if float(dirs[0]) == 0:
            dirs.pop(0)
        elements_outlet, Q_4_outlet = elements_and_flow_for('outlet', cmesh, vel_vec)
        mass_out = np.empty((len(dirs)+1, len(elements_outlet)))
        times_foam = np.array((*[0], *(float(_dir) for _dir in dirs)))
        for i, _dir in enumerate(dirs):
            # Need a +1 since the first entry is kept zero for t=0.
            read_in_concentrations(_dir, mass_out[i+1, :], elements_outlet, Q_4_outlet)

        # Read the inlet concentration from the last file.
        elements_inlet, Q_4_inlet = elements_and_flow_for('inlet', cmesh, vel_vec)
        mass_in = np.empty(len(elements_inlet))
        read_in_concentrations(dirs[-1], mass_in, elements_inlet, Q_4_inlet)

        np.save(dir_folder + '/mass_in.npy', mass_in)
        np.save(dir_folder + '/mass_out.npy', mass_out)
        np.save(dir_folder + '/times_foam.npy', times_foam)

    # Calculate OpenFOAM RTD
    mass_out_total = np.sum(mass_out, axis=1)
    mass_in_total = np.sum(mass_in)
    cm_foam  = mass_out_total / mass_in_total
    rtd_foam = np.gradient(cm_foam, times_foam, edge_order=2)

    # Load OpenCCM RTD results
    data_ccm = np.load('output_ccm/pfr_rtd.npy').squeeze()
    times_ccm, rtd_ccm, cm_ccm = data_ccm[:, 0], data_ccm[:, 1], data_ccm[:, 2]

    t_max = 20

    i_foam = (times_foam >= t_max).argmax() + 1
    i_ccm  = (times_ccm  >= t_max).argmax() + 1

    # Plot OpenFOAM and CCM RTD and CM together
    plt.figure()
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.plot(times_foam[:i_foam], rtd_foam[:i_foam], label='OpenFOAM')
    plt.plot(times_ccm[:i_ccm], rtd_ccm[:i_ccm], label='OpenCCM')
    plt.title('Residence Time Distribution Function')
    plt.ylabel("Fraction of Mass per Unit Time [1/T]")

    plt.xlabel('Time [T]')
    plt.legend()
    plt.show()

    plt.figure()
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.axhline(y=1, color='grey', linestyle='--')
    plt.plot(times_foam[:i_foam], cm_foam[:i_foam], label='OpenFOAM')
    plt.plot(times_ccm[:i_ccm], cm_ccm[:i_ccm], label='OpenCCM')
    plt.title(f"Cumulative Distribution Function")
    plt.ylabel("Fraction of Mass [-]")
    plt.xlabel('Time [T]')
    plt.legend()
    plt.show()
