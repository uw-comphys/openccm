import pickle

import numpy as np
import matplotlib.pyplot as plt

from openccm import ConfigParser, run

if __name__ == '__main__':
    config_parser = ConfigParser('CONFIG')
    run(config_parser)

    i_s = 0
    i_e = -1

    model               = config_parser.get_item(['COMPARTMENT MODELLING', 'model'], str).lower()
    points_per_model    = 1 if model == 'cstr' else config_parser.get_item(['SIMULATION', 'points_per_pfr'], int)
    species             = config_parser.get_list(['SIMULATION', 'specie_names'],     str)

    with (open(f"cache/{model}_network.pickle", 'rb') as handle):
        model_network = pickle.load(handle)
    volumes_model = model_network[1]
    if points_per_model == 1:
        volume_at_dof = volumes_model[:, np.newaxis]
    else:
        volume_at_dof = np.linspace([0] * len(volumes_model), volumes_model, points_per_model).transpose()

    t: np.ndarray = np.load(f'output_ccm/{model}_t.npy')
    c: np.ndarray = np.load(f'output_ccm/{model}_concentrations.npy')
    c_mean = np.mean(c, axis=1)
    c_min  = np.min(c, axis=1)
    c_max  = np.max(c, axis=1)
    c_tot = c.sum(axis=0)
    c_mean_tot = np.mean(c_tot, axis=0)
    c_min_tot  = np.min(c_tot, axis=0)
    c_max_tot  = np.max(c_tot, axis=0)
    s = c.shape
    c = c.reshape((s[0], len(volumes_model), points_per_model, s[-1]))
    _c_mean = c_mean.copy()
    _c_mean[_c_mean == 0] = 1e-35
    s_cm = _c_mean.shape
    _c_mean = _c_mean.reshape((s[0], 1, 1, s[-1]))
    norm_std = ((c - _c_mean) / _c_mean)**2

    mass = np.empty((s[0], s[-1]))
    com = np.empty_like(mass)
    for i_specie in range(s[0]):
        for i_t in range(len(t)):
            if points_per_model == 1:
                mass[i_specie, i_t] = np.sum(c[i_specie, :, :, i_t]*volume_at_dof)
                com [i_specie, i_t] = np.sum(norm_std[i_specie, :, :, i_t]*volume_at_dof)
            else:
                mass[i_specie, i_t] = np.trapz(c[i_specie, :, :, i_t],          volume_at_dof, axis=1).sum(axis=0)
                com [i_specie, i_t] = np.trapz(norm_std[i_specie, :, :, i_t],   volume_at_dof, axis=1).sum(axis=0)
    com / sum(volumes_model)

    colors = ['blue', 'green']
    plt.figure()
    plt.title('CoM')
    for i, specie in enumerate(species):
        plt.plot(t[i_s:i_e], com[i, i_s:i_e], label=specie, color=colors[i])
    plt.xlabel('Time [s]')
    plt.ylabel('CoM [-]')
    plt.semilogy()
    plt.legend()
    # plt.ylim([0, 0.04])
    plt.show()

    plt.figure()
    plt.title('Mass')
    for i, specie in enumerate(species):
        plt.plot(t[i_s:i_e], mass[i, i_s:i_e], label=specie, color=colors[i])
    plt.plot(t[i_s:i_e], mass.sum(axis=0)[i_s:i_e], label='Total Mass', color='red')
    plt.hlines(1.0, t[i_s], t[i_e], colors='grey', linestyles='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Total Tracer Mass [-]')
    plt.legend()
    plt.show()

    # dmdt = np.gradient(mass, t, edge_order=2)
    #
    # plt.figure()
    # plt.title('Derivative')
    # plt.plot(t[i_s:i_e], dmdt[i_s:i_e], label='Total mass')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Derivative of Total Tracer Mass [-/s]')
    # # plt.semilogy()
    # plt.show()

    plt.figure()
    plt.title('Concentrations')
    for i, specie in enumerate(species):
        plt.plot(t[i_s:i_e], c_mean[i, i_s:i_e], label=specie, color=colors[i])
        plt.fill_between(t[i_s:i_e], c_min[i, i_s:i_e], c_max[i, i_s:i_e], color=colors[i], alpha=0.2)
    plt.plot(t[i_s:i_e], c_mean_tot[i_s:i_e], label='Total Concentration', color='red')
    plt.fill_between(t[i_s:i_e], c_min_tot[i_s:i_e], c_max_tot[i_s:i_e], color='red', alpha=0.2)
    # plt.hlines(0.5, t[i_s], t[i_e], colors='grey', linestyles='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Tracer Concentration [-/V]')
    # plt.ylim([0.88, 0.95])
    plt.legend()
    plt.show()
