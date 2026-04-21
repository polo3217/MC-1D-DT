import openmc
import numpy as np
from openmc.data import K_BOLTZMANN

def find_energy_range(nuclide, E, T_min, T_max, Q):
    E_min = (np.sqrt(E) - Q * np.sqrt(K_BOLTZMANN * (T_max-T_min))/nuclide.sqrtAWR)**2
    E_max = (np.sqrt(E) + Q* np.sqrt(K_BOLTZMANN * (T_max-T_min))/nuclide.sqrtAWR)**2

    return E_min, E_max


def find_majorant_xs_rothenstein(nuclide, E, Q=2, T_min=293, T_max=3000, n_points_per_window=100):
    E_min, E_max = find_energy_range(nuclide, E, T_min, T_max, Q)

    Energies = np.linspace(E_min, E_max, n_points_per_window)
    maj_sigm_tot_E = - np.inf
    for i, e in enumerate(Energies):
        nuclide_xs = nuclide(e, 293)
        sigm_tot_E = nuclide_xs[0] + nuclide_xs[1]
        maj_sigm_tot_E = np.max([maj_sigm_tot_E, sigm_tot_E])

    return maj_sigm_tot_E


    
    
