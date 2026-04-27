import sys
import os
import numpy as np

sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')
sys.path.append('/home/paule/open_mc_projects/xs_lib/endfb-vii.1-hdf5/neutron')
sys.path.append('/home/paule/open_mc_projects/MC-1D_DT')

""
from src.neutron_class import Neutron
from src.source_class import Source, _BatchSource
from src.geometry_classes import Geometry, Material
import src.geometry_classes as geom
import src.performance_classes as perf
import src.tally_classes as tally
import src.vectfit as vf
import src.export_simulation_v3 as xpsim
import src.export_print_csv as xpcsv
import src.reconr_v2 as reconr
import src.parallel as parallel

import openmc

openmc.config['cross_sections'] = "/home/paule/open_mc_projects/xs_lib/endfb-vii.1-hdf5/cross_sections.xml"

                            
def create_geometry_U8_slab(maj_mat_method, 
                            maj_xs_method, 
                            access_method, 
                            mode,
                            xs_maj_file_dir,
                            T_array=None
                             ) -> Geometry:



    # One slab of U238 — simple test case
    if T_array is None:
        T_array = np.array([293.6, 500, 1000, 1500, 2000])  # K

    # ============================================================================
    # Uranium atom density
    # ============================================================================
    # Uranium metal density ~ 19.1 g/cm³,
    # N = rho * Na / A  (atoms/cm³)
    rho_U   = 19.1               # g/cm³
    NA      = 6.02214076e23      # atoms/mol

    x_U235 = 0.00                # 0% enrichment
    x_U238 = 1.0 - x_U235

    M_U8   = 238.05078826        # g/mol
    M_U5   = 235.0439299         # g/mol
    N_total_U8 = rho_U * NA / (x_U238 * M_U8 + x_U235 * M_U5)
    N_total_U5 = rho_U * NA / (x_U235 * M_U5 + x_U238 * M_U8)

    N_U235 = 1.0 * N_total_U5   # 0%  enrichment
    N_U238 = 1.0 * N_total_U8   # 100%

    slab1 = Material(
        name     = "cell 1",
        nuclides = [('U238', N_U238)],
        T        = 293.6,    # K (~20 °C)
    )

    slab2 = Material(
        name     = "cell 2",
        nuclides = [('U238', N_U238)],
        T        = 2000,     # K
    )





    # ============================================================================
    # Geometry  (two slabs)
    # ============================================================================
    geom = Geometry(majorant_log=True, poll_interval=0.001, verbose=False)
    geom.xs_dir = '/home/paule/open_mc_projects/xs_lib/endfb-vii.1-hdf5/wmp'

    geom.add_material(slab1)
    geom.add_material(slab2)
    geom.boundaries     = [0.0, 2.0, 15.0]  # cm
    geom.material_array = [slab1, slab2]

    # Flux tally: 5 energy groups
    energy_bins = [10.0, 600.0, 1e4, 1e6, 2e7]  # eV
    geom.attach_flux_tally(energy_bins)

    # Verification tally: same energy bins, surface at slab1/slab2 interface
    geom.attach_verification_tally(
        energy_bins = energy_bins,
        surface_xs  = [2.0],        # slab1/slab2 interface
    )

    # Majorant and XS method
    geom.maj_mat_method = maj_mat_method # majorant method: "maj_mat" 
    geom.set_maj_xs_method(method=maj_xs_method, T_array=T_array, xs_maj_file_dir=xs_maj_file_dir)  # majorant XS method: "vectfit" 
    geom.set_access_method(access_method, last_energy= 600, err_lim=0.0001, err_max=0.001)  # access method: "fly" or "simple"
    geom.set_mode(mode, filename='validation/xs_generation/statepoint.200.h5')
    
    return geom



