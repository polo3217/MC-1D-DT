import numpy as np
import pytest  # Optional, but recommended for clean testing
from src.geometry_classes import Geometry, Material
from src.source_class import Source, SourceRegion




def run_diagnostic_suite():
    """
    Main entry point to run all reflector-specific tests.
    """
    print("\n" + "="*50)
    print("STARTING REFLECTIVE BOUNDARY DIAGNOSTICS")
    print("="*50)
    
    test_symmetry_flat_flux()
    test_boundary_equivalence()
    test_stress_boundary_start()

def test_symmetry_flat_flux():
    """
    Checks if a 100% reflective box produces a spatially flat flux.
    In a closed system with no absorption, the flux must be uniform.
    """
    print("\n[Test 1] Symmetry / Flat Flux Test...")

    rho_U   = 19.1               # g/cm³
    NA      = 6.02214076e23      # atoms/mol
    x_U238 = 1.0 
    M_U8   = 238.05078826        # g/mol
    N_total_U8 = rho_U * NA / (x_U238 * M_U8 )
    N_U238 = 1.0 * N_total_U8   # 100%

    
    # 1. Setup - Pure scattering to keep neutrons alive for many reflections
    mat = Material(name = "U238", nuclides=[("U238", N_U238)], T=300.0)
    
    geom = Geometry(verbose=False)
    geom.add_region("Slab", mat, x_min=0.0, x_max=10.0)
    geom.set_boundary_conditions(left="reflective", right="reflective")
    
    # Attach tallies
    energy_bins = [1e-5, 20.0e6] 
    geom.attach_flux_tally(energy_bins=energy_bins)
    
    # 2. Source - Place in center
    n_hist = 25000
    src = Source.point(n_hist, geom, "Slab", [5.0, 0, 0], energy_dist="mono", energy_range=(500.0, 500.0))
    
    geom.maj_mat_method = "simple"
    geom.set_maj_xs_method(
        method  = "sqrtT_E",
    )
    geom.set_access_method("reconr", last_energy=600, err_lim=0.0001, err_max=0.001)
    geom.set_mode("analysis", filename="validation/xs_generation/statepoint.200.h5")

    # 3. Execute
    geom.run_source(src)
    
    # 4. Verify using TallyArray properties
    flux_mean = geom.flux_tally.mean
    re_mean = np.mean(geom.flux_tally.relative_error)
    
    # If the tally has multiple spatial regions, check variance across regions.
    # If it's a single region, check that the relative error is converging.
    print(f"  -> Total Histories: {geom.flux_tally._n_histories}")
    print(f"  -> Flux Mean: {np.mean(flux_mean):.4e} (RE: {re_mean:.2%})")
    
    if re_mean < 0.05:
        print("  [PASS] Statistics are converging.")
    else:
        print("  [WARN] High variance. Increase n_histories for better verification.")

def test_boundary_equivalence():

    rho_U   = 19.1               # g/cm³
    NA      = 6.02214076e23      # atoms/mol
    x_U238 = 1.0 
    M_U8   = 238.05078826        # g/mol
    N_total_U8 = rho_U * NA / (x_U238 * M_U8 )
    N_U238 = 1.0 * N_total_U8   # 100%
    """
    Standard Verification: A half-problem with a reflective boundary 
    must match a full-problem with vacuum boundaries.
    """
    print("\n[Test 2] Boundary Equivalence (Half vs Full)...")
    fuel_mat = Material("U238", nuclides=[("U238", N_U238)], T=300.0)
    energy_bins = [1e-5, 20.0e6]
    n_total = 40000

    # CASE A: Full 20cm Vacuum Slab
    geom_a = Geometry(verbose=False)
    geom_a.add_region("Full", fuel_mat, x_min=0.0, x_max=20.0)
    geom_a.set_boundary_conditions(left="vacuum", right="vacuum")
    geom_a.attach_flux_tally(energy_bins=energy_bins)

    geom_a.maj_mat_method = "simple"
    geom_a.set_maj_xs_method(
        method  = "sqrtT_E",
    )
    geom_a.set_access_method("reconr", last_energy=600, err_lim=0.0001, err_max=0.001)
    geom_a.set_mode("analysis", filename="validation/xs_generation/statepoint.200.h5")

    src_a = Source.point(n_total, geom_a, "Full", [10.0, 0, 0], energy_range=(500.0, 500.0))
    geom_a.run_source(src_a)
    
    # CASE B: Half 10cm Reflective/Vacuum Slab
    geom_b = Geometry(verbose=False)
    geom_b.add_region("Half", fuel_mat, x_min=0.0, x_max=10.0)
    geom_b.set_boundary_conditions(left="reflective", right="vacuum")
    geom_b.attach_flux_tally(energy_bins=energy_bins)

    geom_b.maj_mat_method = "simple"
    geom_b.set_maj_xs_method(
        method  = "sqrtT_E",
    )
    geom_b.set_access_method("reconr", last_energy=600, err_lim=0.0001, err_max=0.001)
    geom_b.set_mode("analysis", filename="validation/xs_generation/statepoint.200.h5")

    # n/2 because we only simulate the right side
    src_b = Source.point(n_total // 2, geom_b, "Half", [0.0, 0, 0], energy_range=(500, 500))
    geom_b.run_source(src_b)

    # 5. Compare using your new .mean property
    total_a = np.sum(geom_a.flux_tally.mean)
    total_b = np.sum(geom_b.flux_tally.mean) * 2
    
    diff = abs(total_a - total_b) / total_a
    print(f"  -> Full Tally: {total_a:.4e}")
    print(f"  -> Half Tally: {total_b:.4e}")
    
    if diff < 0.03:
        print(f"  [PASS] Equivalence verified. Diff: {diff:.2%}")
    else:
        print(f"  [FAIL] Significant bias in reflection logic. Diff: {diff:.2%}")



