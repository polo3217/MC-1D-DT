import numpy as np
from typing import Optional, Tuple

from src.geometry_classes import Geometry, Material
from src.source_class import Source, SourceRegion


# =============================================================================
# Log-normal source parameter helper
# =============================================================================

def _lognormal_params(E_mean_eV: float) -> Tuple[float, float]:
    """
    Compute (mu, sigma) in log(eV) space for a log-normal energy distribution
    whose mode sits at E_mean_eV.

    For a log-normal:   mode = exp(mu - sigma²)
    We fix sigma = 1.0 (moderate spread) and solve:
        mu = ln(E_mean_eV) + sigma²

    The hard upper cutoff E_max is enforced by the rejection loop already
    inside Source._sample_energy via energy_range.
    """
    sigma = 1.0
    mu    = np.log(E_mean_eV) + sigma ** 2
    return mu, sigma


def _get_E_max_source(geometry) -> float:
    """
    Return min(nuclide.E_max) across all nuclides loaded in the geometry.
    Must be called after set_maj_xs_method() so nuclide objects are populated.
    """
    if not geometry._nuclides:
        raise RuntimeError(
            "_get_E_max_source: geometry._nuclides is empty. "
            "Ensure set_maj_xs_method() has been called first."
        )

    return min(nuclide_obj.E_max for nuclide_obj, density in geometry._nuclides.values())


E_MIN_SOURCE = 0.1  # eV — lower bound (below thermal, negligible flux)


# =============================================================================
# Physical constants
# =============================================================================
NA = 6.02214076e23  # atoms/mol


def _atom_density(rho: float, M: float) -> float:
    """Return atom number density [atoms/cm³] for a pure element / compound."""
    return rho * NA / M


# =============================================================================
# Benchmark dimensions  (all in cm)
# =============================================================================
# 17x17A PWR assembly — USNRC ML1806 report
# 1 in = 2.54 cm

PITCH_CM        = 0.496  * 2.54        # 1.26 cm  — rod pitch
CLAD_OD_CM      = 0.360  * 2.54        # 0.9144 cm  — clad outer diameter (neglected)
PELLET_DIAM_CM  = 0.3088 * 2.54        # 0.78475 cm — fuel pellet diameter

N_RODS          = 17                    # rods in one row

# Assembly total width
H_CM  = N_RODS * PITCH_CM              # 17 * 1.26 = 21.42 cm

# Equivalent 1-D slab widths derived from area conservation
# W_f : fuel plate width  = 17*pi*d_f^2 / (4*H)
W_FUEL_CM  = N_RODS * np.pi * PELLET_DIAM_CM**2 / (4.0 * H_CM)

# W_w : water inter-slab width  = (H - 17*W_f) / 18
#   (there are 18 water gaps: before rod 1, between each pair, after rod 17)
W_WATER_CM = (H_CM - N_RODS * W_FUEL_CM) / (N_RODS + 1)

# Sub-slab structure
N_FUEL_SUBSLAB  = 5   # equal-width subdivisions per fuel plate
N_WATER_SUBSLAB = 3   # equal-width subdivisions per water gap

W_FUEL_SUB  = W_FUEL_CM  / N_FUEL_SUBSLAB
W_WATER_SUB = W_WATER_CM / N_WATER_SUBSLAB

# Fuel sub-slab temperatures (symmetric, 5 zones: boundary→center→boundary)
T_FUEL_SUBS = [650.0, 900.0, 1250.0, 900.0, 650.0]  # K

# Water sub-slab temperatures (symmetric, 3 zones: contact→center→contact)
T_WATER_SUBS = [600.0, 580.0, 600.0]  # K


# =============================================================================
# Material helpers
# =============================================================================

def _make_fuel_material(rod_index: int, sub_index: int,
                         enrichment_wt: float, T: float) -> Material:
    """
    Build a UO2 fuel Material for rod `rod_index` (0-based), sub-slab
    `sub_index` (0-based), at temperature T.

    Enrichment is given as weight-percent U-235.
    Fuel: UO2, density ≈ 10.4 g/cm³
    """
    rho_UO2 = 10.4          # g/cm³  (typical sintered UO2)
    M_U235  = 235.0439299   # g/mol
    M_U238  = 238.05078826  # g/mol
    M_O     = 15.9994       # g/mol

    # Weight fractions of U-235 and U-238 in uranium
    w5 = enrichment_wt / 100.0          # wt fraction U-235 in total
    w8 = 1.0 - w5                       # wt fraction U-238 in total

    # Molar mass of uranium mixture
    M_U = 1.0 / (w5 / M_U235 + w8 / M_U238)

    # Molar mass of UO2
    M_UO2 = M_U + 2.0 * M_O

    # Total molecule density  [molecules/cm³]
    N_mol = rho_UO2 * NA / M_UO2

    # Atom densities
    N_U235 = N_mol * w5 * M_UO2 / M_U235
    N_U238 = N_mol * w8 * M_UO2 / M_U238
    N_O    = 2.0 * N_mol

    return Material(
        name     = f"fuel_rod{rod_index+1:02d}_sub{sub_index+1}",
        nuclides = [('U235', N_U235), ('U238', N_U238), ('O16', N_O)],
        T        = T,
    )


def _make_water_material(rod_gap_index: int, sub_index: int, T: float) -> Material:
    """
    Build a light-water (H2O) Material for water gap `rod_gap_index` (0-based),
    sub-slab `sub_index` (0-based), at temperature T.

    Water density is temperature-dependent; simple approximation used here.
    """
    # Very simple linear fit for liquid water density vs T (K), 580–600 K
    # At ~590 K, ~155 bar (PWR), density ≈ 0.71 g/cm³
    rho_water = 0.71   # g/cm³  (compressed water at ~590 K, ~155 bar)

    M_H  = 1.00794    # g/mol
    M_O  = 15.9994    # g/mol
    M_H2O = 2 * M_H + M_O

    N_mol = rho_water * NA / M_H2O
    N_H   = 2.0 * N_mol
    N_O   = 1.0 * N_mol

    return Material(
        name     = f"water_gap{rod_gap_index:02d}_sub{sub_index+1}",
        nuclides = [('H1', N_H), ('O16', N_O)],
        T        = T,
    )


# =============================================================================
# Geometry constructor
# =============================================================================

def create_geometry_PWR_fresh_fuel(
    maj_mat_method: str,
    maj_xs_method:  str,
    access_method:  str,
    mode:           str,
    xs_maj_file_dir: Optional[str] = None,
    T_array: Optional[np.ndarray] = None,
    neutron_nbr: int = 10_000,
) -> Tuple[Geometry, Source]:
    """
    1-D representation of a fresh-fuel 17×17 PWR assembly.

    Layout (left → right):
        [water gap 0] [fuel rod 1] [water gap 1] ... [fuel rod 17] [water gap 17]

    Each fuel plate  → 5 equal sub-slabs at T = [650, 900, 1250, 900, 650] K
    Each water gap   → 3 equal sub-slabs at T = [600, 580, 600] K

    Enrichment of rod i (1-based): i wt%  (1 % … 17 %)

    Source: homogeneous in all fuel sub-slabs, log-normal energy distribution
    truncated at E_max = 20 keV (WMP upper limit).  Each fuel sub-slab carries
    equal weight so the spatial sampling is uniform across all fuel material.

    Returns
    -------
    geometry : Geometry
    source   : Source
    """
    if T_array is None:
        T_array = np.array([293.6, 500.0, 600.0, 650.0,
                            900.0, 1000.0, 1250.0, 1500.0, 2000.0])  # K

    # ------------------------------------------------------------------
    # Build Geometry object
    # ------------------------------------------------------------------
    geometry = Geometry(majorant_log=False, poll_interval=0.001, verbose=False)
    geometry.xs_dir = '/home/paule/open_mc_projects/xs_lib/endfb-vii.1-hdf5/wmp'

    # ------------------------------------------------------------------
    # Build all materials and register regions
    # ------------------------------------------------------------------
    x_cursor = 0.0        # running x-position [cm]
    # One entry per rod: (region_name_of_first_subslab, x_min_rod, x_max_rod)
    # Used to build one SourceRegion per rod (not per sub-slab).
    fuel_rod_extents: list[tuple[str, float, float]] = []
    fuel_subslab_names: list[str] = []

    for rod in range(N_RODS):
        enrichment = float(rod + 1)   # 1 % … 17 %  wt U-235

        # ── Leading water gap ─────────────────────────────────────────
        for ws in range(N_WATER_SUBSLAB):
            T_w = T_WATER_SUBS[ws]
            mat = _make_water_material(rod_gap_index=rod, sub_index=ws, T=T_w)
            geometry.add_material(mat)
            geometry.add_region(
                name     = mat.name,
                material = mat,
                x_min    = x_cursor,
                x_max    = x_cursor + W_WATER_SUB,
            )
            x_cursor += W_WATER_SUB

        # ── Fuel plate ────────────────────────────────────────────────
        x_rod_min = x_cursor
        first_sub_name = None
        for fs in range(N_FUEL_SUBSLAB):
            T_f = T_FUEL_SUBS[fs]
            mat = _make_fuel_material(rod_index=rod, sub_index=fs,
                                      enrichment_wt=enrichment, T=T_f)
            geometry.add_material(mat)
            geometry.add_region(
                name     = mat.name,
                material = mat,
                x_min    = x_cursor,
                x_max    = x_cursor + W_FUEL_SUB,
            )
            if first_sub_name is None:
                first_sub_name = mat.name
            x_cursor += W_FUEL_SUB
            fuel_subslab_names.append(mat.name)
        x_rod_max = x_cursor

        # Store (anchor_name, x_rod_min, x_rod_max) for source construction.
        # The anchor_name is a real registered region used only for Source
        # validation; position sampling uses the full rod extent via override.
        fuel_rod_extents.append((first_sub_name, x_rod_min, x_rod_max))

    # ── Trailing water gap (gap index = N_RODS) ──────────────────────
    for ws in range(N_WATER_SUBSLAB):
        T_w = T_WATER_SUBS[ws]
        mat = _make_water_material(rod_gap_index=N_RODS, sub_index=ws, T=T_w)
        geometry.add_material(mat)
        geometry.add_region(
            name     = mat.name,
            material = mat,
            x_min    = x_cursor,
            x_max    = x_cursor + W_WATER_SUB,
        )
        x_cursor += W_WATER_SUB

    # Sanity-check: x_cursor should equal H_CM
    assert abs(x_cursor - H_CM) < 1e-8, (
        f"Geometry width mismatch: got {x_cursor:.6f} cm, expected {H_CM:.6f} cm"
    )

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    geometry.set_boundary_conditions("reflective", "reflective")

    # ------------------------------------------------------------------
    # Tallies
    # ------------------------------------------------------------------
    energy_bins = [1e-3, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e7]  # eV

    geometry.attach_flux_tally(
        boundaries= [0, H_CM/4, H_CM/2, 3*H_CM/4, H_CM],
        energy_bins= energy_bins
    )

    # Verification tally surfaces: fuel/water interfaces only.
    # 2 surfaces per rod (water→fuel and fuel→water) + 2 assembly boundaries
    # = 36 surfaces total.  No sub-slab internal boundaries are included.
    verification_surfaces = [0.0]
    x_tmp = 0.0
    for rod in range(N_RODS):
        x_tmp += W_WATER_CM
        verification_surfaces.append(x_tmp)   # water → fuel
        x_tmp += W_FUEL_CM
        verification_surfaces.append(x_tmp)   # fuel  → water
    verification_surfaces.append(H_CM)

    # Pass verification_surfaces as both the spatial bin boundaries (for
    # reaction rates) and the surface positions (for currents).
    # This gives 35 meaningful spatial bins — one per fuel/water zone —
    # instead of one per sub-slab.
    geometry.attach_verification_tally(
        energy_bins = energy_bins,
        surface_xs  = [0, H_CM/4, H_CM/2, 3*H_CM/4, H_CM],
        boundaries= [0, H_CM/4, H_CM/2, 3*H_CM/4, H_CM]
    )

    # ------------------------------------------------------------------
    # Cutoff energy
    # ------------------------------------------------------------------
    geometry.set_cutoff_energy(70.0)  # eV

    # ------------------------------------------------------------------
    # Majorant / XS / access / mode settings
    # ------------------------------------------------------------------
    geometry.maj_mat_method = maj_mat_method
    geometry.set_maj_xs_method(
        method          = maj_xs_method,
        T_array         = T_array,
        xs_maj_file_dir = xs_maj_file_dir,
    )

    # Resolve the WMP upper energy limit from the loaded nuclide data.
    # This single value drives both:
    #   - set_access_method  (last_energy: upper bound for reconr grid / access switch)
    #   - Source             (energy_range upper bound)
    # Using min(nuclide.E_max) guarantees consistency across all nuclides.
    E_max_wmp = _get_E_max_source(geometry)
    print(f"[geometry] E_max_wmp = {E_max_wmp:.2f} eV  "
          f"(min nuclide.E_max — used for access method and source)")

    geometry.set_access_method(
        access_method,
        last_energy = E_max_wmp,   # was hardcoded 600 eV — now driven by nuclide data
        err_lim     = 0.001,
        err_max     = 0.01,
    )
    geometry.set_mode(mode, filename='validation/xs_generation/statepoint.200.h5')

    # ------------------------------------------------------------------
    # Source  — one SourceRegion per fuel sub-slab (17 rods × 5 = 85)
    # ------------------------------------------------------------------
    source_mu, source_sigma = _lognormal_params(E_mean_eV=500.0)
    print(f"[source]   log-normal: mu={source_mu:.4f}, sigma={source_sigma:.4f}, "
          f"mode={np.exp(source_mu - source_sigma**2):.1f} eV, "
          f"E_range=[{E_MIN_SOURCE:.1f}, {E_max_wmp:.2f}] eV")
 
    source_regions = [
        SourceRegion(
            region_name    = name,
            weight         = 1.0,
            energy_dist    = "log_normal",
            energy_range   = (E_MIN_SOURCE, E_max_wmp),
            mu             = source_mu,
            sigma          = source_sigma,
            direction_dist = "isotropic",
        )
        for name in fuel_subslab_names
    ]
 
    source = Source(
        neutron_nbr    = neutron_nbr,
        geometry       = geometry,
        source_regions = source_regions,
    )
 
    print(f"[source]   {len(source_regions)} source regions ({N_RODS} rods × {N_FUEL_SUBSLAB} sub-slabs)")
 
    return geometry, source


# =============================================================================
# Quick geometry report (optional, called when run as a script)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PWR Fresh-Fuel Benchmark — Geometry & Source Summary")
    print("=" * 60)
    print(f"  Rod pitch              : {PITCH_CM:.4f} cm")
    print(f"  Pellet diameter        : {PELLET_DIAM_CM:.4f} cm")
    print(f"  Assembly width H       : {H_CM:.4f} cm")
    print(f"  Fuel slab width  W_f   : {W_FUEL_CM:.4f} cm")
    print(f"  Water gap width  W_w   : {W_WATER_CM:.4f} cm")
    print(f"  Fuel sub-slab width    : {W_FUEL_SUB:.4f} cm  (x{N_FUEL_SUBSLAB})")
    print(f"  Water sub-slab width   : {W_WATER_SUB:.4f} cm  (x{N_WATER_SUBSLAB})")
    print(f"  Total region count     : "
          f"{N_RODS * (N_FUEL_SUBSLAB + N_WATER_SUBSLAB) + N_WATER_SUBSLAB}")
    print(f"  Fuel regions (source)  : {N_RODS * N_FUEL_SUBSLAB}")
    print(f"  Geometry spans         : 0.0 → {H_CM:.4f} cm")
    print()
    print("  Fuel sub-slab temperatures  (K) :", T_FUEL_SUBS)
    print("  Water sub-slab temperatures (K) :", T_WATER_SUBS)
    print()
    print("  Enrichments (wt% U-235) :")
    for i in range(N_RODS):
        print(f"    Rod {i+1:2d} : {i+1:2d} %")
    print()
    print("  Source parameters:")
    print(f"    energy_dist  : log_normal")
    print(f"    E_min        : {E_MIN_SOURCE:.1f} eV")
    print(f"    E_max        : min(nuclide.E_max)  — resolved at runtime after XS loading")
    print(f"    mu           : ln(500 eV) + sigma²  — mode fixed at 500 eV")
    print(f"    sigma        : 1.0")
    print(f"    spatial      : uniform across all {N_RODS * N_FUEL_SUBSLAB} fuel sub-slabs")
    print()
    print("  Usage:")
    print("    geometry, source = create_geometry_PWR_fresh_fuel(...)")
    print("    source.run_source(geometry)")