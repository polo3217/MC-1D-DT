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


E_MIN_SOURCE = 70.0  # eV — cutoff energy (start of thermal region)


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
# ESFR-M1 fissile fuel assembly — Mikityuk et al.

N_PINS    = 271
N_CELLS   = 15

PIN_PITCH_CM  = 1.1670           # p  [cm]
CLAD_IN_CM    = 0.48623          # R_clad,in  [cm]
CLAD_OUT_CM   = 0.53886          # R_clad,out [cm]
PELLET_RAD_CM = 0.39168          # R_f        [cm]
SIDE_CM       = 9 * PIN_PITCH_CM  # s = 9p = 10.503 cm

# Cell height H (area-conservation)
H_CM = (3 * np.sqrt(3) / 2 * SIDE_CM**2) / (N_CELLS * PIN_PITCH_CM)

# Slab widths (area-conservation)
W_FUEL_CM = (N_PINS * np.pi * PELLET_RAD_CM**2) / (N_CELLS * H_CM)

W_NA1_CM  = (N_PINS * np.pi * (CLAD_IN_CM**2 - PELLET_RAD_CM**2)) / \
             (2 * N_CELLS * H_CM)

W_FE_CM   = (N_PINS * np.pi * (CLAD_OUT_CM**2 - CLAD_IN_CM**2)) / \
             (2 * N_CELLS * H_CM)

W_NA2_CM  = ((3 * np.sqrt(3) / 2 * SIDE_CM**2) - N_PINS * np.pi * CLAD_OUT_CM**2) / \
             (N_CELLS * H_CM)

# Total assembly width:
#   N_CELLS × (Na2 + Fe + Na1 + Fuel + Na1 + Fe)  +  trailing Na2
#   = N_CELLS × (W_NA2 + 2*W_FE + 2*W_NA1 + W_FUEL)  +  W_NA2
ASSEMBLY_WIDTH_CM = N_CELLS * (W_NA2_CM + 2 * W_FE_CM + 2 * W_NA1_CM + W_FUEL_CM) \
                  + W_NA2_CM

# Sub-slab structure
N_FUEL_SUBSLAB = 5  # equal-width subdivisions per fuel slab
N_NA1_SUBSLAB  = 2  # inner sodium: 2 sub-slabs
N_FE_SUBSLAB   = 1  # single iron slab
N_NA2_SUBSLAB  = 1  # single outer sodium slab

W_FUEL_SUB = W_FUEL_CM / N_FUEL_SUBSLAB
W_NA1_SUB  = W_NA1_CM  / N_NA1_SUBSLAB

# Fuel sub-slab temperatures (boundary → center → boundary, 5 zones)
T_FUEL_SUBS = [850.0, 1000.0, 1200.0, 1000.0, 850.0]  # K

# Inner sodium sub-slab temperatures (clad side → fuel side)
T_NA1_SUBS  = [800.0, 850.0]  # K

T_FE  = 800.0  # K
T_NA2 = 800.0  # K


# =============================================================================
# Per-slab isotopic data
# =============================================================================
# Each entry: (Pu_wt_pct, HM_MT, {isotope: mass_%})
# Isotopic fractions are mass % of the Pu vector (sum = 100 %).
# A missing key means that nuclide is absent in this slab.

PER_SLAB_DATA = [
    # slab 1
    (14.87, 5.112, {'Pu238': 3.57, 'Pu239': 47.38, 'Pu240': 29.66,
                    'Pu241': 8.23, 'Pu242': 10.38, 'Am241':  0.78}),
    # slab 2
    (15.43, 5.334, {'Pu238': 3.61, 'Pu239': 47.10, 'Pu240': 29.82,
                    'Pu241': 8.31, 'Pu242': 10.42, 'Am241':  0.74}),
    # slab 3
    (14.62, 5.089, {'Pu238': 3.48, 'Pu239': 47.65, 'Pu240': 29.51,
                    'Pu241': 8.17, 'Pu242': 10.44, 'Am241':  0.75}),
    # slab 4  — no Pu-238
    (15.76, 5.401, {'Pu239': 47.93, 'Pu240': 30.11,
                    'Pu241': 8.35, 'Pu242': 10.52, 'Am241':  0.60}),
    # slab 5
    (15.08, 5.198, {'Pu238': 3.52, 'Pu239': 47.21, 'Pu240': 30.04,
                    'Pu241': 8.19, 'Pu242': 10.27, 'Am241':  0.77}),
    # slab 6  — no Am-241
    (15.91, 5.447, {'Pu238': 3.63, 'Pu239': 47.55, 'Pu240': 29.43,
                    'Pu241': 8.28, 'Pu242': 11.11}),
    # slab 7
    (14.43, 4.973, {'Pu238': 3.44, 'Pu239': 47.82, 'Pu240': 29.77,
                    'Pu241': 8.14, 'Pu242': 10.15, 'Am241':  0.68}),
    # slab 8  — no Pu-238, no Am-241
    (15.55, 5.362, {'Pu239': 48.21, 'Pu240': 29.58,
                    'Pu241': 8.40, 'Pu242': 10.63}),
    # slab 9
    (14.99, 5.223, {'Pu238': 3.59, 'Pu239': 46.98, 'Pu240': 30.21,
                    'Pu241': 8.26, 'Pu242': 10.18, 'Am241':  0.78}),
    # slab 10
    (15.62, 5.388, {'Pu238': 3.66, 'Pu239': 47.44, 'Pu240': 29.33,
                    'Pu241': 8.09, 'Pu242': 10.71, 'Am241':  0.77}),
    # slab 11  — no Pu-242
    (14.74, 5.067, {'Pu238': 3.51, 'Pu239': 47.71, 'Pu240': 29.88,
                    'Pu241': 8.31,                  'Am241':  0.59}),
    # slab 12
    (16.02, 5.489, {'Pu238': 3.55, 'Pu239': 46.87, 'Pu240': 30.34,
                    'Pu241': 8.44, 'Pu242': 10.02, 'Am241':  0.78}),
    # slab 13  — no Pu-238
    (15.21, 5.241, {'Pu239': 48.05, 'Pu240': 29.64,
                    'Pu241': 8.22, 'Pu242': 10.89, 'Am241':  0.71}),
    # slab 14
    (14.55, 5.031, {'Pu238': 3.60, 'Pu239': 47.29, 'Pu240': 29.95,
                    'Pu241': 8.37, 'Pu242': 10.08, 'Am241':  0.71}),
    # slab 15  — no Am-241
    (15.84, 5.442, {'Pu238': 3.53, 'Pu239': 47.63, 'Pu240': 29.41,
                    'Pu241': 8.18, 'Pu242': 10.47}),
]

assert len(PER_SLAB_DATA) == N_CELLS, "PER_SLAB_DATA must have one entry per cell"


# =============================================================================
# Fixed material properties
# =============================================================================
ZR_WT_PCT    = 10.0   # wt% Zirconium in fuel
FUEL_DENSITY = 15.80  # g/cm³ (cold, 100 % TD)

U235_WT_PCT  = 0.25   # wt% U-235 in uranium
U238_WT_PCT  = 99.75  # wt% U-238 in uranium

# Molar masses [g/mol]
M_MOLAR = {
    'U235':  235.0439299,
    'U238':  238.0507883,
    'Pu238': 238.0495599,
    'Pu239': 239.0521634,
    'Pu240': 240.0538135,
    'Pu241': 241.0568515,
    'Pu242': 242.0587426,
    'Am241': 241.0568291,
    'Zr90':   89.9047044,
    'Na23':   22.9897693,
    'Fe56':   55.9349375,
}

RHO_NA = 0.850  # g/cm³ — sodium density at ~800–850 K
RHO_FE = 7.874  # g/cm³ — iron density at 800 K


# =============================================================================
# Material helpers
# =============================================================================

def _make_fuel_material(cell_index: int, sub_index: int,
                        pu_wt_pct: float, pu_vector: dict,
                        T: float) -> Material:
    """
    Build a U-Pu-Zr metallic fuel Material for cell `cell_index` (0-based),
    sub-slab `sub_index` (0-based), at temperature T.

    Composition:
        - Zirconium : ZR_WT_PCT wt% of total fuel
        - Plutonium : pu_wt_pct wt% of total fuel (Am-241 included)
        - Uranium   : remainder, split U235/U238 per U235_WT_PCT/U238_WT_PCT

    pu_vector gives mass % of each Pu/Am isotope; renormalized to 100% internally.
    Atom densities computed from FUEL_DENSITY.
    """
    w_Zr = ZR_WT_PCT / 100.0
    w_Pu = pu_wt_pct / 100.0
    w_U  = 1.0 - w_Zr - w_Pu

    # Renormalize Pu isotopic vector
    total_pv = sum(pu_vector.values())
    pv_norm  = {iso: frac / total_pv for iso, frac in pu_vector.items()}

    nuclides = []

    # Uranium isotopes
    for iso, w_iso_in_U in [('U235', U235_WT_PCT / 100.0),
                             ('U238', U238_WT_PCT / 100.0)]:
        w_iso = w_U * w_iso_in_U
        N_iso = FUEL_DENSITY * NA * w_iso / M_MOLAR[iso]
        nuclides.append((iso, N_iso))

    # Plutonium / Am isotopes
    for iso, frac_in_Pu in pv_norm.items():
        w_iso = w_Pu * frac_in_Pu
        N_iso = FUEL_DENSITY * NA * w_iso / M_MOLAR[iso]
        nuclides.append((iso, N_iso))

    # Zirconium (single-isotope proxy: Zr-90)
    N_Zr = FUEL_DENSITY * NA * w_Zr / M_MOLAR['Zr90']
    nuclides.append(('Zr90', N_Zr))

    return Material(
        name     = f"fuel_cell{cell_index+1:02d}_sub{sub_index+1}",
        nuclides = nuclides,
        T        = T,
    )


def _make_na_material(cell_index: int, sub_index: int,
                      region_tag: str, T: float) -> Material:
    """Sodium (Na-23) material."""
    N_Na = _atom_density(RHO_NA, M_MOLAR['Na23'])
    return Material(
        name     = f"na_{region_tag}_cell{cell_index+1:02d}_sub{sub_index+1}",
        nuclides = [('Na23', N_Na)],
        T        = T,
    )


def _make_fe_material(cell_index: int, side: str) -> Material:
    """Iron (Fe-56) cladding material. side is 'L' or 'R'."""
    N_Fe = _atom_density(RHO_FE, M_MOLAR['Fe56'])
    return Material(
        name     = f"fe_{side}_cell{cell_index+1:02d}",
        nuclides = [('Fe56', N_Fe)],
        T        = T_FE,
    )


# =============================================================================
# Geometry constructor
# =============================================================================

def create_geometry_SFR_fresh_fuel(
    maj_mat_method:  str,
    maj_xs_method:   str,
    access_method:   str,
    mode:            str,
    xs_maj_file_dir: Optional[str] = None,
    T_array:         Optional[np.ndarray] = None,
    neutron_nbr:     int = 10_000,
) -> Tuple[Geometry, Source]:
    """
    1-D representation of the ESFR-M1 fresh-fuel SFR fissile assembly.

    Layout (left → right):
        [Na2][Fe_L][Na1 Na1][Fuel×5][Na1 Na1][Fe_R] × 15  +  [Na2]

    Each cell contains:
        Na2  : 1 slab  at 800 K
        Fe_L : 1 slab  at 800 K
        Na1  : 2 sub-slabs at [800, 850] K  (clad → fuel)
        Fuel : 5 sub-slabs at [850, 1000, 1200, 1000, 850] K
        Na1  : 2 sub-slabs at [850, 800] K  (fuel → clad)
        Fe_R : 1 slab  at 800 K

    Trailing Na2 closes the assembly symmetrically.
    Reflective boundary conditions at both ends.

    Per-cell composition taken from PER_SLAB_DATA.
    Source: homogeneous log-normal in all fuel sub-slabs.

    Returns
    -------
    geometry : Geometry
    source   : Source
    """
    if T_array is None:
        T_array = np.array([293.6, 500.0, 600.0, 800.0,
                            850.0, 1000.0, 1200.0, 1500.0, 2000.0])  # K

    # ------------------------------------------------------------------
    # Build Geometry object
    # ------------------------------------------------------------------
    geometry = Geometry(majorant_log=False, poll_interval=0.001, verbose=False)
    geometry.xs_dir = '/home/paule/open_mc_projects/xs_lib/endfb-vii.1-hdf5/wmp'

    # ------------------------------------------------------------------
    # Build all materials and register regions
    # ------------------------------------------------------------------
    x_cursor = 0.0
    fuel_subslab_names: list[str] = []

    for cell in range(N_CELLS):
        pu_wt_pct, hm_mt, pu_vector = PER_SLAB_DATA[cell]

        # ── Outer sodium (Na2) ────────────────────────────────────────
        mat = _make_na_material(cell, 0, region_tag='outer', T=T_NA2)
        geometry.add_material(mat)
        geometry.add_region(
            name     = mat.name,
            material = mat,
            x_min    = x_cursor,
            x_max    = x_cursor + W_NA2_CM,
        )
        x_cursor += W_NA2_CM

        # ── Iron cladding — left ──────────────────────────────────────
        mat = _make_fe_material(cell, side='L')
        geometry.add_material(mat)
        geometry.add_region(
            name     = mat.name,
            material = mat,
            x_min    = x_cursor,
            x_max    = x_cursor + W_FE_CM,
        )
        x_cursor += W_FE_CM

        # ── Inner sodium (Na1) — left of fuel ────────────────────────
        # Temperature increases from clad toward fuel: 800 K → 850 K
        for s, T_na in enumerate(T_NA1_SUBS):
            mat = _make_na_material(cell, s, region_tag='inner_L', T=T_na)
            geometry.add_material(mat)
            geometry.add_region(
                name     = mat.name,
                material = mat,
                x_min    = x_cursor,
                x_max    = x_cursor + W_NA1_SUB,
            )
            x_cursor += W_NA1_SUB

        # ── Fuel slab — 5 sub-slabs ───────────────────────────────────
        for s, T_f in enumerate(T_FUEL_SUBS):
            mat = _make_fuel_material(cell, s, pu_wt_pct, pu_vector, T_f)
            geometry.add_material(mat)
            geometry.add_region(
                name     = mat.name,
                material = mat,
                x_min    = x_cursor,
                x_max    = x_cursor + W_FUEL_SUB,
            )
            x_cursor += W_FUEL_SUB
            fuel_subslab_names.append(mat.name)

        # ── Inner sodium (Na1) — right of fuel ───────────────────────
        # Temperature decreases from fuel toward clad: 850 K → 800 K
        for s, T_na in enumerate(reversed(T_NA1_SUBS)):
            mat = _make_na_material(cell, s, region_tag='inner_R', T=T_na)
            geometry.add_material(mat)
            geometry.add_region(
                name     = mat.name,
                material = mat,
                x_min    = x_cursor,
                x_max    = x_cursor + W_NA1_SUB,
            )
            x_cursor += W_NA1_SUB

        # ── Iron cladding — right ─────────────────────────────────────
        mat = _make_fe_material(cell, side='R')
        geometry.add_material(mat)
        geometry.add_region(
            name     = mat.name,
            material = mat,
            x_min    = x_cursor,
            x_max    = x_cursor + W_FE_CM,
        )
        x_cursor += W_FE_CM

    # ── Trailing Na2 to close the assembly ───────────────────────────
    mat = _make_na_material(N_CELLS, 0, region_tag='outer', T=T_NA2)
    geometry.add_material(mat)
    geometry.add_region(
        name     = mat.name,
        material = mat,
        x_min    = x_cursor,
        x_max    = x_cursor + W_NA2_CM,
    )
    x_cursor += W_NA2_CM

    # Sanity check
    assert abs(x_cursor - ASSEMBLY_WIDTH_CM) < 1e-8, (
        f"Geometry width mismatch: got {x_cursor:.6f} cm, "
        f"expected {ASSEMBLY_WIDTH_CM:.6f} cm"
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
        boundaries  = [0, ASSEMBLY_WIDTH_CM / 4,
                       ASSEMBLY_WIDTH_CM / 2,
                       3 * ASSEMBLY_WIDTH_CM / 4,
                       ASSEMBLY_WIDTH_CM],
        energy_bins = energy_bins,
    )

    geometry.attach_verification_tally(
        energy_bins = energy_bins,
        surface_xs  = [0, ASSEMBLY_WIDTH_CM / 4,
                       ASSEMBLY_WIDTH_CM / 2,
                       3 * ASSEMBLY_WIDTH_CM / 4,
                       ASSEMBLY_WIDTH_CM],
        boundaries  = [0, ASSEMBLY_WIDTH_CM / 4,
                       ASSEMBLY_WIDTH_CM / 2,
                       3 * ASSEMBLY_WIDTH_CM / 4,
                       ASSEMBLY_WIDTH_CM],
    )

    # ------------------------------------------------------------------
    # Cutoff energy
    # ------------------------------------------------------------------
    geometry.set_cutoff_energy(70.0)  # eV — start of thermal region

    # ------------------------------------------------------------------
    # Majorant / XS / access / mode settings
    # ------------------------------------------------------------------
    geometry.maj_mat_method = maj_mat_method
    geometry.set_maj_xs_method(
        method          = maj_xs_method,
        T_array         = T_array,
        xs_maj_file_dir = xs_maj_file_dir,
    )

    E_max_wmp = _get_E_max_source(geometry)
    print(f"[geometry] E_max_wmp = {E_max_wmp:.2f} eV  "
          f"(min nuclide.E_max — used for access method and source)")

    geometry.set_access_method(
        access_method,
        last_energy = E_max_wmp,
        err_lim     = 0.001,
        err_max     = 0.01,
    )
    geometry.set_mode(mode, filename='validation/xs_generation/statepoint.200.h5')

    # ------------------------------------------------------------------
    # Source — one SourceRegion per fuel sub-slab (15 cells × 5 = 75)
    # ------------------------------------------------------------------
    source_mu, source_sigma = _lognormal_params(E_mean_eV=115.0)
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

    print(f"[source]   {len(source_regions)} source regions "
          f"({N_CELLS} cells × {N_FUEL_SUBSLAB} fuel sub-slabs)")

    return geometry, source


# =============================================================================
# Quick geometry report
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SFR Fresh-Fuel Benchmark — Geometry & Source Summary")
    print("=" * 60)
    print(f"  Assembly side length s  : {SIDE_CM:.4f} cm")
    print(f"  Pin pitch p             : {PIN_PITCH_CM:.4f} cm")
    print(f"  Number of pins          : {N_PINS}")
    print(f"  Number of cells         : {N_CELLS}")
    print(f"  Cell height H           : {H_CM:.4f} cm")
    print(f"  Fuel slab width  W_f    : {W_FUEL_CM:.4f} cm")
    print(f"  Inner Na width   W_Na1  : {W_NA1_CM:.4f} cm")
    print(f"  Fe clad width    W_Fe   : {W_FE_CM:.4f} cm  (×2 per cell)")
    print(f"  Outer Na width   W_Na2  : {W_NA2_CM:.4f} cm")
    print(f"  Fuel sub-slab width     : {W_FUEL_SUB:.4f} cm  (×{N_FUEL_SUBSLAB})")
    print(f"  Inner Na sub-slab width : {W_NA1_SUB:.4f} cm  (×{N_NA1_SUBSLAB} per side)")
    print(f"  Assembly total width    : {ASSEMBLY_WIDTH_CM:.4f} cm")
    print()
    print("  Fuel sub-slab temperatures  (K) :", T_FUEL_SUBS)
    print("  Inner Na sub-slab temps     (K) :", T_NA1_SUBS)
    print(f"  Fe temperature              (K) : {T_FE}")
    print(f"  Outer Na temperature        (K) : {T_NA2}")
    print()
    print("  Per-cell Pu enrichment and HM mass:")
    for i, (pu_wt, hm, pv) in enumerate(PER_SLAB_DATA):
        present = ', '.join(pv.keys())
        print(f"    Cell {i+1:2d} : {pu_wt:.2f} wt%  {hm:.3f} MT  [{present}]")
    print()
    print("  Source parameters:")
    print(f"    energy_dist  : log_normal")
    print(f"    E_min        : {E_MIN_SOURCE:.1f} eV")
    print(f"    E_max        : min(nuclide.E_max) — resolved at runtime after XS loading")
    print(f"    mu           : ln(500 eV) + sigma²  — mode fixed at 500 eV")
    print(f"    sigma        : 1.0")
    print(f"    spatial      : uniform across all "
          f"{N_CELLS * N_FUEL_SUBSLAB} fuel sub-slabs")
    print()
    print("  Usage:")
    print("    geometry, source = create_geometry_SFR_fresh_fuel(...)")
    print("    source.run_source(geometry)")