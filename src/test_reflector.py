"""
test_reflector.py
=================
Diagnostic suite for the reflective boundary implementation in
src.geometry_classes.Geometry.

Each test is independent and prints a [PASS] / [FAIL] line. Call
run_diagnostic_suite() to execute them all, or invoke any test
function on its own for targeted debugging.

Tests
-----
1. test_boundary_equivalence
       Half-problem ≡ full-problem with central source.
       Per-source flux must match (NO factor of 2 — see notes below).

2. test_full_reflection_mass_balance
       Reflective-reflective BCs trap every neutron, so
       leakage = 0 and absorption per source = 1.

3. test_spatial_symmetry
       In a closed slab with a centered source, the absorption
       density must be left-right symmetric: abs(x) = abs(L - x).
       Catches reflectors that bias neutrons toward one wall.

Why no factor of 2 in test 1
-----------------------------
flux_tally.mean returns Σ track_length / N_histories — already
per-source-normalised. Each Case-A source neutron contributes the
same total track length as a Case-B source neutron (the reflective
wall mirrors the −x hemisphere into +x at unit weight, not double),
so flux_A == flux_B per source. The factor of 2 belongs to ABSOLUTE
track lengths, not to per-source flux.
"""

import numpy as np

from src.geometry_classes import Geometry, Material
from src.source_class    import Source


# ═════════════════════════════════════════════════════════════════════════════
# COMMON SETUP
# ═════════════════════════════════════════════════════════════════════════════
def _make_material(T: float = 300.0) -> Material:
    """Pure U-238 metal at temperature T [K] — the reference test material."""
    rho_U  = 19.1                        # g/cm³
    NA     = 6.02214076e23               # atoms/mol
    M_U8   = 238.05078826                # g/mol
    N_U238 = rho_U * NA / M_U8           # atoms/cm³
    return Material(name="U238", nuclides=[("U238", N_U238)], T=T)


def _setup_majorant(geom: Geometry) -> Geometry:
    """Standard sqrtT_E majorant + reconr access on the 10–600 eV window."""
    geom.maj_mat_method = "simple"
    geom.set_maj_xs_method(method="sqrtT_E")
    geom.set_access_method("reconr",
                           last_energy=600,
                           err_lim=1e-4,
                           err_max=1e-3)
    geom.set_mode("analysis",
                  filename="validation/xs_generation/statepoint.200.h5")
    return geom


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def run_diagnostic_suite() -> None:
    print("\n" + "=" * 60)
    print("  REFLECTIVE BOUNDARY DIAGNOSTIC SUITE")
    print("=" * 60)
    test_boundary_equivalence()
    test_full_reflection_mass_balance()
    test_spatial_symmetry()
    print("\n" + "=" * 60)
    print("  SUITE COMPLETE")
    print("=" * 60 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — half-problem ≡ full-problem
# ═════════════════════════════════════════════════════════════════════════════
def test_boundary_equivalence(n_a: int = 40_000, n_b: int = 20_000) -> None:
    """
    Standard symmetry check:

        full slab [0, 20], vac–vac, isotropic source at x=10
            ≡
        half slab [0, 10], refl–vac, isotropic source at x=0

    Both per-source fluxes must agree within statistics. The 3σ
    tolerance is built from the actual relative errors of the two
    runs, not a fixed percentage — that lets the test catch real
    bugs while staying immune to ordinary statistical noise.
    """
    print("\n[Test 1] Boundary equivalence (half ≡ full)...")

    mat         = _make_material()
    energy_bins = [1e-5, 20.0e6]

    # ── CASE A: full slab, vacuum on both ends, source at the centre ────────
    geom_a = Geometry(verbose=False)
    geom_a.add_region("Full", mat, x_min=0.0, x_max=20.0)
    geom_a.set_boundary_conditions(left="vacuum", right="vacuum")
    geom_a.attach_flux_tally(energy_bins=energy_bins)
    _setup_majorant(geom_a)

    src_a = Source.point(n_a, geom_a, "Full", [10.0, 0.0, 0.0],
                         energy_dist="mono", energy_range=(500.0, 500.0))
    geom_a.run_source(src_a)

    # ── CASE B: half slab, reflective at x=0, vacuum at x=10, source at x=0 ─
    geom_b = Geometry(verbose=False)
    geom_b.add_region("Half", mat, x_min=0.0, x_max=10.0)
    geom_b.set_boundary_conditions(left="reflective", right="vacuum")
    geom_b.attach_flux_tally(energy_bins=energy_bins)
    _setup_majorant(geom_b)

    src_b = Source.point(n_b, geom_b, "Half", [1.0e-7, 0.0, 0.0],
                         energy_dist="mono", energy_range=(500.0, 500.0))
    geom_b.run_source(src_b)

    # ── Compare per-source fluxes (NO factor of 2) ──────────────────────────
    flux_a = float(np.sum(geom_a.flux_tally.mean))
    flux_b = float(np.sum(geom_b.flux_tally.mean))
    re_a   = float(np.mean(geom_a.flux_tally.relative_error))
    re_b   = float(np.mean(geom_b.flux_tally.relative_error))

    sigma  = flux_a * np.sqrt(re_a**2 + re_b**2)        # combined 1σ
    diff   = abs(flux_a - flux_b)
    n_sig  = diff / sigma if sigma > 0 else float("inf")

    print(f"    Full per-source flux : {flux_a:.4e}  (RE {re_a:.2%})")
    print(f"    Half per-source flux : {flux_b:.4e}  (RE {re_b:.2%})")
    print(f"    Δ = {diff:.2e}    3σ tol = {3*sigma:.2e}    distance = {n_sig:.2f}σ")

    if diff <= 3 * sigma:
        print(f"    [PASS] Half ≡ Full within statistics ({n_sig:.2f}σ)")
    else:
        rel = diff / flux_a
        print(f"    [FAIL] Reflector biased by {rel:.2%}  ({n_sig:.2f}σ)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — fully-reflective box, mass balance
# ═════════════════════════════════════════════════════════════════════════════
def test_full_reflection_mass_balance(n_hist: int = 25_000) -> None:
    """
    Reflective–reflective BCs trap every neutron forever, so:

        Σ leakage   = 0
        Σ absorption = 1   per source neutron

    Because the custom code forces an absorption when E < 10 eV
    (`_sample_collision`), every neutron must end its life as an
    absorption event inside the box.

    Catches:
      * any reflective branch that lets a neutron escape
      * miscounted absorption tally
      * lost neutrons (e.g. position folded outside the geometry)
    """
    print("\n[Test 2] Reflective–reflective mass balance...")

    mat         = _make_material()
    energy_bins = [1e-5, 20.0e6]

    geom = Geometry(verbose=False)
    geom.add_region("Box", mat, x_min=0.0, x_max=10.0)
    geom.set_boundary_conditions(left="reflective", right="reflective")
    geom.attach_flux_tally(energy_bins=energy_bins)
    geom.attach_verification_tally(energy_bins=energy_bins,
                                   surface_xs=[0.0, 5.0, 10.0])
    _setup_majorant(geom)

    src = Source.point(n_hist, geom, "Box", [5.0, 0.0, 0.0],
                       energy_dist="mono", energy_range=(500.0, 500.0))
    geom.run_source(src)

    abs_total  = float(np.sum(geom.verif_tally.absorption))
    leak_left  = float(geom.verif_tally.leak_left)
    leak_right = float(geom.verif_tally.leak_right)

    abs_re = float(np.max(geom.verif_tally.absorption_re))
    sigma  = abs_total * abs_re
    diff   = abs(abs_total - 1.0)

    print(f"    Total absorption (per source) : {abs_total:.6f}")
    print(f"    Left  leakage    (per source) : {leak_left:.2e}")
    print(f"    Right leakage    (per source) : {leak_right:.2e}")

    leaked  = (leak_left > 0.0) or (leak_right > 0.0)
    abs_off = diff > 3 * sigma

    if not leaked and not abs_off:
        print(f"    [PASS] No leakage; absorption = 1 ± 3σ ({sigma:.2e})")
    else:
        if leaked:
            print(f"    [FAIL] Reflective wall leaked: "
                  f"left={leak_left:.2e}  right={leak_right:.2e}")
        if abs_off:
            n_sig = diff / sigma if sigma > 0 else float("inf")
            print(f"    [FAIL] Absorption ≠ 1.0 per source: "
                  f"{abs_total:.6f}  ({n_sig:.2f}σ off)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — left-right spatial symmetry
# ═════════════════════════════════════════════════════════════════════════════
def test_spatial_symmetry(n_hist: int = 50_000) -> None:
    """
    With reflective–reflective BCs and a source at the geometric centre,
    the absorption density profile must be mirror-symmetric:

        abs(R_i)  ==  abs(R_{N-1-i})    (within statistics)

    This is a stronger test than uniformity — uniformity needs neutrons
    to thermalise across the whole box, which doesn't always happen
    when resonance absorption dominates. SYMMETRY only requires the
    reflector to be unbiased between left and right; it must hold
    independent of how quickly absorption occurs.

    Catches:
      * direction-flip bugs (e.g. flipping y/z instead of x at the wall)
      * asymmetric overshoot folding
      * any drift introduced by the reflective branch
    """
    print("\n[Test 3] Spatial symmetry around centred source...")

    mat         = _make_material()
    energy_bins = [1e-5, 20.0e6]

    # Five identical 2-cm-wide regions, total slab = 10 cm
    edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    geom  = Geometry(verbose=False)
    for i in range(len(edges) - 1):
        geom.add_region(f"R{i}", mat, x_min=edges[i], x_max=edges[i + 1])
    geom.set_boundary_conditions(left="reflective", right="reflective")
    geom.attach_verification_tally(energy_bins=energy_bins,
                                   surface_xs=edges)
    _setup_majorant(geom)

    # Source dead-centre — equidistant from both reflective walls
    src = Source.point(n_hist, geom, "R2", [5.0, 0.0, 0.0],
                       energy_dist="mono", energy_range=(500.0, 500.0))
    geom.run_source(src)

    # absorption shape (n_space, n_energy) → sum over energy → (n_space,)
    abs_per_region = np.sum(geom.verif_tally.absorption,    axis=1)
    re_per_region  = np.max(geom.verif_tally.absorption_re, axis=1)
    sig_per_region = abs_per_region * re_per_region              # 1σ per bin

    n          = len(abs_per_region)                              # = 5
    pair_diffs = []
    pair_sigs  = []
    print(f"    Per-region absorption (per source):")
    for i, a, sg, re in zip(range(n), abs_per_region,
                             sig_per_region, re_per_region):
        print(f"      R{i}  [{edges[i]:>4.1f}–{edges[i+1]:<4.1f}]  "
              f"abs = {a:.4f}  ±{sg:.2e}  (RE {re:.2%})")

    # Compare each region to its mirror image about the centre
    print(f"    Symmetric pair comparisons:")
    worst_n_sig = 0.0
    for i in range(n // 2):
        j     = n - 1 - i
        d     = abs(abs_per_region[i] - abs_per_region[j])
        s     = np.sqrt(sig_per_region[i]**2 + sig_per_region[j]**2)
        n_sig = d / s if s > 0 else float("inf")
        worst_n_sig = max(worst_n_sig, n_sig)
        print(f"      R{i} ↔ R{j} : Δ = {d:.2e}    3σ = {3*s:.2e}    "
              f"distance = {n_sig:.2f}σ")

    if worst_n_sig <= 3.0:
        print(f"    [PASS] Symmetric within 3σ "
              f"(worst pair: {worst_n_sig:.2f}σ)")
    else:
        print(f"    [FAIL] Asymmetric — reflector biased "
              f"(worst pair: {worst_n_sig:.2f}σ)")


