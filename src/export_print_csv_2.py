import numpy as np
import csv
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.neutron_class import Neutron
from src.source_class import Source, _BatchSource
from src.geometry_classes import Geometry, Material

# ============================================================================
# Helper — safe array extraction from batch_stats
# ============================================================================
def _arr(d, *keys):
    """Walk nested dict keys and return a numpy array."""
    for k in keys:
        d = d[k]
    return np.array(d)

# ============================================================================
# Print and or save memory usage from a batch
# ============================================================================
def export_memory_stats(geom, output_dir=None):
    summary  = geom.memory.summary()
    df_poll  = geom.memory.poll_to_dataframe()

    base = Path(output_dir) if output_dir is not None else Path(".")
    #check if output_dir exists, if not create it
    if output_dir is not None:
        base.mkdir(parents=True, exist_ok=True)

    summary_path = base / "memory_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Memory summary saved → {summary_path}")

    poll_path = base / "memory_poll.csv"
    df_poll.to_csv(poll_path, index=False)
    print(f"Memory poll data saved → {poll_path}")

    return


# ============================================================================
# Print geometry configuration (regions, materials, nuclides, settings)
# ============================================================================
def print_geometry_summary(geom, source=None):
    """
    Print a human-readable summary of the geometry configuration,
    including regions, materials, nuclides, tally flags, solver settings,
    and (optionally) source definition.

    Parameters
    ----------
    geom   : Geometry
    source : Source or _BatchSource, optional
    """
    W = 70
    print("\n" + "=" * W)
    print("  GEOMETRY CONFIGURATION")
    print("=" * W)

    # ── Solver settings ───────────────────────────────────────────────────────
    print(f"\n  {'Mode':<30} {geom.mode}")
    print(f"  {'Majorant XS method':<30} {geom.maj_xs_method}")
    print(f"  {'Access method':<30} {geom.access_method}")
    print(f"  {'Cutoff energy [eV]':<30} {geom.cutoff_energy:.3e}")
    bc_l = geom.boundary_conditions.get("left",  "vacuum")
    bc_r = geom.boundary_conditions.get("right", "vacuum")
    print(f"  {'Boundary conditions':<30} left={bc_l}  right={bc_r}")

    # ── Active tallies / trackers ─────────────────────────────────────────────
    print(f"\n  {'Flux tally':<30} {'ON' if geom.flux_tally_flag   else 'OFF'}")
    print(f"  {'Verification tally':<30} {'ON' if geom.verif_tally_flag  else 'OFF'}")
    print(f"  {'Performance tracker':<30} {'ON' if geom.perf_tracker_flag else 'OFF'}")
    print(f"  {'Majorant log':<30} {'ON' if geom.majorant_log_flag else 'OFF'}")
    print(f"  {'Neutron history':<30} {'ON' if geom.history_flag       else 'OFF'}")
    print(f"  {'Memory tracker':<30} {'ON' if geom.memory_tracker_flag else 'OFF'}")

    # ── Regions ───────────────────────────────────────────────────────────────
    sorted_regions = geom.get_regions()
    print(f"\n  REGIONS  ({len(sorted_regions)} total)")
    print(f"  {'Name':<20} {'Material':<16} {'x_min':>10} {'x_max':>10} {'Width':>10}")
    print("  " + "-" * 68)
    for r in sorted_regions:
        width = r.x_max - r.x_min
        print(f"  {r.name:<20} {r.material.name:<16} "
              f"{r.x_min:>10.4f} {r.x_max:>10.4f} {width:>10.4f}  cm")
    total_width = sorted_regions[-1].x_max - sorted_regions[0].x_min if sorted_regions else 0.0
    print(f"  {'Total geometry width':<20} {'':>16} {'':>10} {'':>10} {total_width:>10.4f}  cm")

    # ── Materials & nuclides ──────────────────────────────────────────────────
    print(f"\n  MATERIALS  ({len(geom.materials)} unique)")
    print("  " + "-" * 68)
    for mat in geom.materials:
        print(f"  {mat.name}  (T = {mat.T:.1f} K,  total density = {mat.total_density:.4e} at/cm³)")
        for nuclide_name, density in mat.nuclides:
            frac = density / mat.total_density * 100.0 if mat.total_density > 0 else 0.0
            print(f"    {'':2}{nuclide_name:<10}  density = {density:.4e} at/cm³  ({frac:.1f}%)")

    # ── Flux tally configuration ──────────────────────────────────────────────
    if geom.flux_tally_flag and geom.flux_tally is not None:
        ft = geom.flux_tally
        eb = ft.energy_bins
        n_e = len(eb) - 1
        print(f"\n  FLUX TALLY (TLE)")
        print(f"  {'Energy groups':<30} {n_e}")
        print(f"  {'Energy range [eV]':<30} [{eb[0]:.3e}, {eb[-1]:.3e}]")
        print(f"  {'Transverse area [cm²]':<30} {ft.transverse_area:.4f}")

    # ── Verification tally configuration ─────────────────────────────────────
    if geom.verif_tally_flag and geom.verif_tally is not None:
        vt = geom.verif_tally
        eb = vt.energy_bins
        sb = vt.boundaries
        sxs = vt.surface_xs
        n_e = len(eb) - 1
        n_s = len(sb) - 1
        print(f"\n  VERIFICATION TALLY")
        print(f"  {'Spatial bins':<30} {n_s}")
        print(f"  {'Geometry span [cm]':<30} [{sb[0]:.3f}, {sb[-1]:.3f}]")
        print(f"  {'Energy groups':<30} {n_e}")
        print(f"  {'Energy range [eV]':<30} [{eb[0]:.3e}, {eb[-1]:.3e}]")
        print(f"  {'Surface current detectors':<30} {len(sxs)}")
        print(f"  {'  at x [cm]':<30} " + ", ".join(f"{x:.2f}" for x in sxs))

    # ── Source summary (if provided) ──────────────────────────────────────────
    if source is not None:
        _print_source_summary(source)

    print("\n" + "=" * W)


# ============================================================================
# Print source configuration
# ============================================================================
def _print_source_summary(source):
    """Inner helper — prints source block within a wider summary."""
    W = 70
    print(f"\n  SOURCE DEFINITION")
    print("  " + "-" * W)

    if isinstance(source, _BatchSource):
        print(f"  Type                           _BatchSource (pre-sampled batch)")
        print(f"  Neutrons in batch              {source.neutron_nbr:,}")
        return

    # Source object
    print(f"  {'Total neutrons':<30} {source.neutron_nbr:,}")
    print(f"  {'Source regions':<30} {len(source.source_regions)}")

    is_point = hasattr(source, "_point_position")
    if is_point:
        pos = source._point_position
        print(f"  {'Type':<30} Point source")
        print(f"  {'Position [cm]':<30} x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")

    for i, sr in enumerate(source.source_regions):
        tag = f"  Region [{i}]"
        print(f"\n{tag}")
        print(f"    {'name':<26} {sr.region_name}")
        print(f"    {'weight (normalised)':<26} {source._norm_weights[i]:.4f}")
        print(f"    {'energy_dist':<26} {sr.energy_dist}")
        if sr.energy_dist == "log_normal":
            print(f"    {'mu':<26} {sr.mu}")
            print(f"    {'sigma':<26} {sr.sigma}")
        print(f"    {'energy_range [eV]':<26} [{sr.energy_range[0]:.3e}, {sr.energy_range[1]:.3e}]")
        print(f"    {'direction_dist':<26} {sr.direction_dist}")
        if sr.direction_dist == "forward":
            print(f"    {'direction':<26} {sr.direction}")


def print_source_summary(source):
    """
    Standalone pretty-printer for a Source or _BatchSource object.

    Parameters
    ----------
    source : Source | _BatchSource
    """
    W = 70
    print("\n" + "=" * W)
    print("  SOURCE CONFIGURATION")
    print("=" * W)
    _print_source_summary(source)
    print("=" * W)


# ============================================================================
# Print and or save all cross-batch statistics
# ============================================================================

def export_cross_batch_stats(batch_stats, geom,
                             print_to_console=True,
                             save_csv=False,
                             output_dir=None,
                             source=None):
    """
    Parameters
    ----------
    batch_stats      : dict returned by geom.cross_batch_stats()
    geom             : Geometry instance
    print_to_console : bool — print formatted tables
    save_csv         : bool — write CSVs to output_dir
    output_dir       : str | None — directory for CSV output
    source           : Source | _BatchSource | None — if supplied, the source
                       configuration is printed / saved alongside the stats
    """
    eb = np.array(batch_stats["flux"]["energy_bins"])
    n_groups = len(eb) - 1
    group_labels = [f"{eb[i]:.0f}-{eb[i+1]:.0f} eV" for i in range(n_groups)]

    # ── Flux ──────────────────────────────────────────────────────────────────
    flux_mean = _arr(batch_stats, "flux", "mean")
    flux_std  = _arr(batch_stats, "flux", "std")
    flux_re   = _arr(batch_stats, "flux", "relative_error")

    # ── Absorption ────────────────────────────────────────────────────────────
    abs_mean = _arr(batch_stats, "verif", "absorption", "mean")
    abs_std  = _arr(batch_stats, "verif", "absorption", "std")
    abs_re   = _arr(batch_stats, "verif", "absorption", "relative_error")
    sxb      = np.array(batch_stats["verif"]["boundaries"])
    n_space  = len(sxb) - 1
    space_labels = [f"{sxb[i]:.1f}-{sxb[i+1]:.1f} cm" for i in range(n_space)]

    # ── Scatter ───────────────────────────────────────────────────────────────
    sct_mean = _arr(batch_stats, "verif", "scatter", "mean")
    sct_std  = _arr(batch_stats, "verif", "scatter", "std")
    sct_re   = _arr(batch_stats, "verif", "scatter", "relative_error")

    # ── Surface currents ──────────────────────────────────────────────────────
    sfx      = np.array(batch_stats["verif"]["surface_xs"])
    fwd_mean = _arr(batch_stats, "verif", "current_fwd", "mean")
    fwd_std  = _arr(batch_stats, "verif", "current_fwd", "std")
    bwd_mean = _arr(batch_stats, "verif", "current_bwd", "mean")
    bwd_std  = _arr(batch_stats, "verif", "current_bwd", "std")

    # ── Leakage ───────────────────────────────────────────────────────────────
    leak_l = batch_stats["verif"]["leak_left"]
    leak_r = batch_stats["verif"]["leak_right"]

    perf = batch_stats["perf"]

    if print_to_console:

        # ── Geometry + source header ───────────────────────────────────────────
        print_geometry_summary(geom, source=source)

        print("\n" + "="*70)
        print("  CROSS-BATCH STATISTICS")
        print("="*70)

        # ── Flux tally ─────────────────────────────────────────────────────────
        print("\n  FLUX TALLY [cm · src-n⁻¹]")
        print(f"  {'Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*65)
        for i in range(n_groups):
            print(f"  {group_labels[i]:<25} {flux_mean[i]:>14.4e} "
                  f"{flux_std[i]:>14.4e} {flux_re[i]:>10.4f}")

        # ── Verification tally: absorption ─────────────────────────────────────
        print("\n  ABSORPTION RATE [reactions · src-n⁻¹]")
        print(f"  {'Region / Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*65)
        for si in range(n_space):
            for gi in range(n_groups):
                label = f"{space_labels[si]} | {group_labels[gi]}"
                print(f"  {label:<25} {abs_mean[si][gi]:>14.4e} "
                      f"{abs_std[si][gi]:>14.4e} {abs_re[si][gi]:>10.4f}")

        # ── Verification tally: scatter ────────────────────────────────────────
        print("\n  SCATTER RATE [reactions · src-n⁻¹]")
        print(f"  {'Region / Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*65)
        for si in range(n_space):
            for gi in range(n_groups):
                label = f"{space_labels[si]} | {group_labels[gi]}"
                print(f"  {label:<25} {sct_mean[si][gi]:>14.4e} "
                      f"{sct_std[si][gi]:>14.4e} {sct_re[si][gi]:>10.4f}")

        # ── Verification tally: surface currents ───────────────────────────────
        print("\n  SURFACE CURRENTS [particles · src-n⁻¹]")
        print(f"  {'Surface':>10} {'Fwd Mean':>14} {'±Std':>12} "
              f"{'Bwd Mean':>14} {'±Std':>12} {'Net Mean':>14}")
        print("  " + "-"*80)
        for si, sx in enumerate(sfx):
            net = fwd_mean[si] - bwd_mean[si]
            print(f"  {sx:>10.2f} cm  {fwd_mean[si]:>14.4e} {fwd_std[si]:>12.4e} "
                  f"{bwd_mean[si]:>14.4e} {bwd_std[si]:>12.4e} {net:>14.4e}")

        # ── Verification tally: leakage ────────────────────────────────────────
        print("\n  LEAKAGE [particles · src-n⁻¹]")
        print(f"  {'Quantity':<20} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*60)
        print(f"  {'Left  (x=0.0 cm)':<20} {leak_l['mean']:>14.4e} "
              f"{leak_l['std']:>14.4e} {leak_l['relative_error']:>10.4f}")
        print(f"  {'Right (x=15.0 cm)':<20} {leak_r['mean']:>14.4e} "
              f"{leak_r['std']:>14.4e} {leak_r['relative_error']:>10.4f}")
        total_mean = leak_l['mean'] + leak_r['mean']
        print(f"  {'Total':<20} {total_mean:>14.4e}")

        # ── Performance tracker ────────────────────────────────────────────────
        print("\n  PERFORMANCE (cross-batch)")
        print(f"  {'Metric':<38} {'Mean':>14} {'±Std':>14}")
        print("  " + "-"*68)

        # timing keys — aggregated totals
        for key in (
            "time_preprocessing_s", "time_run_source_s", "time_total_s",
            "neutrons_per_second",  "rejection_fraction", "cpu_efficiency",
        ):
            if key not in perf: continue
            print(f"  {key:<38} {perf[key]['mean']:>14.4f} {perf[key]['std']:>14.4f}")

        # majorant timing — total then WMP/ENDF split
        for key in ("time_majorant_s", "time_majorant_wmp_s", "time_majorant_endf_s"):
            if key not in perf: continue
            label = key.replace("time_majorant", "  ↳ majorant") if "wmp" in key or "endf" in key else key
            print(f"  {label:<38} {perf[key]['mean']:>14.4f} {perf[key]['std']:>14.4f}")

        # xs eval timing — total then WMP/ENDF split
        for key in ("time_xs_eval_s", "time_xs_eval_wmp_s", "time_xs_eval_endf_s"):
            if key not in perf: continue
            label = key.replace("time_xs_eval", "  ↳ xs eval") if "wmp" in key or "endf" in key else key
            print(f"  {label:<38} {perf[key]['mean']:>14.4f} {perf[key]['std']:>14.4f}")

        print("  " + "-"*68)

        # collision counts — total then WMP/ENDF split
        print(f"  {'n_neutrons (total)':<38} {perf['n_neutrons']:>14,}")
        for label, key in (
            ("n_real_collisions",       "n_real_collisions"),
            ("  ↳ WMP",                 "n_real_collisions_wmp"),
            ("  ↳ ENDF",                "n_real_collisions_endf"),
            ("n_virtual_collisions",    "n_virtual_collisions"),
            ("  ↳ WMP",                 "n_virtual_collisions_wmp"),
            ("  ↳ ENDF",               "n_virtual_collisions_endf"),
            ("n_xs_evaluations",        "n_xs_evaluations"),
            ("  ↳ WMP",                "n_xs_evaluations_wmp"),
            ("  ↳ ENDF",               "n_xs_evaluations_endf"),
            ("n_majorant_updates",      "n_majorant_updates"),
            ("  ↳ WMP",                "n_majorant_updates_wmp"),
            ("  ↳ ENDF",              "n_majorant_updates_endf"),
        ):
            if key not in perf: continue
            print(f"  {label:<38} {perf[key]:>14,}")

        # per-event timing — total + WMP/ENDF split
        print("  " + "-"*68)
        if perf.get("n_majorant_updates", 0) > 0:
            print(f"  {'time per majorant update (ms)':<38} "
                f"{1000 * perf['time_majorant_s']['mean'] / perf['n_majorant_updates']:>14.4f}")
        if perf.get("n_majorant_updates_wmp", 0) > 0:
            print(f"  {'  ↳ WMP':<38} "
                f"{1000 * perf['time_majorant_wmp_s']['mean'] / perf['n_majorant_updates_wmp']:>14.4f}")
        if perf.get("n_majorant_updates_endf", 0) > 0:
            print(f"  {'  ↳ ENDF':<38} "
                f"{1000 * perf['time_majorant_endf_s']['mean'] / perf['n_majorant_updates_endf']:>14.4f}")
        if perf.get("n_xs_evaluations", 0) > 0:
            print(f"  {'time per xs eval (ms)':<38} "
                f"{1000 * perf['time_xs_eval_s']['mean'] / perf['n_xs_evaluations']:>14.4f}")
        if perf.get("n_xs_evaluations_wmp", 0) > 0:
            print(f"  {'  ↳ WMP':<38} "
                f"{1000 * perf['time_xs_eval_wmp_s']['mean'] / perf['n_xs_evaluations_wmp']:>14.4f}")
        if perf.get("n_xs_evaluations_endf", 0) > 0:
            print(f"  {'  ↳ ENDF':<38} "
                f"{1000 * perf['time_xs_eval_endf_s']['mean'] / perf['n_xs_evaluations_endf']:>14.4f}")

        print("="*70)

        # wrong majorant — total then WMP/ENDF split
        print("\n  WRONG MAJORANT STATISTICS")
        print(f"  {'Metric':<38} {'Mean':>14} {'±Std':>14} {'Min':>10} {'Max':>10}")
        print("  " + "-"*80)
        for label, k in (
            ("wrong_majorant_fraction",       "wrong_majorant_fraction"),
            ("  ↳ WMP",                       "wrong_majorant_fraction_wmp"),
            ("  ↳ ENDF",                      "wrong_majorant_fraction_endf"),
            ("wrong_majorant_mean_error",      "wrong_majorant_mean_error"),
            ("  ↳ WMP",                       "wrong_majorant_mean_error_wmp"),
            ("  ↳ ENDF",                      "wrong_majorant_mean_error_endf"),
        ):
            if k not in perf: continue
            d = perf[k]
            print(f"  {label:<38} {d['mean']:>14.4e} {d['std']:>14.4e} "
                  f"{d.get('min', float('nan')):>10.4e} {d.get('max', float('nan')):>10.4e}")

        for label, k in (
            ("n_wrong_majorant (total)", "n_wrong_majorant"),
            ("  ↳ WMP",                 "n_wrong_majorant_wmp"),
            ("  ↳ ENDF",                "n_wrong_majorant_endf"),
        ):
            if k not in perf: continue
            print(f"  {label:<38} {perf[k]:>14,}")

    # ============================================================================
    # Save cross-batch statistics CSV
    # ============================================================================
    if save_csv:
        base = Path(output_dir) if output_dir is not None else Path(".")
        if output_dir is not None:
            base.mkdir(parents=True, exist_ok=True)

        cross_rows = []

        # flux
        for i in range(n_groups):
            cross_rows.append({
                "tally": "flux", "region": "all",
                "energy_group": group_labels[i],
                "mean": flux_mean[i], "std": flux_std[i],
                "relative_error": flux_re[i],
            })

        # absorption
        for si in range(n_space):
            for gi in range(n_groups):
                cross_rows.append({
                    "tally": "absorption", "region": space_labels[si],
                    "energy_group": group_labels[gi],
                    "mean": abs_mean[si][gi], "std": abs_std[si][gi],
                    "relative_error": abs_re[si][gi],
                })

        # scatter
        for si in range(n_space):
            for gi in range(n_groups):
                cross_rows.append({
                    "tally": "scatter", "region": space_labels[si],
                    "energy_group": group_labels[gi],
                    "mean": sct_mean[si][gi], "std": sct_std[si][gi],
                    "relative_error": sct_re[si][gi],
                })

        # surface currents
        for si, sx in enumerate(sfx):
            for direction, m, s in [("forward",  fwd_mean[si], fwd_std[si]),
                                     ("backward", bwd_mean[si], bwd_std[si])]:
                cross_rows.append({
                    "tally": f"current_{direction}", "region": f"x={sx:.2f} cm",
                    "energy_group": "all", "mean": m, "std": s,
                    "relative_error": s / abs(m) if m != 0 else float("inf"),
                })

        # leakage
        for side, d in [("leak_left", leak_l), ("leak_right", leak_r)]:
            cross_rows.append({
                "tally": side, "region": "boundary", "energy_group": "all",
                "mean": d["mean"], "std": d["std"],
                "relative_error": d["relative_error"],
            })

        # performance — timing keys (total + split)
        for key in (
            "time_preprocessing_s",
            "time_run_source_s", "time_total_s",
            "neutrons_per_second", "rejection_fraction",
            "rejection_fraction_wmp", "rejection_fraction_endf",
            "cpu_efficiency",
            "time_majorant_s",    "time_majorant_wmp_s",    "time_majorant_endf_s",
            "time_xs_eval_s",     "time_xs_eval_wmp_s",     "time_xs_eval_endf_s",
        ):
            if key not in perf: continue
            cross_rows.append({
                "tally": key, "region": "performance", "energy_group": "all",
                "mean": perf[key]["mean"], "std": perf[key]["std"],
                "relative_error": float("nan"),
            })

        # integer count keys (total + split)
        for key in (
            "n_neutrons",
            "n_real_collisions",    "n_real_collisions_wmp",    "n_real_collisions_endf",
            "n_virtual_collisions", "n_virtual_collisions_wmp", "n_virtual_collisions_endf",
            "n_xs_evaluations",     "n_xs_evaluations_wmp",     "n_xs_evaluations_endf",
            "n_majorant_updates",   "n_majorant_updates_wmp",   "n_majorant_updates_endf",
            "n_wrong_majorant",     "n_wrong_majorant_wmp",     "n_wrong_majorant_endf",
        ):
            if key not in perf: continue
            cross_rows.append({
                "tally": key, "region": "performance", "energy_group": "all",
                "mean": perf[key], "std": float("nan"),
                "relative_error": float("nan"),
            })

        # wrong-majorant fraction/error keys (total + split)
        for k in (
            "wrong_majorant_fraction",      "wrong_majorant_mean_error",
            "wrong_majorant_fraction_wmp",  "wrong_majorant_mean_error_wmp",
            "wrong_majorant_fraction_endf", "wrong_majorant_mean_error_endf",
        ):
            if k not in perf: continue
            cross_rows.append({
                "tally": k, "region": "performance", "energy_group": "all",
                "mean": perf[k]["mean"], "std": perf[k]["std"],
                "relative_error": (perf[k]["std"] / abs(perf[k]["mean"])
                                   if perf[k]["mean"] != 0 else float("inf")),
            })

        # per-event timing derived metrics — total + WMP/ENDF split
        for label, t_key, n_key in (
            ("time_per_majorant_update_ms",      "time_majorant_s",      "n_majorant_updates"),
            ("time_per_majorant_update_wmp_ms",  "time_majorant_wmp_s",  "n_majorant_updates_wmp"),
            ("time_per_majorant_update_endf_ms", "time_majorant_endf_s", "n_majorant_updates_endf"),
            ("time_per_xs_eval_ms",              "time_xs_eval_s",       "n_xs_evaluations"),
            ("time_per_xs_eval_wmp_ms",          "time_xs_eval_wmp_s",   "n_xs_evaluations_wmp"),
            ("time_per_xs_eval_endf_ms",         "time_xs_eval_endf_s",  "n_xs_evaluations_endf"),
        ):
            n = perf.get(n_key, 0)
            if n and n > 0 and t_key in perf:
                val = 1000 * perf[t_key]["mean"] / n
                cross_rows.append({
                    "tally": label, "region": "performance", "energy_group": "all",
                    "mean": val, "std": float("nan"), "relative_error": float("nan"),
                })

        cross_path = base / "cross_batch_statistics_corrected.csv"
        with open(cross_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tally", "region", "energy_group",
                                              "mean", "std", "relative_error"])
            w.writeheader()
            w.writerows(cross_rows)
        print(f"\nCross-batch statistics saved → {cross_path}")

        # ── Geometry summary CSV ───────────────────────────────────────────────
        geom_rows = []
        sorted_regions = geom.get_regions()
        for r in sorted_regions:
            geom_rows.append({
                "type": "region",
                "name": r.name,
                "material": r.material.name,
                "x_min_cm": r.x_min,
                "x_max_cm": r.x_max,
                "width_cm": r.x_max - r.x_min,
                "nuclide": "",
                "density_at_per_cm3": "",
                "temperature_K": r.material.T,
                "detail": "",
            })
            for nuclide_name, density in r.material.nuclides:
                geom_rows.append({
                    "type": "nuclide",
                    "name": r.name,
                    "material": r.material.name,
                    "x_min_cm": r.x_min,
                    "x_max_cm": r.x_max,
                    "width_cm": r.x_max - r.x_min,
                    "nuclide": nuclide_name,
                    "density_at_per_cm3": density,
                    "temperature_K": r.material.T,
                    "detail": "",
                })
        # settings rows
        for key, val in [
            ("mode",          geom.mode),
            ("maj_xs_method", geom.maj_xs_method),
            ("access_method", geom.access_method),
            ("cutoff_energy", geom.cutoff_energy),
            ("bc_left",       geom.boundary_conditions.get("left",  "vacuum")),
            ("bc_right",      geom.boundary_conditions.get("right", "vacuum")),
            ("flux_tally",    geom.flux_tally_flag),
            ("verif_tally",   geom.verif_tally_flag),
            ("perf_tracker",  geom.perf_tracker_flag),
            ("majorant_log",  geom.majorant_log_flag),
            ("history_flag",  geom.history_flag),
            ("memory_flag",   geom.memory_tracker_flag),
        ]:
            geom_rows.append({
                "type": "setting", "name": key, "material": "", "x_min_cm": "",
                "x_max_cm": "", "width_cm": "", "nuclide": "",
                "density_at_per_cm3": "", "temperature_K": "", "detail": str(val),
            })

        geom_path = base / "geometry_summary.csv"
        with open(geom_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "type", "name", "material", "x_min_cm", "x_max_cm", "width_cm",
                "nuclide", "density_at_per_cm3", "temperature_K", "detail"])
            w.writeheader()
            w.writerows(geom_rows)
        print(f"Geometry summary saved       → {geom_path}")

        # ── Source summary CSV ─────────────────────────────────────────────────
        if source is not None:
            src_rows = _source_to_rows(source)
            src_path = base / "source_summary.csv"
            with open(src_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "region_index", "region_name", "norm_weight",
                    "energy_dist", "energy_min_eV", "energy_max_eV",
                    "mu", "sigma", "direction_dist", "direction",
                    "point_position", "total_neutrons"])
                w.writeheader()
                w.writerows(src_rows)
            print(f"Source summary saved         → {src_path}")


def _source_to_rows(source) -> list:
    """Convert a Source or _BatchSource to a list of CSV-ready dicts."""
    if isinstance(source, _BatchSource):
        return [{
            "region_index": 0, "region_name": "batch",
            "norm_weight": 1.0, "energy_dist": "pre-sampled",
            "energy_min_eV": "", "energy_max_eV": "",
            "mu": "", "sigma": "", "direction_dist": "pre-sampled",
            "direction": "", "point_position": "",
            "total_neutrons": source.neutron_nbr,
        }]
    rows = []
    is_point = hasattr(source, "_point_position")
    pos_str = (f"[{source._point_position[0]:.4f}, "
               f"{source._point_position[1]:.4f}, "
               f"{source._point_position[2]:.4f}]") if is_point else ""
    for i, sr in enumerate(source.source_regions):
        rows.append({
            "region_index":   i,
            "region_name":    sr.region_name,
            "norm_weight":    round(source._norm_weights[i], 8),
            "energy_dist":    sr.energy_dist,
            "energy_min_eV":  sr.energy_range[0],
            "energy_max_eV":  sr.energy_range[1],
            "mu":             sr.mu    if sr.mu    is not None else "",
            "sigma":          sr.sigma if sr.sigma is not None else "",
            "direction_dist": sr.direction_dist,
            "direction":      str(sr.direction) if sr.direction_dist == "forward" else "",
            "point_position": pos_str,
            "total_neutrons": source.neutron_nbr,
        })
    return rows


def export_inner_batch_stats_csv(batch_stats, geom, output_dir=None):
    eb = np.array(batch_stats["flux"]["energy_bins"])
    n_groups = len(eb) - 1
    group_labels = [f"{eb[i]:.0f}-{eb[i+1]:.0f} eV" for i in range(n_groups)]
    sxb      = np.array(geom.boundaries)
    n_space  = len(sxb) - 1
    space_labels = [f"{sxb[i]:.1f}-{sxb[i+1]:.1f} cm" for i in range(n_space)]
    sfx      = np.array(batch_stats["verif"]["surface_xs"])

    inner_rows = []

    for snap in geom.batch_results:
        b      = snap["batch"]
        b_n    = snap["n_neutrons"]
        b_time = snap["perf"]["total_time_s"]

        # flux per batch
        if "flux" in snap:
            b_flux_mean = snap["flux"]["flux"]["mean"]
            b_flux_std  = snap["flux"]["flux"]["std"]
            b_flux_re   = snap["flux"]["flux"]["relative_error"]
            for i in range(n_groups):
                inner_rows.append({
                    "batch": b, "n_neutrons": b_n, "wall_time_s": b_time,
                    "tally": "flux", "region": "all",
                    "energy_group": group_labels[i],
                    "mean": b_flux_mean[i], "std": b_flux_std[i],
                    "relative_error": b_flux_re[i],
                })

        # verif per batch
        if "verif" in snap:
            vsnap = snap["verif"]

            # absorption + scatter
            for tally_key in ("absorption", "scatter"):
                t_mean = np.array(vsnap[tally_key]["mean"])
                t_std  = np.array(vsnap[tally_key]["std"])
                t_re   = np.array(vsnap[tally_key]["relative_error"])
                for si in range(n_space):
                    for gi in range(n_groups):
                        inner_rows.append({
                            "batch": b, "n_neutrons": b_n, "wall_time_s": b_time,
                            "tally": tally_key, "region": space_labels[si],
                            "energy_group": group_labels[gi],
                            "mean": t_mean[si][gi], "std": t_std[si][gi],
                            "relative_error": t_re[si][gi],
                        })

            # surface currents
            for direction, key in [("forward",  "current_fwd"),
                                    ("backward", "current_bwd")]:
                c_mean = np.array(vsnap[key]["mean"])
                c_std  = np.array(vsnap[key]["std"])
                c_re   = np.array(vsnap[key]["relative_error"])
                for si, sx in enumerate(sfx):
                    inner_rows.append({
                        "batch": b, "n_neutrons": b_n, "wall_time_s": b_time,
                        "tally": f"current_{direction}",
                        "region": f"x={sx:.2f} cm", "energy_group": "all",
                        "mean": c_mean[si], "std": c_std[si],
                        "relative_error": c_re[si],
                    })

            # leakage
            for side in ("leak_left", "leak_right"):
                ld = vsnap[side]
                inner_rows.append({
                    "batch": b, "n_neutrons": b_n, "wall_time_s": b_time,
                    "tally": side, "region": "boundary", "energy_group": "all",
                    "mean": ld["mean"], "std": ld["std"],
                    "relative_error": ld["relative_error"],
                })

    base = Path(output_dir) if output_dir is not None else Path(".")
    if output_dir is not None:
        base.mkdir(parents=True, exist_ok=True)

    inner_path = base / "inner_batch_statistics_corrected.csv"
    with open(inner_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "n_neutrons", "wall_time_s",
                                          "tally", "region", "energy_group",
                                          "mean", "std", "relative_error"])
        w.writeheader()
        w.writerows(inner_rows)
    print(f"Inner-batch statistics saved  → {inner_path}")