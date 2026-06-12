"""
export_stats.py
===============
Export and print cross-batch simulation statistics.
Based on Document 7, updated for the new performance_classes fields:

  New fields printed/saved vs Document 7:
  - Memory section: baseline_rss_mb, setup_mem_mb, peak_mem_mb,
                    python_heap_setup_kb, python_heap_transport_kb
  - µs per operation: us_per_majorant_*, us_per_xs_eval_* (WMP/ENDF split)
  - Performance by energy bin: Extracted from perf["energy_bins"], printed 
    to console, and saved to a dedicated 'energy_bins_statistics.csv'.

  Bug fixed vs Document 7 (_compute_batch_stats in geometry_classes.py):
  - n_majorant_updates and n_xs_evaluations were in BOTH perf_float_keys
    (producing a {'mean':..., 'std':...} dict) AND in the integer sum loop
    (overwriting with a plain int). The integer loop wins, so in the export
    they must be read as plain int — which is what this file does.
    Fix in geometry_classes.py: remove "n_majorant_updates" and
    "n_xs_evaluations" from perf_float_keys. They belong only in the
    integer sum loop.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.neutron_class import Neutron
from src.source_class import Source, _BatchSource
from src.geometry_classes import Geometry, Material


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arr(d, *keys):
    """Walk nested dict keys and return a numpy array."""
    for k in keys:
        d = d[k]
    return np.array(d)


def _fmean(perf, key, fmt=".4f"):
    """Format a float-stat key (mean ± std dict) or plain value safely."""
    v = perf.get(key)
    if v is None:
        return "N/A", "N/A"
    if isinstance(v, dict):
        m = v.get("mean", float("nan"))
        s = v.get("std",  float("nan"))
    else:
        m, s = float(v), float("nan")
    fm = f"{m:{fmt}}" if math.isfinite(m) else "N/A"
    fs = f"{s:{fmt}}" if math.isfinite(s) else "N/A"
    return fm, fs


def _fint(perf, key):
    """Format an integer-stat key safely."""
    v = perf.get(key)
    if v is None:
        return "N/A"
    if isinstance(v, dict):          # should not happen for int keys, but guard
        v = v.get("mean", 0)
    return f"{int(v):,}"


# ─────────────────────────────────────────────────────────────────────────────
# Memory stats
# ─────────────────────────────────────────────────────────────────────────────

def export_memory_stats(geom, output_dir=None):
    """Save MemoryTracker summary and poll CSV."""
    summary = geom.memory.summary()
    df_poll = geom.memory.poll_to_dataframe()

    base = Path(output_dir) if output_dir is not None else Path(".")
    if output_dir is not None:
        base.mkdir(parents=True, exist_ok=True)

    summary_path = base / "memory_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Memory summary saved → {summary_path}")

    poll_path = base / "memory_poll.csv"
    df_poll.to_csv(poll_path, index=False)
    print(f"Memory poll data saved → {poll_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry + source summary (console)
# ─────────────────────────────────────────────────────────────────────────────

def print_geometry_summary(geom, source=None):
    W = 70
    print("\n" + "=" * W)
    print("  GEOMETRY CONFIGURATION")
    print("=" * W)

    print(f"\n  {'Mode':<30} {geom.mode}")
    print(f"  {'Majorant XS method':<30} {geom.maj_xs_method}")
    print(f"  {'Material majorant method':<30} {geom.maj_mat_method}")
    print(f"  {'Access method':<30} {geom.access_method}")
    print(f"  {'Cutoff energy [eV]':<30} {geom.cutoff_energy:.3e}")
    bc_l = geom.boundary_conditions.get("left",  "vacuum")
    bc_r = geom.boundary_conditions.get("right", "vacuum")
    print(f"  {'Boundary conditions':<30} left={bc_l}  right={bc_r}")

    print(f"\n  {'Flux tally':<30} {'ON' if geom.flux_tally_flag    else 'OFF'}")
    print(f"  {'Verification tally':<30} {'ON' if geom.verif_tally_flag   else 'OFF'}")
    print(f"  {'Performance tracker':<30} {'ON' if geom.perf_tracker_flag  else 'OFF'}")
    print(f"  {'Majorant log':<30} {'ON' if geom.majorant_log_flag  else 'OFF'}")
    print(f"  {'Neutron history':<30} {'ON' if geom.history_flag        else 'OFF'}")
    print(f"  {'Memory tracker':<30} {'ON' if geom.memory_tracker_flag else 'OFF'}")
    print(f"  {'tracemalloc':<30} {'ON' if geom.perf._tracemalloc_enabled else 'OFF'}")

    sorted_regions = geom.get_regions()
    print(f"\n  REGIONS  ({len(sorted_regions)} total)")
    print(f"  {'Name':<20} {'Material':<16} {'x_min':>10} {'x_max':>10} {'Width':>10}")
    print("  " + "-" * 68)
    for r in sorted_regions:
        width = r.x_max - r.x_min
        print(f"  {r.name:<20} {r.material.name:<16} "
              f"{r.x_min:>10.4f} {r.x_max:>10.4f} {width:>10.4f}  cm")
    total_width = (sorted_regions[-1].x_max - sorted_regions[0].x_min
                   if sorted_regions else 0.0)
    print(f"  {'Total width':<20} {'':>16} {'':>10} {'':>10} {total_width:>10.4f}  cm")

    print(f"\n  MATERIALS  ({len(geom.materials)} unique)")
    print("  " + "-" * 68)
    for mat in geom.materials:
        print(f"  {mat.name}  (T = {mat.T:.1f} K,  "
              f"total density = {mat.total_density:.4e} at/cm³)")
        for nuclide_name, density in mat.nuclides:
            frac = (density / mat.total_density * 100.0
                    if mat.total_density > 0 else 0.0)
            print(f"    {nuclide_name:<10}  density = {density:.4e} at/cm³  "
                  f"({frac:.1f}%)")

    if geom.flux_tally_flag and geom.flux_tally is not None:
        ft = geom.flux_tally
        eb = ft.energy_bins
        print(f"\n  FLUX TALLY  ({len(eb)-1} groups, "
              f"{eb[0]:.3e}–{eb[-1]:.3e} eV, "
              f"area = {ft.transverse_area:.4f} cm²)")

    if geom.verif_tally_flag and geom.verif_tally is not None:
        vt = geom.verif_tally
        print(f"\n  VERIFICATION TALLY  ({len(vt.boundaries)-1} spatial bins, "
              f"{len(vt.energy_bins)-1} energy groups)")
        print(f"  Surface detectors at: "
              + ", ".join(f"{x:.2f}" for x in vt.surface_xs) + " cm")

    if source is not None:
        _print_source_summary(source)

    print("\n" + "=" * W)


def _print_source_summary(source):
    W = 70
    print(f"\n  SOURCE DEFINITION")
    print("  " + "-" * W)

    if isinstance(source, _BatchSource):
        print(f"  Type                           _BatchSource (pre-sampled)")
        print(f"  Neutrons in batch              {source.neutron_nbr:,}")
        return

    print(f"  {'Total neutrons':<30} {source.neutron_nbr:,}")
    print(f"  {'Source regions':<30} {len(source.source_regions)}")
    is_point = hasattr(source, "_point_position")
    if is_point:
        pos = source._point_position
        print(f"  {'Type':<30} Point source")
        print(f"  {'Position [cm]':<30} "
              f"x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
    for i, sr in enumerate(source.source_regions):
        print(f"\n  Region [{i}]  {sr.region_name}")
        print(f"    {'weight':<26} {source._norm_weights[i]:.4f}")
        print(f"    {'energy_dist':<26} {sr.energy_dist}")
        if sr.energy_dist == "log_normal":
            print(f"    {'mu':<26} {sr.mu}")
            print(f"    {'sigma':<26} {sr.sigma}")
        print(f"    {'energy_range [eV]':<26} "
              f"[{sr.energy_range[0]:.3e}, {sr.energy_range[1]:.3e}]")
        print(f"    {'direction_dist':<26} {sr.direction_dist}")
        if sr.direction_dist == "forward":
            print(f"    {'direction':<26} {sr.direction}")


def print_source_summary(source):
    W = 70
    print("\n" + "=" * W)
    print("  SOURCE CONFIGURATION")
    print("=" * W)
    _print_source_summary(source)
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-batch statistics (console + CSV)
# ─────────────────────────────────────────────────────────────────────────────

def export_cross_batch_stats(batch_stats, geom,
                             print_to_console=True,
                             print_geometry_summary=False,
                             save_csv=False,
                             output_dir=None,
                             source=None):
    """
    Parameters
    ----------
    batch_stats      : dict returned by geom.run_batch() / _compute_batch_stats()
    geom             : Geometry instance
    print_to_console : bool
    save_csv         : bool
    output_dir       : str | None
    source           : Source | _BatchSource | None
    """
    eb       = np.array(batch_stats["flux"]["energy_bins"])
    n_groups = len(eb) - 1
    group_labels = [f"{eb[i]:.2e}-{eb[i+1]:.2e} eV" for i in range(n_groups)]

    flux_mean = _arr(batch_stats, "flux", "mean")
    flux_std  = _arr(batch_stats, "flux", "std")
    flux_re   = _arr(batch_stats, "flux", "relative_error")

    abs_mean  = _arr(batch_stats, "verif", "absorption", "mean")
    abs_std   = _arr(batch_stats, "verif", "absorption", "std")
    abs_re    = _arr(batch_stats, "verif", "absorption", "relative_error")
    sxb       = np.array(batch_stats["verif"]["boundaries"])
    n_space   = len(sxb) - 1
    space_labels = [f"{sxb[i]:.1f}-{sxb[i+1]:.1f} cm" for i in range(n_space)]

    sct_mean  = _arr(batch_stats, "verif", "scatter", "mean")
    sct_std   = _arr(batch_stats, "verif", "scatter", "std")
    sct_re    = _arr(batch_stats, "verif", "scatter", "relative_error")

    sfx       = np.array(batch_stats["verif"]["surface_xs"])
    fwd_mean  = _arr(batch_stats, "verif", "current_fwd", "mean")
    fwd_std   = _arr(batch_stats, "verif", "current_fwd", "std")
    bwd_mean  = _arr(batch_stats, "verif", "current_bwd", "mean")
    bwd_std   = _arr(batch_stats, "verif", "current_bwd", "std")

    leak_l    = batch_stats["verif"]["leak_left"]
    leak_r    = batch_stats["verif"]["leak_right"]
    perf      = batch_stats["perf"]

    if print_to_console:
        if print_geometry_summary:
            print_geometry_summary(geom, source=source)

        print("\n" + "=" * 70)
        print("  CROSS-BATCH STATISTICS")
        print("=" * 70)

        # ── Flux ──────────────────────────────────────────────────────────────
        print("\n  FLUX TALLY [cm · src-n⁻¹]")
        print(f"  {'Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-" * 65)
        for i in range(n_groups):
            print(f"  {group_labels[i]:<25} {flux_mean[i]:>14.4e} "
                  f"{flux_std[i]:>14.4e} {flux_re[i]:>10.4f}")

        # ── Absorption ────────────────────────────────────────────────────────
        print("\n  ABSORPTION RATE [reactions · src-n⁻¹]")
        print(f"  {'Region / Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-" * 65)
        for si in range(n_space):
            for gi in range(n_groups):
                label = f"{space_labels[si]} | {group_labels[gi]}"
                print(f"  {label:<25} {abs_mean[si][gi]:>14.4e} "
                      f"{abs_std[si][gi]:>14.4e} {abs_re[si][gi]:>10.4f}")

        # ── Scatter ───────────────────────────────────────────────────────────
        print("\n  SCATTER RATE [reactions · src-n⁻¹]")
        print(f"  {'Region / Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-" * 65)
        for si in range(n_space):
            for gi in range(n_groups):
                label = f"{space_labels[si]} | {group_labels[gi]}"
                print(f"  {label:<25} {sct_mean[si][gi]:>14.4e} "
                      f"{sct_std[si][gi]:>14.4e} {sct_re[si][gi]:>10.4f}")

        # ── Surface currents ──────────────────────────────────────────────────
        print("\n  SURFACE CURRENTS [particles · src-n⁻¹]")
        print(f"  {'Surface':>10} {'Fwd Mean':>14} {'±Std':>12} "
              f"{'Bwd Mean':>14} {'±Std':>12} {'Net Mean':>14}")
        print("  " + "-" * 80)
        for si, sx in enumerate(sfx):
            net = fwd_mean[si] - bwd_mean[si]
            print(f"  {sx:>10.2f} cm  {fwd_mean[si]:>14.4e} {fwd_std[si]:>12.4e} "
                  f"{bwd_mean[si]:>14.4e} {bwd_std[si]:>12.4e} {net:>14.4e}")

        # ── Leakage ───────────────────────────────────────────────────────────
        print("\n  LEAKAGE [particles · src-n⁻¹]")
        print(f"  {'Quantity':<20} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-" * 60)
        print(f"  {'Left':<20} {leak_l['mean']:>14.4e} "
              f"{leak_l['std']:>14.4e} {leak_l['relative_error']:>10.4f}")
        print(f"  {'Right':<20} {leak_r['mean']:>14.4e} "
              f"{leak_r['std']:>14.4e} {leak_r['relative_error']:>10.4f}")
        print(f"  {'Total':<20} {leak_l['mean'] + leak_r['mean']:>14.4e}")

        # ── Performance: timing ───────────────────────────────────────────────
        print("\n  PERFORMANCE (cross-batch mean ± std)")
        print(f"  {'Metric':<42} {'Mean':>14} {'±Std':>14}")
        print("  " + "-" * 72)

        for key in ("time_preprocessing_s", "time_run_source_s", "time_total_s",
                    "neutrons_per_second", "rejection_fraction", "cpu_efficiency"):
            if key not in perf:
                continue
            m, s = _fmean(perf, key)
            print(f"  {key:<42} {m:>14} {s:>14}")

        print("  " + "-" * 72)

        for key in ("time_majorant_s", "time_majorant_wmp_s", "time_majorant_endf_s"):
            if key not in perf:
                continue
            lbl = ("  -> WMP  " if "wmp" in key else
                   "  -> ENDF " if "endf" in key else key)
            m, s = _fmean(perf, key)
            print(f"  {lbl:<42} {m:>14} {s:>14}")

        for key in ("time_xs_eval_s", "time_xs_eval_wmp_s", "time_xs_eval_endf_s"):
            if key not in perf:
                continue
            lbl = ("  -> WMP  " if "wmp" in key else
                   "  -> ENDF " if "endf" in key else key)
            m, s = _fmean(perf, key)
            print(f"  {lbl:<42} {m:>14} {s:>14}")

        # ── Performance: µs per operation ─────────────────────────────────────
        print("  " + "-" * 72)
        print(f"  {'µs per operation':<42} {'Mean (µs)':>14} {'±Std':>14}")
        print("  " + "-" * 72)
        for key in ("us_per_majorant_s", "us_per_majorant_wmp_s",
                    "us_per_majorant_endf_s",
                    "us_per_xs_eval_s", "us_per_xs_eval_wmp_s",
                    "us_per_xs_eval_endf_s"):
            if key not in perf:
                continue
            lbl = key.replace("_s", "").replace("us_per_", "µs/")
            lbl = ("  -> WMP  " if "wmp" in key else
                   "  -> ENDF " if "endf" in key else lbl)
            m, s = _fmean(perf, key, fmt=".4f")
            print(f"  {lbl:<42} {m:>14} {s:>14}")

        # ── Performance: memory ───────────────────────────────────────────────
        print("  " + "-" * 72)
        print(f"  {'Memory':<42} {'Mean (MB/kB)':>14} {'±Std':>14}")
        print("  " + "-" * 72)
        for key, unit in (
            ("baseline_rss_mb",         "MB"),
            ("setup_mem_mb",            "MB"),
            ("peak_mem_mb",             "MB"),
            ("python_heap_setup_kb",    "kB"),
            ("python_heap_transport_kb","kB"),
        ):
            if key not in perf:
                continue
            lbl = f"{key.replace('_mb','').replace('_kb','')} ({unit})"
            m, s = _fmean(perf, key, fmt=".1f")
            print(f"  {lbl:<42} {m:>14} {s:>14}")

        # ── Performance: event counts ─────────────────────────────────────────
        print("  " + "-" * 72)
        print(f"  {'n_neutrons (total)':<42} {_fint(perf,'n_neutrons'):>14}")
        for label, key in (
            ("n_real_collisions",        "n_real_collisions"),
            ("  -> WMP",                 "n_real_collisions_wmp"),
            ("  -> ENDF",                "n_real_collisions_endf"),
            ("n_virtual_collisions",     "n_virtual_collisions"),
            ("  -> WMP",                 "n_virtual_collisions_wmp"),
            ("  -> ENDF",                "n_virtual_collisions_endf"),
            ("n_xs_evaluations",         "n_xs_evaluations"),
            ("  -> WMP",                 "n_xs_evaluations_wmp"),
            ("  -> ENDF",                "n_xs_evaluations_endf"),
            ("n_majorant_updates",       "n_majorant_updates"),
            ("  -> WMP",                 "n_majorant_updates_wmp"),
            ("  -> ENDF",                "n_majorant_updates_endf"),
        ):
            if key not in perf:
                continue
            print(f"  {label:<42} {_fint(perf, key):>14}")

        # ── Performance: Energy Bins ──────────────────────────────────────────
        if "energy_bins" in perf and perf["energy_bins"]:
            print("\n  PERFORMANCE BY ENERGY BIN (cross-batch mean)")
            print(f"  {'Energy Bin':<24} {'n_maj':>12} {'µs/maj':>9} {'n_xs':>12} {'µs/xs':>9} {'Rej %':>8} {'WMaj %':>8}")
            print("  " + "-" * 86)
            for b in perf["energy_bins"]:
                lbl = b.get("label", "Unknown")
                
                def _sint(k):
                    if k in b and isinstance(b[k], dict) and math.isfinite(b[k].get("mean", float("nan"))):
                        return f"{int(b[k]['mean']):,}"
                    return "N/A"

                n_maj = _sint("n_majorant_updates")
                n_xs  = _sint("n_xs_evaluations")
                
                us_maj = b.get("us_per_majorant", float("nan"))
                us_maj_str = f"{us_maj:.2f}" if math.isfinite(us_maj) else "N/A"
                
                us_xs = b.get("us_per_xs_eval", float("nan"))
                us_xs_str = f"{us_xs:.2f}" if math.isfinite(us_xs) else "N/A"
                
                rej = b.get("rejection_fraction", float("nan"))
                rej_str = f"{rej*100:.1f}" if math.isfinite(rej) else "N/A"
                
                wmaj = b.get("wrong_majorant_fraction", float("nan"))
                wmaj_str = f"{wmaj*100:.2f}" if math.isfinite(wmaj) else "N/A"
                
                print(f"  {lbl:<24} {n_maj:>12} {us_maj_str:>9} {n_xs:>12} {us_xs_str:>9} {rej_str:>8} {wmaj_str:>8}")
            print("  " + "-" * 86)

        # ── Wrong majorant ────────────────────────────────────────────────────
        print("\n  WRONG MAJORANT STATISTICS")
        print(f"  {'Metric':<38} {'Mean':>14} {'±Std':>14} {'Min':>10} {'Max':>10}")
        print("  " + "-" * 80)
        for label, k in (
            ("wrong_majorant_fraction",       "wrong_majorant_fraction"),
            ("  -> WMP",                      "wrong_majorant_fraction_wmp"),
            ("  -> ENDF",                     "wrong_majorant_fraction_endf"),
            ("wrong_majorant_mean_error",      "wrong_majorant_mean_error"),
            ("  -> WMP",                      "wrong_majorant_mean_error_wmp"),
            ("  -> ENDF",                     "wrong_majorant_mean_error_endf"),
        ):
            if k not in perf:
                continue
            d = perf[k]
            if isinstance(d, dict):
                m   = d.get("mean", float("nan"))
                s   = d.get("std",  float("nan"))
                mn  = d.get("min",  float("nan"))
                mx  = d.get("max",  float("nan"))
                fmt = lambda v: f"{v:.4e}" if math.isfinite(v) else "N/A"
                print(f"  {label:<38} {fmt(m):>14} {fmt(s):>14} "
                      f"{fmt(mn):>10} {fmt(mx):>10}")

        for label, k in (
            ("n_wrong_majorant (total)", "n_wrong_majorant"),
            ("  -> WMP",                "n_wrong_majorant_wmp"),
            ("  -> ENDF",               "n_wrong_majorant_endf"),
        ):
            if k not in perf:
                continue
            print(f"  {label:<38} {_fint(perf, k):>14}")

        print("=" * 70)

    # ── CSV export ────────────────────────────────────────────────────────────
    if save_csv:
        base = Path(output_dir) if output_dir is not None else Path(".")
        if output_dir is not None:
            base.mkdir(parents=True, exist_ok=True)

        cross_rows = []

        def _row(tally, region, group, mean, std=float("nan"), re=float("nan")):
            cross_rows.append({"tally": tally, "region": region,
                                "energy_group": group,
                                "mean": mean, "std": std,
                                "relative_error": re})

        # physics tallies
        for i in range(n_groups):
            _row("flux", "all", group_labels[i],
                 flux_mean[i], flux_std[i], flux_re[i])
        for si in range(n_space):
            for gi in range(n_groups):
                for tally, m, s, r in (
                    ("absorption", abs_mean[si][gi], abs_std[si][gi], abs_re[si][gi]),
                    ("scatter",    sct_mean[si][gi], sct_std[si][gi], sct_re[si][gi]),
                ):
                    _row(tally, space_labels[si], group_labels[gi], m, s, r)
        for si, sx in enumerate(sfx):
            for dirn, m, s in (("forward",  fwd_mean[si], fwd_std[si]),
                               ("backward", bwd_mean[si], bwd_std[si])):
                re = s / abs(m) if m != 0 else float("inf")
                _row(f"current_{dirn}", f"x={sx:.2f} cm", "all", m, s, re)
        for side, d in (("leak_left", leak_l), ("leak_right", leak_r)):
            _row(side, "boundary", "all",
                 d["mean"], d["std"], d["relative_error"])

        # performance: float-stat keys (mean ± std)
        for key in (
            "time_preprocessing_s", "time_run_source_s", "time_total_s",
            "neutrons_per_second", "rejection_fraction",
            "rejection_fraction_wmp", "rejection_fraction_endf",
            "cpu_efficiency",
            "time_majorant_s",     "time_majorant_wmp_s",    "time_majorant_endf_s",
            "time_xs_eval_s",      "time_xs_eval_wmp_s",     "time_xs_eval_endf_s",
            # new: µs per operation
            "us_per_majorant_s",   "us_per_majorant_wmp_s",  "us_per_majorant_endf_s",
            "us_per_xs_eval_s",    "us_per_xs_eval_wmp_s",   "us_per_xs_eval_endf_s",
            # new: memory
            "baseline_rss_mb", "setup_mem_mb", "peak_mem_mb",
            "python_heap_setup_kb", "python_heap_transport_kb",
        ):
            if key not in perf:
                continue
            v = perf[key]
            if isinstance(v, dict):
                _row(key, "performance", "all", v["mean"], v["std"])
            else:
                _row(key, "performance", "all", float(v))

        # performance: integer keys (plain int in perf dict)
        for key in (
            "n_neutrons",
            "n_real_collisions",    "n_real_collisions_wmp",    "n_real_collisions_endf",
            "n_virtual_collisions", "n_virtual_collisions_wmp", "n_virtual_collisions_endf",
            "n_xs_evaluations",     "n_xs_evaluations_wmp",     "n_xs_evaluations_endf",
            "n_majorant_updates",   "n_majorant_updates_wmp",   "n_majorant_updates_endf",
            "n_wrong_majorant",     "n_wrong_majorant_wmp",     "n_wrong_majorant_endf",
        ):
            if key not in perf:
                continue
            v = perf[key]
            # guard: if it is a dict (float-key collision) take mean
            mean_val = v["mean"] if isinstance(v, dict) else int(v)
            _row(key, "performance", "all", mean_val)

        # wrong-majorant fraction / error (mean ± std + min/max)
        for k in (
            "wrong_majorant_fraction",      "wrong_majorant_mean_error",
            "wrong_majorant_fraction_wmp",  "wrong_majorant_mean_error_wmp",
            "wrong_majorant_fraction_endf", "wrong_majorant_mean_error_endf",
        ):
            if k not in perf:
                continue
            d = perf[k]
            if not isinstance(d, dict):
                continue
            re = (d["std"] / abs(d["mean"])
                  if d.get("mean") and d["mean"] != 0 else float("inf"))
            _row(k, "performance", "all", d["mean"], d["std"], re)

        # per-event timing derived from float mean / int count
        for label, t_key, n_key in (
            ("time_per_majorant_update_ms",      "time_majorant_s",      "n_majorant_updates"),
            ("time_per_majorant_update_wmp_ms",  "time_majorant_wmp_s",  "n_majorant_updates_wmp"),
            ("time_per_majorant_update_endf_ms", "time_majorant_endf_s", "n_majorant_updates_endf"),
            ("time_per_xs_eval_ms",              "time_xs_eval_s",       "n_xs_evaluations"),
            ("time_per_xs_eval_wmp_ms",          "time_xs_eval_wmp_s",   "n_xs_evaluations_wmp"),
            ("time_per_xs_eval_endf_ms",         "time_xs_eval_endf_s",  "n_xs_evaluations_endf"),
        ):
            n = perf.get(n_key, 0)
            n = n["mean"] if isinstance(n, dict) else n
            if n and n > 0 and t_key in perf:
                t = perf[t_key]
                t_mean = t["mean"] if isinstance(t, dict) else float(t)
                _row(label, "performance", "all", 1000 * t_mean / n)

        # Save main cross-batch CSV
        cross_path = base / "cross_batch_statistics.csv"
        with open(cross_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tally", "region", "energy_group",
                                              "mean", "std", "relative_error"])
            w.writeheader()
            w.writerows(cross_rows)
        print(f"\nCross-batch statistics saved  → {cross_path}")

        # ── Separate CSV export for Energy Bins ───────────────────────────────
        if "energy_bins" in perf and perf["energy_bins"]:
            bin_rows = []
            for b in perf["energy_bins"]:
                lbl = b.get("label", "unknown")
                
                # Nested dicts (mean/std)
                for k in ("time_majorant_s", "cpu_time_majorant_s", "n_majorant_updates",
                          "time_xs_eval_s", "cpu_time_xs_eval_s", "n_xs_evaluations",
                          "n_real_collisions", "n_virtual_collisions",
                          "n_wrong_majorant", "wrong_majorant_error"):
                    if k in b and isinstance(b[k], dict):
                        bin_rows.append({
                            "energy_bin_label": lbl,
                            "metric": k,
                            "mean": b[k].get("mean", float("nan")),
                            "std": b[k].get("std", float("nan"))
                        })
                
                # Single floats (derived calculations)
                for k in ("rejection_fraction", "wrong_majorant_fraction", 
                          "wrong_majorant_mean_error", "us_per_majorant", "us_per_xs_eval"):
                    if k in b and isinstance(b[k], float):
                        bin_rows.append({
                            "energy_bin_label": lbl,
                            "metric": k,
                            "mean": b[k],
                            "std": float("nan")
                        })
                        
            bin_path = base / "energy_bins_statistics.csv"
            with open(bin_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["energy_bin_label", "metric", "mean", "std"])
                w.writeheader()
                w.writerows(bin_rows)
            print(f"Energy bins statistics saved  → {bin_path}")

        # geometry summary CSV
        _save_geometry_csv(geom, base)

        # source summary CSV
        if source is not None:
            _save_source_csv(source, base)


# ─────────────────────────────────────────────────────────────────────────────
# Inner-batch CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_inner_batch_stats_csv(batch_stats, geom, output_dir=None):
    eb       = np.array(batch_stats["flux"]["energy_bins"])
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

        if "flux" in snap:
            for i in range(n_groups):
                inner_rows.append({
                    "batch": b, "n_neutrons": b_n, "wall_time_s": b_time,
                    "tally": "flux", "region": "all",
                    "energy_group": group_labels[i],
                    "mean":  snap["flux"]["flux"]["mean"][i],
                    "std":   snap["flux"]["flux"]["std"][i],
                    "relative_error": snap["flux"]["flux"]["relative_error"][i],
                })

        if "verif" in snap:
            vsnap = snap["verif"]
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
            for direction, key in (("forward", "current_fwd"),
                                   ("backward", "current_bwd")):
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

    inner_path = base / "inner_batch_statistics.csv"
    with open(inner_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "n_neutrons", "wall_time_s",
                                          "tally", "region", "energy_group",
                                          "mean", "std", "relative_error"])
        w.writeheader()
        w.writerows(inner_rows)
    print(f"Inner-batch statistics saved  → {inner_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_geometry_csv(geom, base: Path):
    rows = []
    for r in geom.get_regions():
        rows.append({
            "type": "region", "name": r.name, "material": r.material.name,
            "x_min_cm": r.x_min, "x_max_cm": r.x_max,
            "width_cm": r.x_max - r.x_min, "nuclide": "",
            "density_at_per_cm3": "", "temperature_K": r.material.T, "detail": "",
        })
        for nuclide_name, density in r.material.nuclides:
            rows.append({
                "type": "nuclide", "name": r.name, "material": r.material.name,
                "x_min_cm": r.x_min, "x_max_cm": r.x_max,
                "width_cm": r.x_max - r.x_min, "nuclide": nuclide_name,
                "density_at_per_cm3": density, "temperature_K": r.material.T, "detail": "",
            })
    for key, val in [
        ("mode",              geom.mode),
        ("maj_xs_method",     geom.maj_xs_method),
        ("maj_mat_method",    geom.maj_mat_method),
        ("access_method",     geom.access_method),
        ("cutoff_energy",     geom.cutoff_energy),
        ("bc_left",           geom.boundary_conditions.get("left",  "vacuum")),
        ("bc_right",          geom.boundary_conditions.get("right", "vacuum")),
        ("flux_tally",        geom.flux_tally_flag),
        ("verif_tally",       geom.verif_tally_flag),
        ("perf_tracker",      geom.perf_tracker_flag),
        ("majorant_log",      geom.majorant_log_flag),
        ("history_flag",      geom.history_flag),
        ("memory_flag",       geom.memory_tracker_flag),
        ("tracemalloc",       geom.perf._tracemalloc_enabled),
    ]:
        rows.append({
            "type": "setting", "name": key, "material": "",
            "x_min_cm": "", "x_max_cm": "", "width_cm": "",
            "nuclide": "", "density_at_per_cm3": "",
            "temperature_K": "", "detail": str(val),
        })

    path = base / "geometry_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "type", "name", "material", "x_min_cm", "x_max_cm", "width_cm",
            "nuclide", "density_at_per_cm3", "temperature_K", "detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"Geometry summary saved        → {path}")


def _save_source_csv(source, base: Path):
    if isinstance(source, _BatchSource):
        rows = [{"region_index": 0, "region_name": "batch",
                 "norm_weight": 1.0, "energy_dist": "pre-sampled",
                 "energy_min_eV": "", "energy_max_eV": "",
                 "mu": "", "sigma": "", "direction_dist": "pre-sampled",
                 "direction": "", "point_position": "",
                 "total_neutrons": source.neutron_nbr}]
    else:
        is_point = hasattr(source, "_point_position")
        pos_str  = (f"[{source._point_position[0]:.4f},"
                    f"{source._point_position[1]:.4f},"
                    f"{source._point_position[2]:.4f}]") if is_point else ""
        rows = []
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

    path = base / "source_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "region_index", "region_name", "norm_weight",
            "energy_dist", "energy_min_eV", "energy_max_eV",
            "mu", "sigma", "direction_dist", "direction",
            "point_position", "total_neutrons"])
        w.writeheader()
        w.writerows(rows)
    print(f"Source summary saved          → {path}")