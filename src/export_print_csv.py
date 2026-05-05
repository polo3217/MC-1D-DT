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
# Print and or save all cross-batch statistics
# ============================================================================

def export_cross_batch_stats(batch_stats, geom, 
                             print_to_console=True, 
                             save_csv=False, 
                             output_dir=None, ):
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

    # performance
    perf = batch_stats["perf"]

    

    
    if print_to_console:

        print("\n" + "="*70)
        print("  CROSS-BATCH STATISTICS")
        print("="*70)


        print("\n  FLUX TALLY [cm · src-n⁻¹]")
        print(f"  {'Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*65)
        for i in range(n_groups):
            print(f"  {group_labels[i]:<25} {flux_mean[i]:>14.4e} "
                f"{flux_std[i]:>14.4e} {flux_re[i]:>10.4f}")

        

        print("\n  ABSORPTION RATE [reactions · src-n⁻¹]")
        print(f"  {'Region / Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*65)
        for si in range(n_space):
            for gi in range(n_groups):
                label = f"{space_labels[si]} | {group_labels[gi]}"
                print(f"  {label:<25} {abs_mean[si][gi]:>14.4e} "
                    f"{abs_std[si][gi]:>14.4e} {abs_re[si][gi]:>10.4f}")

        

        print("\n  SCATTER RATE [reactions · src-n⁻¹]")
        print(f"  {'Region / Group':<25} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*65)
        for si in range(n_space):
            for gi in range(n_groups):
                label = f"{space_labels[si]} | {group_labels[gi]}"
                print(f"  {label:<25} {sct_mean[si][gi]:>14.4e} "
                    f"{sct_std[si][gi]:>14.4e} {sct_re[si][gi]:>10.4f}")



        print("\n  SURFACE CURRENTS [particles · src-n⁻¹]")
        print(f"  {'Surface':>10} {'Fwd Mean':>14} {'±Std':>12} "
            f"{'Bwd Mean':>14} {'±Std':>12} {'Net Mean':>14}")
        print("  " + "-"*80)
        for si, sx in enumerate(sfx):
            net = fwd_mean[si] - bwd_mean[si]
            print(f"  {sx:>10.2f} cm  {fwd_mean[si]:>14.4e} {fwd_std[si]:>12.4e} "
                f"{bwd_mean[si]:>14.4e} {bwd_std[si]:>12.4e} {net:>14.4e}")



        print("\n  LEAKAGE [particles · src-n⁻¹]")
        print(f"  {'Quantity':<20} {'Mean':>14} {'±Std':>14} {'Rel.Err':>10}")
        print("  " + "-"*60)
        print(f"  {'Left  (x=0.0 cm)':<20} {leak_l['mean']:>14.4e} "
            f"{leak_l['std']:>14.4e} {leak_l['relative_error']:>10.4f}")
        print(f"  {'Right (x=15.0 cm)':<20} {leak_r['mean']:>14.4e} "
            f"{leak_r['std']:>14.4e} {leak_r['relative_error']:>10.4f}")
        total_mean = leak_l['mean'] + leak_r['mean']
        print(f"  {'Total':<20} {total_mean:>14.4e}")

        # ── Performance ───────────────────────────────────────────────────────────
        perf = batch_stats["perf"]
        print("\n  PERFORMANCE (cross-batch)")
        print(f"  {'Metric':<30} {'Mean':>14} {'±Std':>14}")
        print("  " + "-"*60)
        for key in (
            "time_preprocessing_s", "time_run_source_s", "time_total_s",
            "neutrons_per_second", "rejection_fraction", "cpu_efficiency",
            "time_majorant_s", "time_xs_eval_s",
        ):
            if key not in perf:
                continue
            print(f"  {key:<30} {perf[key]['mean']:>14.4f} {perf[key]['std']:>14.4f}")
        print(f"  {'n_neutrons (total)':<30} {perf['n_neutrons']:>14,}")
        print(f"  {'n_real_collisions (total)':<30} {perf['n_real_collisions']:>14,}")
        print(f"  {'n_virtual_collisions (total)':<30} {perf['n_virtual_collisions']:>14,}")
        print("="*70)

        print("\n  WRONG MAJORANT STATISTICS")
        print(f"  {'Metric':<30} {'Mean':>14} {'±Std':>14} {'Min':>10} {'Max':>10}")
        print("  " + "-"*80)
        for k in ("wrong_majorant_fraction", "wrong_majorant_mean_error"):
            d = perf[k]
            print(f"  {k:<30} {d['mean']:>14.4e} {d['std']:>14.4e} "
                  f"{d['min']:>10.4e} {d['max']:>10.4e}")
        print(f"  {'n_wrong_majorant (total)':<30} {perf['n_wrong_majorant']:>14,}")

# ============================================================================
# Save cross-batch statistics CSV
# ============================================================================


    if save_csv:
        cross_rows = []

        # flux
        for i in range(n_groups):
            cross_rows.append({
                "tally"          : "flux",
                "region"         : "all",
                "energy_group"   : group_labels[i],
                "mean"           : flux_mean[i],
                "std"            : flux_std[i],
                "relative_error" : flux_re[i],
            })

        # absorption
        for si in range(n_space):
            for gi in range(n_groups):
                cross_rows.append({
                    "tally"          : "absorption",
                    "region"         : space_labels[si],
                    "energy_group"   : group_labels[gi],
                    "mean"           : abs_mean[si][gi],
                    "std"            : abs_std[si][gi],
                    "relative_error" : abs_re[si][gi],
                })

        # scatter
        for si in range(n_space):
            for gi in range(n_groups):
                cross_rows.append({
                    "tally"          : "scatter",
                    "region"         : space_labels[si],
                    "energy_group"   : group_labels[gi],
                    "mean"           : sct_mean[si][gi],
                    "std"            : sct_std[si][gi],
                    "relative_error" : sct_re[si][gi],
                })

        # surface currents
        for si, sx in enumerate(sfx):
            for direction, m, s in [("forward",  fwd_mean[si], fwd_std[si]),
                                    ("backward", bwd_mean[si], bwd_std[si])]:
                cross_rows.append({
                    "tally"          : f"current_{direction}",
                    "region"         : f"x={sx:.2f} cm",
                    "energy_group"   : "all",
                    "mean"           : m,
                    "std"            : s,
                    "relative_error" : s / abs(m) if m != 0 else float("inf"),
                })

        # leakage
        for side, d in [("leak_left", leak_l), ("leak_right", leak_r)]:
            cross_rows.append({
                "tally"          : side,
                "region"         : "boundary",
                "energy_group"   : "all",
                "mean"           : d["mean"],
                "std"            : d["std"],
                "relative_error" : d["relative_error"],
            })

        # performance
        perf = batch_stats["perf"]
        for key in (
            "time_preprocessing_s",
            "time_run_source_s",
            "time_total_s",
            "neutrons_per_second",
            "rejection_fraction",
            "cpu_efficiency",
            "time_majorant_s",
            "time_xs_eval_s",
        ):
            if key not in perf:
                continue   # graceful fallback for older batch_stats dicts
            cross_rows.append({
                "tally"          : key,
                "region"         : "performance",
                "energy_group"   : "all",
                "mean"           : perf[key]['mean'],
                "std"            : perf[key]['std'],
                "relative_error" : float("nan"),
            })
        cross_rows.append({
            "tally"          : "n_real_collisions (total)",
            "region"         : "performance",
            "energy_group"   : "all",
            "mean"           : perf['n_real_collisions'],
            "std"            : float("nan"),
            "relative_error" : float("nan"),
        })
        cross_rows.append({
            "tally"          : "n_virtual_collisions (total)",
            "region"         : "performance",
            "energy_group"   : "all",
            "mean"           : perf['n_virtual_collisions'],
            "std"            : float("nan"),
            "relative_error" : float("nan"),
        })


        # In the CSV block, alongside the other perf keys:
        for k in ("wrong_majorant_fraction", "wrong_majorant_mean_error"):
                    cross_rows.append({
                        "tally"          : k,
                        "region"         : "performance",
                        "energy_group"   : "all",
                        "mean"           : perf[k]["mean"],
                        "std"            : perf[k]["std"],
                        "relative_error" : perf[k]["std"] / abs(perf[k]["mean"])
                                        if perf[k]["mean"] != 0 else float("inf"),
                    })
        cross_rows.append({
                    "tally"          : "n_wrong_majorant (total)",
                    "region"         : "performance",
                    "energy_group"   : "all",
                    "mean"           : perf["n_wrong_majorant"],
                    "std"            : float("nan"),
                    "relative_error" : float("nan"),
                })

        if output_dir == None:
            cross_path = "cross_batch_statistics_corrected.csv"
        else: 
            cross_path = output_dir + "/cross_batch_statistics_corrected.csv"
        with open(cross_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tally", "region", "energy_group",
                                            "mean", "std", "relative_error"])
            w.writeheader()
            w.writerows(cross_rows)
        print(f"\nCross-batch statistics saved → {cross_path}")

  

def export_inner_batch_stats_csv(batch_stats, geom, output_dir=None):
    eb = np.array(batch_stats["flux"]["energy_bins"])
    n_groups = len(eb) - 1
    group_labels = [f"{eb[i]:.0f}-{eb[i+1]:.0f} eV" for i in range(n_groups)]
    sxb      = np.array(geom.boundaries)
    n_space  = len(sxb) - 1
    space_labels = [f"{sxb[i]:.1f}-{sxb[i+1]:.1f} cm" for i in range(n_space)]
    sfx      = np.array(batch_stats["verif"]["surface_xs"])

    # ============================================================================
    # Save inner-batch (per-batch) statistics CSV
    # ============================================================================
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
                    "batch"          : b,
                    "n_neutrons"     : b_n,
                    "wall_time_s"    : b_time,
                    "tally"          : "flux",
                    "region"         : "all",
                    "energy_group"   : group_labels[i],
                    "mean"           : b_flux_mean[i],
                    "std"            : b_flux_std[i],
                    "relative_error" : b_flux_re[i],
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
                            "batch"          : b,
                            "n_neutrons"     : b_n,
                            "wall_time_s"    : b_time,
                            "tally"          : tally_key,
                            "region"         : space_labels[si],
                            "energy_group"   : group_labels[gi],
                            "mean"           : t_mean[si][gi],
                            "std"            : t_std[si][gi],
                            "relative_error" : t_re[si][gi],
                        })

            # surface currents
            for direction, key in [("forward", "current_fwd"),
                                    ("backward", "current_bwd")]:
                c_mean = np.array(vsnap[key]["mean"])
                c_std  = np.array(vsnap[key]["std"])
                c_re   = np.array(vsnap[key]["relative_error"])
                for si, sx in enumerate(sfx):
                    inner_rows.append({
                        "batch"          : b,
                        "n_neutrons"     : b_n,
                        "wall_time_s"    : b_time,
                        "tally"          : f"current_{direction}",
                        "region"         : f"x={sx:.2f} cm",
                        "energy_group"   : "all",
                        "mean"           : c_mean[si],
                        "std"            : c_std[si],
                        "relative_error" : c_re[si],
                    })

            # leakage
            for side in ("leak_left", "leak_right"):
                ld = vsnap[side]
                inner_rows.append({
                    "batch"          : b,
                    "n_neutrons"     : b_n,
                    "wall_time_s"    : b_time,
                    "tally"          : side,
                    "region"         : "boundary",
                    "energy_group"   : "all",
                    "mean"           : ld["mean"],
                    "std"            : ld["std"],
                    "relative_error" : ld["relative_error"],
                })
    if output_dir == None:
        inner_path = "inner_batch_statistics_corrected.csv"
    else:
        inner_path = output_dir + "/inner_batch_statistics_corrected.csv"
    with open(inner_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "n_neutrons", "wall_time_s",
                                        "tally", "region", "energy_group",
                                        "mean", "std", "relative_error"])
        w.writeheader()
        w.writerows(inner_rows)
    print(f"Inner-batch statistics saved  → {inner_path}")