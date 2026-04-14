"""
export_simulation.py
====================
Saves every piece of data stored in the simulation classes into an
organized folder structure.

--- CHANGES FROM ORIGINAL ---
[CHANGE] NumpyEncoder added to handle np.ndarray / np.integer / np.floating /
         np.bool_ that json.dump cannot serialise natively.  All json.dump
         calls now pass cls=NumpyEncoder.
[CHANGE] flux_data now includes per-bin std and relative_error from the
         TallyArray stored inside FluxTallyTLE.
[CHANGE] verif_data section added: exports absorption, scatter, current, and
         leakage means ± std from VerificationTally.
[NEW]    export_batch_stats() helper: writes batch_stats dict produced by
         geometry.run_batch() to a separate JSON file.
No existing keys have been removed so downstream dashboard code keeps working.
"""

import csv
import json
import numpy as np
from datetime import datetime
from pathlib import Path


# ==========================================
# [NEW] NumpyEncoder
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """
    Serialise NumPy types that the default JSONEncoder cannot handle.

    Covers:
        np.ndarray  → list  (via .tolist())
        np.integer  → int
        np.floating → float
        np.bool_    → bool
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _write_json(path: Path, data):
    # [CHANGE] pass cls=NumpyEncoder to handle ndarray / numpy scalar types
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

def _write_csv(path: Path, fieldnames: list, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def export_simulation(geom, src, output_dir: str = "sim_output"):
    """
    Export all simulation data to an organized folder.

    Parameters
    ----------
    geom        : geometry object (post run_source)
    src         : source object
    output_dir  : root folder name (created if absent)
    """
    root = Path(output_dir)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # ============================================================================
    # Export JSON for the diagnostic dashboard
    # ============================================================================
    histories = geom.histories

    # Per-neutron summary
    neutron_records = []
    for h in histories:
        neutron_records.append({
            "id"                : h.neutron_id,
            "birth_energy"      : h.birth_energy,
            "final_energy"      : h.final_energy,
            "fate"              : h.fate,
            "n_scatters"        : h.n_scatters,
            "n_virtual"         : h.n_virtual,
            "total_path_length" : h.total_path_length,
            "energy_loss_frac"  : h.energy_loss_fraction,
            "birth_x"           : h.birth_position[0],
            "final_x"           : h.positions[-1][0] if h.positions else 0,
        })

    # Majorant log (sample up to 2000 points to keep JSON small)
    maj_log = geom.majorant_log
    step = max(1, len(maj_log) // 2000)
    majorant_records = [
        {
            "energy"            : r.energy,
            "majorant_xs"       : r.value,
            "actual_max_xs"     : r.actual_max_xs,
            "margin"            : r.margin,
            "limiting_material" : r.limiting_material,
        }
        for r in maj_log[::step]
    ]

    # ── Flux tally ────────────────────────────────────────────────────────────
    ft = geom.flux_tally
    flux_data = {
        "spatial_bins"  : geom.boundaries,
        "energy_bins"   : ft.energy_bins.tolist() if ft is not None else [],
        # [CHANGE] mean now comes from TallyArray.mean (same values as before)
        "flux"          : ft.mean.tolist() if ft is not None else [],
        # [NEW] per-bin standard deviation of the mean
        "flux_std"      : ft.std.tolist() if ft is not None else [],
        # [NEW] per-bin relative error  σ/μ
        "flux_re"       : ft.relative_error.tolist() if ft is not None else [],
        "n_histories"   : ft._n_histories if ft is not None else 0,
    }

    # ── Verification tally ────────────────────────────────────────────────────
    # [NEW] export VerificationTally means and uncertainties
    vt = geom.verif_tally
    if vt is not None:
        verif_data = {
            "energy_bins"        : vt.energy_bins.tolist(),
            "surface_xs"         : vt.surface_xs.tolist(),
            # reaction rates
            "absorption"         : vt.absorption.tolist(),
            "absorption_std"     : vt.absorption_std.tolist(),
            "absorption_re"      : vt.absorption_re.tolist(),
            "scatter"            : vt.scatter.tolist(),
            "scatter_std"        : vt.scatter_std.tolist(),
            "scatter_re"         : vt.scatter_re.tolist(),
            # surface currents
            "current_fwd"        : vt.current_fwd.tolist(),
            "current_fwd_std"    : vt.current_fwd_std.tolist(),
            "current_bwd"        : vt.current_bwd.tolist(),
            "current_bwd_std"    : vt.current_bwd_std.tolist(),
            "current_net"        : vt.current_net.tolist(),
            # leakage
            "leak_left"          : vt.leak_left,
            "leak_left_std"      : vt.leak_left_std,
            "leak_left_re"       : vt.leak_left_re,
            "leak_right"         : vt.leak_right,
            "leak_right_std"     : vt.leak_right_std,
            "leak_right_re"      : vt.leak_right_re,
            "leak_total"         : vt.leak_total,
            "n_histories"        : vt._n_histories,
        }
    else:
        verif_data = {}

    # Performance
    p = geom.perf
    perf_data = {
        "n_neutrons"           : p.n_neutrons,
        "total_time_s"         : p.total_time,
        "neutrons_per_second"  : p.neutrons_per_second,
        "n_majorant_updates"   : p.n_majorant_updates,
        "n_real_collisions"    : p.n_real_collisions,
        "n_virtual_collisions" : p.n_virtual_collisions,
        "rejection_fraction"   : p.rejection_fraction,
        "time_xs_eval_s"       : p.time_xs_eval,
        "cpu_efficiency"       : p.cpu_efficiency,
    }

    # Global counters
    counters = {
        "absorption" : geom.absorption_score,
        "scatter"    : geom.scattering_score,
        "leakage"    : geom.leakage_score,
        "accepted"   : geom.acception_score,
        "rejected"   : geom.rejection_score,
    }

    output = {
        "meta"      : {
            "title"      : "5% Enriched Uranium — Two-Slab Delta Tracking",
            "slab1_name" : geom.materials[0].name,
            "slab2_name" : geom.materials[1].name,
            "boundaries" : geom.boundaries,
            "n_neutrons" : src.neutron_nbr,
            "timestamp"  : timestamp,
        },
        "perf"      : perf_data,
        "counters"  : counters,
        "neutrons"  : neutron_records,
        "majorant"  : majorant_records,
        "flux"      : flux_data,
        # [NEW] verification tally block (empty dict if tally not attached)
        "verif"     : verif_data,
    }

    out_path = "sim_results.json"
    with open(out_path, "w") as f:
        # [CHANGE] pass cls=NumpyEncoder so ndarray / numpy scalar types
        # are serialised correctly without having to call .tolist() everywhere.
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nJSON exported → {out_path}")


# ==========================================
# [NEW] export_batch_stats()
# ==========================================
def export_batch_stats(geom, src, out_path: str = "sim_batch_stats.json"):
    """
    Export the cross-batch statistics produced by geometry.run_batch().

    Must be called after geom.run_batch(); raises RuntimeError otherwise.

    Parameters
    ----------
    geom     : geometry object (post run_batch)
    src      : source object
    out_path : output JSON file path
    """
    if not hasattr(geom, "batch_stats") or not geom.batch_stats:
        raise RuntimeError(
            "No batch statistics found.  Call geom.run_batch() before "
            "export_batch_stats()."
        )

    output = {
        "meta": {
            "title"      : "Batch statistics — Delta Tracking",
            "slab1_name" : geom.materials[0].name,
            "slab2_name" : geom.materials[1].name,
            "boundaries" : geom.boundaries,
            "n_neutrons" : src.neutron_nbr,
            "n_batches"  : geom.batch_stats.get("n_batches"),
            "timestamp"  : datetime.now().isoformat(timespec="seconds"),
        },
        "batch_stats"   : geom.batch_stats,
        # also include the raw per-batch snapshots so the user can
        # inspect convergence batch-by-batch if needed.
        "batch_results" : geom.batch_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nBatch stats exported → {out_path}")