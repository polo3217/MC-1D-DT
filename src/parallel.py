"""
parallel.py
===========

## AI Genereted : still to be checked
# It was tested and it works. Not changes from initial AI version


Batch-level parallelism for Monte Carlo neutron transport.

Usage
-----
from src.parallel import run_batch_parallel

batch_stats = run_batch_parallel(geom, src, n_batches=20, n_workers=4)
"""

import pickle
from multiprocessing import Pool
from src.source_class import Source
import psutil
import os


def _run_single_batch(args):
    geom_pickle, neutron_nbr, energy_range, energy_dist, position, direction, track_neutron = args

    geom = pickle.loads(geom_pickle)
    src  = Source(
        neutron_nbr  = neutron_nbr,
        energy_range = energy_range,
        energy_dist  = energy_dist,
        position     = position,
        direction    = direction,
    )

    geom.reset()
    geom.run_source(src, track_neutron=track_neutron)

    # ── Report worker peak memory ──────────────────────────────────────
    process   = psutil.Process(os.getpid())
    peak_mb   = process.memory_info().rss / 1e6

    snap = {
        "perf"     : geom.perf.snapshot(),
        "peak_mb"  : peak_mb,              # ← worker peak RSS
    }

    if geom.flux_tally_flag and geom.flux_tally is not None:
        snap["flux"] = {
            "energy_bins": geom.flux_tally.energy_bins.tolist(),
            "flux": {
                "mean"          : geom.flux_tally.mean.tolist(),
                "std"           : geom.flux_tally.std.tolist(),
                "relative_error": geom.flux_tally.relative_error.tolist(),
            }
        }

    if geom.verif_tally_flag and geom.verif_tally is not None:
        snap["verif"] = geom.verif_tally.snapshot()

    return snap


def run_batch_parallel(geom, src, n_batches, n_workers=None, track_neutron=False):
    """
    Parallel drop-in replacement for geom.run_batch().

    Parameters
    ----------
    geom       : Geometry   — fully configured (reconr grid already built)
    src        : Source     — used only to extract distribution parameters
    n_batches  : int        — number of independent batches
    n_workers  : int|None   — number of worker processes (None = cpu_count)
    track_neutron : bool    — passed through to run_source

    Returns
    -------
    batch_stats : dict      — same format as geom.run_batch()
    """
    if n_batches < 2:
        raise ValueError("n_batches must be >= 2")

    # Pickle the geometry once — workers each deserialize their own copy
    # The reconr grid, vectfit tables, and all settings are captured here
    geom_pickle = pickle.dumps(geom)

    args = [
        (
            geom_pickle,
            src.neutron_nbr,
            src.energy_range,
            src.energy_dist,
            src.position,
            src.direction,
            track_neutron,
        )
        for _ in range(n_batches)
    ]

    # [CHANGED] Added parallel prefix for better log reading
    print(f"\n[Parallel] Running {n_batches} batches on {n_workers or 'all'} workers...")
    with Pool(processes=n_workers) as pool:
        geom.batch_results = pool.map(_run_single_batch, args)

    geom.batch_stats = geom._compute_batch_stats()
    return geom.batch_stats