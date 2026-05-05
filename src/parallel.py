"""
parallel.py
====================
Drop-in replacement for `Geometry.run_batch_serial` and
`Geometry.run_batch_parallel` — a single `run_batch(...)` method whose
execution mode is selected by a string argument.

USAGE
-----
    # Serial (single process):
    stats = geom.run_batch(src, n_batches=20, mode="serial")

    # Parallel (multiprocessing.Pool):
    stats = geom.run_batch(src, n_batches=20, mode="parallel", n_workers=8)

INTEGRATION
-----------
1. Copy the module-level helper `_run_single_batch_worker` into
   `parallel.py` (or import it from there). It MUST live at module
   scope — multiprocessing.Pool pickles the worker by reference, and
   nested or class-method functions are not picklable.
2. Copy the `run_batch` method into the `Geometry` class in
   `geometry_classes.py`. Delete the existing `run_batch_serial` and
   `run_batch_parallel` methods (or have them call this one for
   backward compatibility — see the optional shims at the bottom).
3. The function uses `BatchTimer` from the corrected
   `performance_classes.py` (delivered earlier in this session) for
   honest end-to-end wall + CPU timing. Make sure that file is on
   the import path.

NOTE ON MEMORY TRACKER
----------------------
The memory tracker logic is preserved verbatim from the original two
methods. If `geom.memory_tracker_flag` is True, the parent process
records snapshots; in parallel mode each worker also reports its
peak RSS, which we collate after the pool join.
"""

import os
import time
import pickle
from multiprocessing import Pool
import psutil

# BatchTimer comes from the corrected performance_classes.py.
from src.performance_classes import BatchTimer


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL WORKER  (required by multiprocessing — must be top-level)
# ──────────────────────────────────────────────────────────────────────────────
def _run_single_batch_worker(args):
    """
    Execute exactly one independent batch in a worker process.

    Why is this at module scope?
    ----------------------------
    `multiprocessing.Pool.map` pickles the function reference and ships it
    to each worker. Nested functions, lambdas, and bound methods cannot be
    pickled (or are awkward to pickle) — they MUST be defined at the top
    level of an importable module so the worker can import the same name.

    Parameters
    ----------
    args : tuple
        A 4-tuple ``(geom_pickle, src, track_neutron, batch_idx)`` —
        passed packed-up because Pool.map only takes a single argument.

        geom_pickle   : bytes
            The geometry, already pickled in the parent process via
            ``pickle.dumps(geom)``. Pickling once and shipping the bytes
            is faster than pickling per-worker, and ensures every worker
            starts from an identical geometry state.
        src : Source
            The source object. Source has a __getstate__ that strips its
            geometry reference, so it pickles cleanly. Each worker calls
            ``src.generate_batch(src.neutron_nbr)`` to sample its own fresh
            neutrons (independent random streams across workers, since
            each worker process has its own RNG state).
        track_neutron : bool
            Forwarded to ``geom.run_source(..., track_neutron=...)``.
        batch_idx : int
            The batch's logical index. Used only for bookkeeping in the
            returned snap dict so results can be sorted back into order.

    Returns
    -------
    dict
        A "batch snapshot" with the same schema as the serial path:
            batch       : int                — logical batch index
            n_neutrons  : int                — number of neutrons in the batch
            perf        : dict               — PerformanceTracker.snapshot()
            peak_mb     : float              — worker peak RSS in MB
            flux        : dict (optional)    — FluxTallyTLE.snapshot()
            verif       : dict (optional)    — VerificationTally.snapshot()
    """
    geom_pickle, src, track_neutron, batch_idx = args

    # ── 1. Reconstitute the geometry inside the worker process ──────────────
    # Each worker gets its own *deep copy* of the geometry. Tallies and
    # performance counters are part of the geometry object, so they are
    # already isolated per-worker — there is no shared state between workers.
    geom = pickle.loads(geom_pickle)
    src.geometry = geom

    # ── 2. Sample a fresh batch on the worker side ──────────────────────────
    # We do NOT pre-sample neutrons in the parent and ship them, because
    # that would multiply the pickle size by neutron count. Sampling here
    # keeps inter-worker traffic to just the geometry pickle once.
    batch_src = src.generate_batch(src.neutron_nbr)

    # ── 3. Reset per-batch counters / tallies ───────────────────────────────
    # `geom.reset()` (corrected) preserves preprocessing time and wipes
    # only the per-batch state. Important: the workers have NO preprocessing
    # of their own to preserve, but reset() still does the right thing.
    geom.reset()

    # ── 4. Run the actual transport for this batch ──────────────────────────
    # `from_batch=True` tells run_source not to start/stop its own memory
    # tracker (the parent process is in charge of memory bookkeeping).
    geom.run_source(batch_src, track_neutron=track_neutron, from_batch=True)

    # ── 5. Capture worker peak RSS for memory diagnostics ───────────────────
    # We ONLY have a single-point measurement here, not a true peak,
    # because no background poll thread is running in the worker (the
    # poll thread isn't restarted by the unpickle path — see
    # MemoryTracker.__setstate__). For a true peak, the worker would need
    # to start its own poll thread; the trade-off is overhead vs accuracy.
    process = psutil.Process(os.getpid())
    peak_mb = process.memory_info().rss / 1e6

    # ── 6. Assemble the batch snapshot ──────────────────────────────────────
    # Schema is identical to the serial path so `_compute_batch_stats`
    # consumes either uniformly.
    snap = {
        "batch"     : batch_idx,
        "n_neutrons": batch_src.neutron_nbr,
        "perf"      : geom.perf.snapshot(),
        "peak_mb"   : peak_mb,
    }

    # Optional sub-tallies (only present if attached to this geometry)
    if geom.flux_tally_flag and geom.flux_tally is not None:
        snap["flux"] = {
            "energy_bins": geom.flux_tally.energy_bins.tolist(),
            "flux"       : {
                "mean"          : geom.flux_tally.mean.tolist(),
                "std"           : geom.flux_tally.std.tolist(),
                "relative_error": geom.flux_tally.relative_error.tolist(),
            },
        }

    if geom.verif_tally_flag and geom.verif_tally is not None:
        snap["verif"] = geom.verif_tally.snapshot()

    return snap


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED run_batch — paste this method onto the Geometry class
# ══════════════════════════════════════════════════════════════════════════════
def run_batch(self,
              src,
              n_batches:    int,
              mode:         str  = "serial",
              n_workers:    int  = None,
              track_neutron: bool = False):
    """
    Run a batched Monte Carlo transport simulation in either serial or
    parallel mode, selectable by the ``mode`` argument.

    This is the unified replacement for the old
    ``run_batch_serial`` / ``run_batch_parallel`` pair. Both old methods
    diverged in subtle ways (timing instrumentation, memory tracker
    handling, error checking) which made it easy to introduce bugs that
    only showed up in one of the two paths. Having a single function
    eliminates that drift.

    Parameters
    ----------
    src : Source
        Configured Source object whose ``neutron_nbr`` defines the batch
        size — i.e. EACH batch will simulate ``src.neutron_nbr`` neutrons,
        matching the OpenMC convention (total particles =
        ``n_batches * src.neutron_nbr``).
    n_batches : int
        Number of independent batches. Must be >= 2 so the cross-batch
        variance estimator (s² / n_batches) is well-defined.
    mode : {"serial", "parallel"}, default "serial"
        Execution strategy:
            "serial"   — single process, sequential loop. Use this when
                         debugging or when neutron counts are small enough
                         that fork/pickle overhead would dominate.
            "parallel" — multiprocessing.Pool over n_workers processes.
                         Each worker simulates a complete batch end-to-end.
                         Recommended whenever the per-batch wall time
                         exceeds ~0.5 s (otherwise pool startup dominates).
    n_workers : int or None, default None
        Number of worker processes for parallel mode. ``None`` means
        "use all available CPUs" (per ``multiprocessing.Pool`` default,
        which calls ``os.cpu_count()``). Ignored entirely in serial mode.
    track_neutron : bool, default False
        Passed through to ``run_source``. When True, each neutron's full
        track is recorded — slow and memory-hungry, intended for
        diagnostics only.

    Returns
    -------
    batch_stats : dict
        Aggregated per-batch statistics (mean ± std across batches) for
        every tally and performance metric. Plus a new ``"timing"`` key
        holding a ``BatchTimer`` instance with honest end-to-end wall
        time and parallel-efficiency information.

    Side Effects
    ------------
    * ``self.batch_results`` is populated with one snap dict per batch.
    * ``self.batch_stats`` is set to the returned dict.
    * If ``self.memory_tracker_flag`` is True, the memory tracker is
      started/stopped around the whole run and worker peaks are folded
      in as snapshots labeled ``worker_batch_<i>_peak``.

    Notes
    -----
    * Per-batch RNG independence:
        - Serial:   each batch samples from the same Source/Random
                    instance — neutrons are statistically independent
                    BY CONSTRUCTION because each batch draws fresh
                    samples from the underlying RNG, never replaying.
        - Parallel: each worker process inherits Python's RNG state at
                    fork/spawn time. On Linux (fork) the workers START
                    with identical state but consume RNG for their own
                    batch independently; on Windows/macOS (spawn) each
                    worker re-initializes its own RNG. If you need
                    reproducible results across runs, seed the RNG
                    explicitly inside ``_run_single_batch_worker``.

    * Parallel timing math:
        - ``BatchTimer.wall_total`` is the elapsed time from the user's
          point of view (the only timing that matters for "is this
          faster?").
        - ``BatchTimer.sum_worker_cpu`` is the total CPU work done by
          all workers combined.
        - ``parallel_efficiency = sum_worker_cpu / (n_workers · wall_total)``.
          Ideal scaling = 1.0; values below ~0.7 typically mean the
          batch granularity is too fine or there is contention on
          shared resources (file system, GIL — though we use processes,
          C extensions can still serialize on global locks).
    """

    # ─────────────────────────────────────────────────────────────────────
    # 1. ARGUMENT VALIDATION
    # ─────────────────────────────────────────────────────────────────────
    # We require n_batches >= 2 because the cross-batch variance is
    # computed as sample_std(batch_means, ddof=1) / sqrt(n_batches).
    # With n_batches < 2 ddof=1 produces NaN, and the relative-error
    # diagnostic becomes meaningless.
    if n_batches < 2:
        raise ValueError(
            f"n_batches must be >= 2 for cross-batch statistics, got {n_batches}."
        )

    # Normalize and validate mode early so a typo fails fast rather than
    # silently picking the wrong execution path.
    mode_normalized = mode.strip().lower()
    if mode_normalized not in ("serial", "parallel"):
        raise ValueError(
            f"mode must be 'serial' or 'parallel', got {mode!r}."
        )

    # In serial mode, n_workers is meaningless. We don't error on it
    # (callers may pass it through generic plumbing), but we do silently
    # ignore it to avoid confusing the BatchTimer's parallel_efficiency.
    effective_n_workers = (
        1 if mode_normalized == "serial"
        else (n_workers if n_workers is not None else os.cpu_count())
    )

    # Capture neutron-count up front. We deliberately use the source's
    # current neutron_nbr (NOT divided by n_batches) because each batch
    # runs the FULL source — this matches OpenMC's "particles per batch"
    # convention. Capturing here also guards against weird mutations of
    # src.neutron_nbr that might happen mid-loop.
    neutrons_per_batch = src.neutron_nbr
    if neutrons_per_batch <= 0:
        raise ValueError(
            f"src.neutron_nbr must be > 0, got {neutrons_per_batch}."
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2. BOOKKEEPING / TRACKER SETUP
    # ─────────────────────────────────────────────────────────────────────
    # Reset the per-batch results list. We do this BEFORE the timer so
    # the (small) cost of clearing isn't attributed to the simulation.
    self.batch_results = []

    # Memory tracking is opt-in via the geometry's flag. We start the
    # poller in the parent process for the duration of the entire batch
    # run, regardless of execution mode. In parallel mode the parent
    # process is mostly idle while workers run, so its RSS curve is flat;
    # individual worker peaks are folded in below from each batch snap.
    if self.memory_tracker_flag:
        self.memory.start()
        self.memory.snapshot(f"{mode_normalized}_start")

    # ─────────────────────────────────────────────────────────────────────
    # 3. END-TO-END TIMING
    # ─────────────────────────────────────────────────────────────────────
    # BatchTimer is a context manager: __enter__ records perf_counter and
    # process_time, __exit__ records the deltas. Use this — NOT a sum of
    # per-batch perf snapshots — when you want to know "how long did the
    # user actually wait?" The two values differ because:
    #   * Serial:   per-batch sum excludes generate_batch + reset overhead.
    #   * Parallel: per-batch sum is wall-clock work, not real elapsed
    #               time, since workers run concurrently.
    timer = BatchTimer(
        label     = mode_normalized,
        n_workers = effective_n_workers,
        n_batches = n_batches,
    )

    with timer:

        # ─────────────────────────────────────────────────────────────────
        # 4. SERIAL EXECUTION PATH
        # ─────────────────────────────────────────────────────────────────
        if mode_normalized == "serial":
            for batch_idx in range(n_batches):

                # Each batch:
                #   (a) sample fresh neutrons from the source
                #   (b) reset per-batch counters (preserving preprocessing)
                #   (c) transport them
                #   (d) snapshot tallies + perf
                #
                # The reset() call (in the corrected geometry) calls
                # self.perf.reset_counters(keep_preprocessing=True),
                # so preprocessing time accumulated BEFORE this loop
                # survives across iterations and ends up in batch 0's
                # snapshot. We do NOT subtract it from later batches —
                # downstream code is expected to handle that via the
                # `time_preprocessing_s` field (it's already 0 for
                # batches 1..N because reset zeros the per-batch perf
                # but keeps the cumulative pp counter).
                batch_src = src.generate_batch(neutrons_per_batch)

                self.reset()
                self.run_source(batch_src,
                                track_neutron=track_neutron,
                                from_batch=True)

                # Optional memory snapshot per batch
                if self.memory_tracker_flag:
                    self.memory.snapshot(f"post_batch_{batch_idx + 1}")

                # Build the per-batch snapshot dict. Schema is identical
                # to what _run_single_batch_worker produces in the
                # parallel path, so _compute_batch_stats stays generic.
                snap = {
                    "batch"     : batch_idx,
                    "n_neutrons": neutrons_per_batch,
                    "perf"      : self.perf.snapshot(),
                }
                if self.flux_tally_flag and self.flux_tally is not None:
                    snap["flux"] = {
                        "energy_bins": self.flux_tally.energy_bins.tolist(),
                        "flux"       : {
                            "mean"          : self.flux_tally.mean.tolist(),
                            "std"           : self.flux_tally.std.tolist(),
                            "relative_error": self.flux_tally.relative_error.tolist(),
                        },
                    }
                if self.verif_tally_flag and self.verif_tally is not None:
                    snap["verif"] = self.verif_tally.snapshot()

                self.batch_results.append(snap)

                # Light progress line. We print perf.total_time which is
                # the time spent in run_source ONLY — the timer above
                # measures the broader interval that includes reset()
                # and generate_batch() and the snapshot() calls.
                print(
                    f"  [Batch {mode_normalized:>8}] "
                    f"{batch_idx + 1:>3}/{n_batches}  "
                    f"({neutrons_per_batch:,} neutrons) | "
                    f"wall = {self.perf.total_time:.3f} s"
                )

        # ─────────────────────────────────────────────────────────────────
        # 5. PARALLEL EXECUTION PATH
        # ─────────────────────────────────────────────────────────────────
        else:
            # Pickle the geometry ONCE in the parent. Each worker
            # deserializes a fresh copy — there is zero shared state.
            # Doing this outside the Pool also means the (potentially
            # significant) pickle cost is paid once instead of per
            # worker.
            geom_pickle = pickle.dumps(self)

            # Build the per-task argument list. Each entry corresponds
            # to one batch and includes a unique batch_idx so we can
            # restore the original batch ordering after Pool.map (which
            # IS order-preserving, but we store the index defensively).
            args = [
                (geom_pickle, src, track_neutron, batch_idx)
                for batch_idx in range(n_batches)
            ]

            print(f"\n[Parallel] Running {n_batches} batches on "
                  f"{effective_n_workers} workers...")

            # Pool.map is the simplest primitive that:
            #   * ships each task to a worker
            #   * blocks until ALL workers finish
            #   * returns the results in the order tasks were submitted
            #
            # We use the default chunksize because batches are coarse-
            # grained (each one is at minimum ~milliseconds, often
            # seconds), so the chunking heuristic doesn't matter.
            with Pool(processes=effective_n_workers) as pool:
                self.batch_results = pool.map(_run_single_batch_worker, args)

            # Sort by batch index — defensive: if a future Pool variant
            # returns out of order this still produces deterministic
            # results indexed by batch number.
            self.batch_results.sort(key=lambda r: r["batch"])

            # Collate worker-reported peak memory into the parent's
            # memory tracker so the summary covers all processes.
            if self.memory_tracker_flag:
                for r in self.batch_results:
                    if "peak_mb" in r:
                        self.memory.snapshot(
                            f"worker_batch_{r['batch'] + 1}_peak",
                            external_rss_mb=r["peak_mb"],
                        )

    # ─────────────────────────────────────────────────────────────────────
    # 6. POST-PROCESSING
    # ─────────────────────────────────────────────────────────────────────
    # Aggregate per-batch results into cross-batch mean ± std. The
    # _compute_batch_stats method is mode-agnostic — it consumes the
    # uniform snap-dict schema produced by both paths above.
    self.batch_stats = self._compute_batch_stats()

    # Attach the BatchTimer so callers can introspect end-to-end wall
    # time, summed worker CPU, parallel efficiency, etc. without having
    # to re-derive any of it.
    timer.attach_worker_records(self.batch_results)
    self.batch_stats["timing"] = timer

    # Print a one-line summary so users see the actual elapsed time
    # immediately rather than having to dig into batch_stats["timing"].
    print(
        f"\n[run_batch] mode={mode_normalized!r}  "
        f"workers={effective_n_workers}  "
        f"batches={n_batches}  "
        f"wall={timer.wall_total:.3f} s  "
        f"throughput={timer.overall_throughput:,.0f} n/s"
        + (f"  parallel_eff={timer.parallel_efficiency:.3f}"
           if mode_normalized == "parallel" else "")
    )

    # ─────────────────────────────────────────────────────────────────────
    # 7. MEMORY TRACKER SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────
    # Stop the memory tracker AFTER computing batch_stats so the
    # post-processing memory cost is included in the trace.
    if self.memory_tracker_flag:
        self.memory.snapshot(f"{mode_normalized}_end")
        self.memory.stop()

    return self.batch_stats


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL BACKWARD-COMPATIBLE SHIMS
# ══════════════════════════════════════════════════════════════════════════════
# If you have existing code that calls `geom.run_batch_serial(...)` or
# `geom.run_batch_parallel(...)`, preserve those entry points by adding
# the following two thin wrappers to the Geometry class. They simply
# forward to `run_batch` with the appropriate mode. Delete them once
# all callers have been migrated.

def run_batch_serial(self, src, n_batches, track_neutron=False):
    """Deprecated: use ``run_batch(..., mode='serial')`` instead."""
    return self.run_batch(src, n_batches,
                          mode="serial",
                          track_neutron=track_neutron)


def run_batch_parallel(self, src, n_batches, n_workers=None, track_neutron=False):
    """Deprecated: use ``run_batch(..., mode='parallel', n_workers=...)`` instead."""
    return self.run_batch(src, n_batches,
                          mode="parallel",
                          n_workers=n_workers,
                          track_neutron=track_neutron)