import sys
import os
sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')

from dataclasses import dataclass, field
import time
import math
import threading
from datetime import datetime
from pathlib import Path
from itertools import accumulate

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

import majorant_multipole as maj
import openmc
import reconr
import psutil


#==========================================
# --- Memory usage tracking (unchanged) ---
#==========================================
@dataclass
class MemorySnapshot:
    label       : str
    timestamp   : float
    rss_mb      : float
    vms_mb      : float


class MemoryTracker:
    """
    Unchanged from the original — see prior version for full docstring.
    """

    def __init__(self, poll_interval: float = 0.001, name: str = "MemoryTracker"):
        self._name          = name
        self._process       = psutil.Process(os.getpid())
        self._snapshots     : List[MemorySnapshot] = []
        self._poll_interval = poll_interval
        self._t0            = time.perf_counter()

        self._poll_rss      : List[float] = []
        self._poll_times    : List[float] = []
        self._peak_rss_mb   = 0.0
        self._polling       = False
        self._thread        : Optional[threading.Thread] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_thread']  = None
        state['_polling'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread  = None
        self._polling = False

    def start(self):
        if self._polling:
            return
        self._polling = True
        self._thread  = threading.Thread(
            target = self._poll_loop,
            daemon = True,
            name   = "MemoryPoller",
        )
        self._thread.start()
        self.snapshot("init")
        print(f"\n [Memory] Tracker started (poll interval: {self._poll_interval*1000:.0f} ms)")

    def stop(self):
        self._polling = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.snapshot("stop")
        print(f"\n [Memory] Tracker stopped.")

    def _poll_loop(self):
        while self._polling:
            try:
                rss = self._process.memory_info().rss / 1e6
                t   = time.perf_counter() - self._t0
                self._poll_rss.append(rss)
                self._poll_times.append(t)
                if rss > self._peak_rss_mb:
                    self._peak_rss_mb = rss
            except psutil.NoSuchProcess:
                break
            time.sleep(self._poll_interval)

    def snapshot(self, label: str, external_rss_mb: float = None) -> MemorySnapshot:
        if external_rss_mb is not None:
            rss = external_rss_mb
            vms = 0.0
        else:
            mem = self._process.memory_info()
            rss = mem.rss / 1e6
            vms = mem.vms / 1e6

        snap = MemorySnapshot(
            label     = label,
            timestamp = time.perf_counter() - self._t0,
            rss_mb    = rss,
            vms_mb    = vms,
        )
        self._snapshots.append(snap)
        if rss > self._peak_rss_mb:
            self._peak_rss_mb = rss
        return snap

    def peak_mb(self) -> float:
        return self._peak_rss_mb

    def current_mb(self) -> float:
        return self._process.memory_info().rss / 1e6

    def delta_mb(self, label_a: str, label_b: str) -> float:
        a = self._get(label_a)
        b = self._get(label_b)
        if a is None or b is None:
            raise KeyError(f"Label not found: {label_a!r} or {label_b!r}")
        return b.rss_mb - a.rss_mb

    def snapshots_to_dataframe(self) -> pd.DataFrame:
        base = self._snapshots[0].rss_mb if self._snapshots else 0.0
        return pd.DataFrame([
            {
                "label"    : s.label,
                "time_s"   : round(s.timestamp, 7),
                "rss_mb"   : round(s.rss_mb, 7),
                "vms_mb"   : round(s.vms_mb, 7),
                "delta_mb" : round(s.rss_mb - base, 7),
            }
            for s in self._snapshots
        ])

    def poll_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time_s" : [round(t, 7) for t in self._poll_times],
            "rss_mb" : [round(r, 7) for r in self._poll_rss],
        })

    def summary(self) -> str:
        if not self._snapshots:
            return "No snapshots recorded."

        base  = self._snapshots[0].rss_mb
        df    = self.snapshots_to_dataframe()
        lines = [
            "=" * 65,
            f"  MEMORY TRACKER SUMMARY ({self._name})",
            "=" * 65,
            f"  Baseline RSS          : {base:.3e} MB",
            f"  Peak RSS (continuous) : {self._peak_rss_mb:.3e} MB",
            f"  Total growth          : {self._peak_rss_mb - base:+.3e} MB",
            "-" * 65,
            f"  {'Label':<28} {'RSS (MB)':>9} {'Delta (MB)':>11} {'Time (s)':>9}",
            "-" * 65,
        ]
        for _, row in df.iterrows():
            lines.append(
                f"  {row['label']:<28} {row['rss_mb']:>9.1f} "
                f"{row['delta_mb']:>+11.1f} {row['time_s']:>9.3f}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def _get(self, label: str) -> Optional[MemorySnapshot]:
        for s in reversed(self._snapshots):
            if s.label == label:
                return s
        return None


# ==========================================
# --- PerformanceTracker dataclass (FIXED) ---
# ==========================================
@dataclass
class PerformanceTracker:
    """
    Wall-clock AND CPU-time performance diagnostics.

    Wall time  : time.perf_counter()   — real elapsed time (includes I/O, sleep …)
    CPU time   : time.process_time()   — on-CPU time only (no I/O wait, no sleep)

    Bug-fix changes vs original
    ---------------------------
    [FIX-1]  start()/stop() use None sentinel so total_time returns NaN if the
             timer was never bracketed properly (instead of returning a huge
             nonsense number).
    [FIX-2]  reset_counters() — replaces "self.perf = PerformanceTracker()" in
             Geometry.reset(). Wipes per-batch counters and the simulation timer
             but PRESERVES preprocessing time accumulated outside the batch loop.
    [FIX-3]  snapshot() now exports total_cpu_time_s, cpu_time_majorant_s,
             and cpu_time_xs_eval_s. These were dataclass fields but never
             reached batch_results before.
    [FIX-4]  Helper context-managers `time_xs_eval()` and `time_majorant()`
             so calling code stops forgetting to update one of wall/cpu.
    [FIX-5]  Single source of truth for cpu_efficiency: it is per-process,
             documented as such. Parallel efficiency belongs to BatchTimer.

    See also: BatchTimer (below) for end-to-end serial/parallel timing.
    """
    # ── internal start/end snapshots ──────────────────────────────────────────
    _wall_start: Optional[float] = field(default=None, repr=False)
    _wall_end:   Optional[float] = field(default=None, repr=False)
    _cpu_start:  Optional[float] = field(default=None, repr=False)
    _cpu_end:    Optional[float] = field(default=None, repr=False)

    _wall_preprocessing_start: Optional[float] = field(default=None, repr=False)
    _wall_preprocessing_end:   Optional[float] = field(default=None, repr=False)
    _cpu_preprocessing_start:  Optional[float] = field(default=None, repr=False)
    _cpu_preprocessing_end:    Optional[float] = field(default=None, repr=False)

    # ── preprocessing timers (cumulative) ─────────────────────────────────────
    time_preprocessing:     float = 0.0
    cpu_time_preprocessing: float = 0.0

    # ── event counts ──────────────────────────────────────────────────────────
    n_neutrons:                  int   = 0
    n_real_collisions:           int   = 0
    n_virtual_collisions:        int   = 0
    n_xs_evaluations:            int   = 0
    n_majorant_updates:          int   = 0
    n_wrong_majorant:            int   = 0
    n_wrong_majorant_mean_error: float = 0.0

    # ── wall-clock timers [seconds] ───────────────────────────────────────────
    time_majorant: float = 0.0
    time_xs_eval:  float = 0.0

    # ── CPU-time timers [seconds] ─────────────────────────────────────────────
    cpu_time_majorant: float = 0.0
    cpu_time_xs_eval:  float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # PREPROCESSING TIMERS
    # ──────────────────────────────────────────────────────────────────────────
    def start_preprocessing(self):
        self._wall_preprocessing_start = time.perf_counter()
        self._cpu_preprocessing_start  = time.process_time()

    def stop_preprocessing(self):
        if self._wall_preprocessing_start is None:
            raise RuntimeError("stop_preprocessing() called without start_preprocessing()")
        self._wall_preprocessing_end = time.perf_counter()
        self._cpu_preprocessing_end  = time.process_time()
        self.time_preprocessing     += (self._wall_preprocessing_end -
                                        self._wall_preprocessing_start)
        self.cpu_time_preprocessing += (self._cpu_preprocessing_end -
                                        self._cpu_preprocessing_start)
        # Reset markers so an unmatched stop is detectable next time
        self._wall_preprocessing_start = None
        self._cpu_preprocessing_start  = None

    # ──────────────────────────────────────────────────────────────────────────
    # SIMULATION TIMERS  [FIX-1: None sentinels]
    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        """Mark the start of the simulation timing region."""
        self._wall_start = time.perf_counter()
        self._cpu_start  = time.process_time()
        self._wall_end   = None
        self._cpu_end    = None

    def stop(self):
        """Mark the end of the simulation timing region. Idempotent."""
        # [FIX] Idempotent — second consecutive call is a no-op so the duplicate
        # stop in run_source no longer extends the measured interval.
        if self._wall_end is not None:
            return
        if self._wall_start is None:
            raise RuntimeError("stop() called without start()")
        self._wall_end = time.perf_counter()
        self._cpu_end  = time.process_time()

    # ──────────────────────────────────────────────────────────────────────────
    # SUB-REGION TIMERS  [FIX-4]
    # ──────────────────────────────────────────────────────────────────────────
    class _Region:
        """Context manager that increments wall- and CPU-time counters atomically."""
        __slots__ = ("_perf", "_wall_attr", "_cpu_attr",
                     "_wall_t0", "_cpu_t0", "_count_attr")

        def __init__(self, perf, wall_attr, cpu_attr, count_attr=None):
            self._perf       = perf
            self._wall_attr  = wall_attr
            self._cpu_attr   = cpu_attr
            self._count_attr = count_attr

        def __enter__(self):
            self._wall_t0 = time.perf_counter()
            self._cpu_t0  = time.process_time()
            return self

        def __exit__(self, exc_type, exc, tb):
            wall = time.perf_counter() - self._wall_t0
            cpu  = time.process_time() - self._cpu_t0
            setattr(self._perf, self._wall_attr,
                    getattr(self._perf, self._wall_attr) + wall)
            setattr(self._perf, self._cpu_attr,
                    getattr(self._perf, self._cpu_attr) + cpu)
            if self._count_attr is not None:
                setattr(self._perf, self._count_attr,
                        getattr(self._perf, self._count_attr) + 1)
            return False

    def time_xs_eval_region(self):
        """`with perf.time_xs_eval_region(): xs = mat._xs_evaluation(E)`"""
        return self._Region(self, "time_xs_eval", "cpu_time_xs_eval", "n_xs_evaluations")

    def time_majorant_region(self):
        """`with perf.time_majorant_region(): mxs = self.get_majorant_xs(E)`"""
        return self._Region(self, "time_majorant", "cpu_time_majorant", "n_majorant_updates")

    # ──────────────────────────────────────────────────────────────────────────
    # PER-BATCH RESET  [FIX-2]
    # ──────────────────────────────────────────────────────────────────────────
    def reset_counters(self, keep_preprocessing: bool = True):
        """
        Reset per-batch counters so this tracker can time another batch.

        keep_preprocessing : bool
            If True (default), retain `time_preprocessing` and
            `cpu_time_preprocessing` accumulated outside the batch loop.
            Use False only when you really want a clean slate.
        """
        # Snapshot & restore preprocessing if requested
        if keep_preprocessing:
            saved_pp_wall = self.time_preprocessing
            saved_pp_cpu  = self.cpu_time_preprocessing

        # Per-batch counters
        self._wall_start = None
        self._wall_end   = None
        self._cpu_start  = None
        self._cpu_end    = None
        self._wall_preprocessing_start = None
        self._wall_preprocessing_end   = None
        self._cpu_preprocessing_start  = None
        self._cpu_preprocessing_end    = None

        self.n_neutrons                  = 0
        self.n_real_collisions           = 0
        self.n_virtual_collisions        = 0
        self.n_xs_evaluations            = 0
        self.n_majorant_updates          = 0
        self.n_wrong_majorant            = 0
        self.n_wrong_majorant_mean_error = 0.0

        self.time_majorant     = 0.0
        self.time_xs_eval      = 0.0
        self.cpu_time_majorant = 0.0
        self.cpu_time_xs_eval  = 0.0

        if keep_preprocessing:
            self.time_preprocessing     = saved_pp_wall
            self.cpu_time_preprocessing = saved_pp_cpu
        else:
            self.time_preprocessing     = 0.0
            self.cpu_time_preprocessing = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # DERIVED PROPERTIES
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def total_time(self) -> float:
        """Total wall-clock time of simulation region [s]. NaN if not bracketed."""
        if self._wall_start is None or self._wall_end is None:
            return float("nan")
        return self._wall_end - self._wall_start

    @property
    def total_cpu_time(self) -> float:
        """Total CPU time of simulation region [s]. NaN if not bracketed."""
        if self._cpu_start is None or self._cpu_end is None:
            return float("nan")
        return self._cpu_end - self._cpu_start

    @property
    def cpu_efficiency(self) -> float:
        """
        Per-process CPU/wall ratio over the simulation region.

        Interpretation
        --------------
            ~1.0 : fully CPU-bound, single-threaded (typical for this code)
            <1.0 : I/O-bound, blocked, or sleeping
            >1.0 : multi-threaded inside the region

        WARNING: this is NOT parallel efficiency. For multi-process speedup
        and parallel efficiency, use BatchTimer (defined below).
        """
        wt = self.total_time
        ct = self.total_cpu_time
        if not (math.isfinite(wt) and wt > 0.0 and math.isfinite(ct)):
            return float("nan")
        return ct / wt

    @property
    def neutrons_per_second(self) -> float:
        """Throughput excluding preprocessing time."""
        wt = self.total_time
        if not math.isfinite(wt) or wt <= 0.0:
            return 0.0
        denom = wt - self.time_preprocessing
        if denom <= 0.0:
            return 0.0
        return self.n_neutrons / denom

    @property
    def rejection_fraction(self) -> float:
        total = self.n_virtual_collisions + self.n_real_collisions
        return self.n_virtual_collisions / total if total > 0 else 0.0

    @property
    def wrong_majorant_mean_error(self) -> float:
        return (self.n_wrong_majorant_mean_error / self.n_wrong_majorant
                if self.n_wrong_majorant > 0 else 0.0)

    @property
    def wrong_majorant_fraction(self) -> float:
        return (self.n_wrong_majorant / self.n_majorant_updates
                if self.n_majorant_updates > 0 else 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # SNAPSHOT  [FIX-3]
    # ──────────────────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a dict of scalar performance metrics for batch bookkeeping.

        Includes wall AND CPU for every region; nothing is silently dropped.
        """
        return {
            # Top-level wall + cpu
            "total_time_s"             : self.total_time,
            "total_cpu_time_s"         : self.total_cpu_time,
            "cpu_efficiency"           : self.cpu_efficiency,
            # Preprocessing
            "time_preprocessing_s"     : self.time_preprocessing,
            "cpu_time_preprocessing_s" : self.cpu_time_preprocessing,
            # Sub-region timers (wall + CPU both)
            "time_majorant_s"          : self.time_majorant,
            "cpu_time_majorant_s"      : self.cpu_time_majorant,
            "time_xs_eval_s"           : self.time_xs_eval,
            "cpu_time_xs_eval_s"       : self.cpu_time_xs_eval,
            # Throughput / counters
            "neutrons_per_second"      : self.neutrons_per_second,
            "n_neutrons"               : self.n_neutrons,
            "n_real_collisions"        : self.n_real_collisions,
            "n_virtual_collisions"     : self.n_virtual_collisions,
            "n_xs_evaluations"         : self.n_xs_evaluations,
            "n_majorant_updates"       : self.n_majorant_updates,
            "rejection_fraction"       : self.rejection_fraction,
            # Majorant quality
            "n_wrong_majorant"         : self.n_wrong_majorant,
            "wrong_majorant_mean_error": self.wrong_majorant_mean_error,
            "wrong_majorant_fraction"  : self.wrong_majorant_fraction,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PERFORMANCE SUMMARY",
            "=" * 60,
            f"  {'Metric':<30} {'Wall (s)':>10}  {'CPU (s)':>10}",
            "-" * 60,
            f"  {'Total simulation':<30} {self.total_time:>10.4f}  {self.total_cpu_time:>10.4f}",
            f"  {'Preprocessing':<30} {self.time_preprocessing:>10.4f}  {self.cpu_time_preprocessing:>10.4f}",
            f"  {'  Majorant evaluations':<30} {self.time_majorant:>10.4f}  {self.cpu_time_majorant:>10.4f}",
            f"  {'  XS evaluations':<30} {self.time_xs_eval:>10.4f}  {self.cpu_time_xs_eval:>10.4f}",
            "-" * 60,
            f"  CPU efficiency (cpu/wall)    : {self.cpu_efficiency:>10.3f}    [per-process; not parallel eff.]",
            f"  Neutrons / second            : {self.neutrons_per_second:>10.1f}",
            "-" * 60,
            f"  Neutrons simulated           : {self.n_neutrons:>10,}",
            f"  XS evaluations               : {self.n_xs_evaluations:>10,}",
            f"  Majorant updates             : {self.n_majorant_updates:>10,}",
            f"  Real collisions              : {self.n_real_collisions:>10,}",
            f"  Virtual (rejected)           : {self.n_virtual_collisions:>10,}",
            f"  Rejection fraction           : {self.rejection_fraction:>10.3%}",
            f"  Wrong majorant fraction      : {self.wrong_majorant_fraction:>10.3%}",
            f"  Wrong majorant mean error    : {self.wrong_majorant_mean_error:>10.3%}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ==========================================
# --- BatchTimer (NEW) ---
# ==========================================
@dataclass
class BatchTimer:
    """
    End-to-end timer for run_batch_serial / run_batch_parallel.

    Wraps the entire batch loop (or Pool.map call) so you can compute
    speedup and parallel efficiency. Use as a context manager:

        with BatchTimer(label="parallel", n_workers=8) as bt:
            batch_stats = parallel.run_batch_parallel(...)
        bt.attach_worker_records(geom.batch_results)
        print(bt.summary())

    Definitions
    -----------
        wall_total       : real elapsed time of the wrapped block
        cpu_total_main   : CPU time spent in the *main* process (≈ 0 for parallel
                           since workers do the work in their own processes)
        sum_worker_cpu   : sum of CPU time across all batches (workers + main)
        speedup          : N_workers · wall_serial / wall_parallel
                           (only defined if BOTH a serial and a parallel timer
                           are recorded — see classmethod compare(...))
        parallel_eff     : sum_worker_cpu / (n_workers · wall_total)
                           ~1.0 = ideal scaling, 0.5 = workers idle half the time
    """
    label:        str           = "batch"
    n_workers:    Optional[int] = None
    n_batches:    Optional[int] = None
    n_neutrons_total: Optional[int] = None

    wall_total:       float = 0.0
    cpu_total_main:   float = 0.0
    sum_worker_wall:  float = 0.0   # populated by attach_worker_records
    sum_worker_cpu:   float = 0.0   # populated by attach_worker_records
    sum_worker_pp_wall: float = 0.0
    sum_worker_pp_cpu:  float = 0.0

    _wall_t0: Optional[float] = field(default=None, repr=False)
    _cpu_t0:  Optional[float] = field(default=None, repr=False)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def __enter__(self):
        self._wall_t0 = time.perf_counter()
        self._cpu_t0  = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wall_total     = time.perf_counter() - self._wall_t0
        self.cpu_total_main = time.process_time() - self._cpu_t0
        return False

    # ── populate from per-batch perf snapshots ────────────────────────────────
    def attach_worker_records(self, batch_results: list):
        """
        Aggregate the wall/CPU work each worker did. Pass `geom.batch_results`
        — i.e. the list of per-batch dicts each containing a "perf" snapshot.
        """
        self.n_batches         = len(batch_results)
        self.sum_worker_wall   = 0.0
        self.sum_worker_cpu    = 0.0
        self.sum_worker_pp_wall = 0.0
        self.sum_worker_pp_cpu  = 0.0
        n_neutrons_total       = 0
        for r in batch_results:
            p = r.get("perf", {})
            wt = p.get("total_time_s", float("nan"))
            ct = p.get("total_cpu_time_s", float("nan"))
            if math.isfinite(wt):
                self.sum_worker_wall += wt
            if math.isfinite(ct):
                self.sum_worker_cpu += ct
            self.sum_worker_pp_wall += p.get("time_preprocessing_s", 0.0)
            self.sum_worker_pp_cpu  += p.get("cpu_time_preprocessing_s", 0.0)
            n_neutrons_total        += int(p.get("n_neutrons", 0))
        self.n_neutrons_total = n_neutrons_total

    # ── derived ───────────────────────────────────────────────────────────────
    @property
    def parallel_efficiency(self) -> float:
        if (self.n_workers is None or self.n_workers <= 0
                or self.wall_total <= 0.0 or not math.isfinite(self.sum_worker_cpu)):
            return float("nan")
        return self.sum_worker_cpu / (self.n_workers * self.wall_total)

    @property
    def overall_throughput(self) -> float:
        if self.n_neutrons_total is None or self.wall_total <= 0.0:
            return 0.0
        return self.n_neutrons_total / self.wall_total

    @classmethod
    def compute_speedup(cls, serial: "BatchTimer", parallel: "BatchTimer") -> float:
        """speedup = T_serial / T_parallel, both end-to-end wall times."""
        if serial.wall_total <= 0.0 or parallel.wall_total <= 0.0:
            return float("nan")
        return serial.wall_total / parallel.wall_total

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> str:
        nw = self.n_workers if self.n_workers else 1
        lines = [
            "=" * 60,
            f"  BATCH TIMER — {self.label}",
            "=" * 60,
            f"  n_workers          : {nw}",
            f"  n_batches          : {self.n_batches}",
            f"  n_neutrons (total) : {self.n_neutrons_total:,}"
                if self.n_neutrons_total is not None else "",
            "-" * 60,
            f"  Wall (end-to-end)  : {self.wall_total:>10.4f} s",
            f"  CPU  (main proc)   : {self.cpu_total_main:>10.4f} s",
            f"  Σ worker wall      : {self.sum_worker_wall:>10.4f} s",
            f"  Σ worker CPU       : {self.sum_worker_cpu:>10.4f} s",
            f"  Σ worker prep wall : {self.sum_worker_pp_wall:>10.4f} s",
            "-" * 60,
            f"  Throughput         : {self.overall_throughput:>10.1f} n/s "
            f"({self.overall_throughput/nw:.1f} n/s/worker)",
            f"  Parallel efficiency: {self.parallel_efficiency:>10.3f}",
            "=" * 60,
        ]
        return "\n".join(l for l in lines if l != "")


# ==========================================
# --- MajorantRecord dataclass (unchanged) ---
# ==========================================
@dataclass
class MajorantRecord:
    energy: float
    value: float
    limiting_material: str
    actual_max_xs: float

    @property
    def margin(self) -> float:
        return self.value / self.actual_max_xs if self.actual_max_xs > 0 else float("inf")


# ==========================================
# --- NeutronHistory dataclass (unchanged) ---
# ==========================================
@dataclass
class NeutronHistory:
    neutron_id:     int
    birth_energy:   float
    birth_position: List[float]

    positions:           List[List[float]] = field(default_factory=list)
    energies:            List[float]       = field(default_factory=list)
    events:              List[str]         = field(default_factory=list)
    majorant_xs_at_step: List[float]       = field(default_factory=list)
    local_xs_at_step:    List[float]       = field(default_factory=list)
    material_at_step:    List[str]         = field(default_factory=list)
    distances:           List[float]       = field(default_factory=list)

    fate:       str = ""
    n_scatters: int = 0
    n_virtual:  int = 0

    @property
    def n_steps(self) -> int:
        return len(self.events)

    @property
    def total_path_length(self) -> float:
        return float(np.sum(self.distances))

    @property
    def final_energy(self) -> float:
        return self.energies[-1] if self.energies else self.birth_energy

    @property
    def energy_loss_fraction(self) -> float:
        if self.birth_energy > 0:
            return 1.0 - self.final_energy / self.birth_energy
        return 0.0