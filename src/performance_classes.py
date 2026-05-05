"""
performance_classes.py  (updated timing section)
=================================================
Timing structure:

    time_preprocessing_s   : wall time spent in set_maj_xs_method, set_access_method,
                             set_mode, etc. (accumulated via start/stop_preprocessing)

    time_run_source_s      : wall time of run_source() — pure neutron transport,
                             NO preprocessing included.
                             Used for neutrons_per_second.

    time_total_s           : time_preprocessing_s + time_run_source_s
                             (what the user actually waited for end-to-end)

    time_majorant_s        : cumulative wall time inside get_majorant_xs()
                             (called during transport → subset of time_run_source_s)

    time_xs_eval_s         : cumulative wall time inside _evaluate_acceptance()
                             (called during transport → subset of time_run_source_s)

    neutrons_per_second    : n_neutrons / time_run_source_s
"""

import math
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil


# ─────────────────────────────────────────────────────────────────────────────
# MemorySnapshot / MemoryTracker  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    label:     str
    timestamp: float
    rss_mb:    float
    vms_mb:    float


class MemoryTracker:
    def __init__(self, poll_interval: float = 0.001, name: str = "MemoryTracker"):
        self._name          = name
        self._process       = psutil.Process(os.getpid())
        self._snapshots:    List[MemorySnapshot] = []
        self._poll_interval = poll_interval
        self._t0            = time.perf_counter()
        self._poll_rss:     List[float] = []
        self._poll_times:   List[float] = []
        self._peak_rss_mb   = 0.0
        self._polling       = False
        self._thread:       Optional[threading.Thread] = None

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
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True, name="MemoryPoller")
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
            rss, vms = external_rss_mb, 0.0
        else:
            mem = self._process.memory_info()
            rss, vms = mem.rss / 1e6, mem.vms / 1e6
        snap = MemorySnapshot(label=label, timestamp=time.perf_counter() - self._t0, rss_mb=rss, vms_mb=vms)
        self._snapshots.append(snap)
        if rss > self._peak_rss_mb:
            self._peak_rss_mb = rss
        return snap

    def peak_mb(self) -> float:       return self._peak_rss_mb
    def current_mb(self) -> float:    return self._process.memory_info().rss / 1e6

    def delta_mb(self, label_a: str, label_b: str) -> float:
        a, b = self._get(label_a), self._get(label_b)
        if a is None or b is None:
            raise KeyError(f"Label not found: {label_a!r} or {label_b!r}")
        return b.rss_mb - a.rss_mb

    def snapshots_to_dataframe(self) -> pd.DataFrame:
        base = self._snapshots[0].rss_mb if self._snapshots else 0.0
        return pd.DataFrame([
            {"label": s.label, "time_s": round(s.timestamp, 7),
             "rss_mb": round(s.rss_mb, 7), "vms_mb": round(s.vms_mb, 7),
             "delta_mb": round(s.rss_mb - base, 7)}
            for s in self._snapshots
        ])

    def poll_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"time_s": [round(t, 7) for t in self._poll_times],
                             "rss_mb": [round(r, 7) for r in self._poll_rss]})

    def summary(self) -> str:
        if not self._snapshots:
            return "No snapshots recorded."
        base = self._snapshots[0].rss_mb
        df   = self.snapshots_to_dataframe()
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
            lines.append(f"  {row['label']:<28} {row['rss_mb']:>9.1f} "
                         f"{row['delta_mb']:>+11.1f} {row['time_s']:>9.3f}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def _get(self, label: str) -> Optional[MemorySnapshot]:
        for s in reversed(self._snapshots):
            if s.label == label:
                return s
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceTracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceTracker:
    """
    Four-timer performance tracker.

    Timers
    ------
    1. Preprocessing   : accumulated across all set_*() calls before run_source()
    2. Run-source      : pure neutron transport (start()/stop() in run_source)
    3. Total           : preprocessing + run-source  (derived property)
    4. Sub-regions     : majorant XS lookups and local XS evaluations
                         (subsets of run-source time, always reported)

    neutrons_per_second is computed from run-source time only.
    """

    # ── preprocessing timer ───────────────────────────────────────────────────
    _pp_wall_start: Optional[float] = field(default=None, repr=False)
    _pp_cpu_start:  Optional[float] = field(default=None, repr=False)

    time_preprocessing:     float = 0.0   # cumulative wall
    cpu_time_preprocessing: float = 0.0   # cumulative CPU

    # ── run-source timer ──────────────────────────────────────────────────────
    _rs_wall_start: Optional[float] = field(default=None, repr=False)
    _rs_wall_end:   Optional[float] = field(default=None, repr=False)
    _rs_cpu_start:  Optional[float] = field(default=None, repr=False)
    _rs_cpu_end:    Optional[float] = field(default=None, repr=False)

    # ── sub-region timers (subsets of run-source) ─────────────────────────────
    time_majorant:     float = 0.0
    cpu_time_majorant: float = 0.0
    time_xs_eval:      float = 0.0
    cpu_time_xs_eval:  float = 0.0

    # ── event counters ────────────────────────────────────────────────────────
    n_neutrons:                  int   = 0
    n_real_collisions:           int   = 0
    n_virtual_collisions:        int   = 0
    n_xs_evaluations:            int   = 0
    n_majorant_updates:          int   = 0
    n_wrong_majorant:            int   = 0
    n_wrong_majorant_mean_error: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # PREPROCESSING TIMER
    # ─────────────────────────────────────────────────────────────────────────

    def start_preprocessing(self):
        self._pp_wall_start = time.perf_counter()
        self._pp_cpu_start  = time.process_time()

    def stop_preprocessing(self):
        if self._pp_wall_start is None:
            raise RuntimeError("stop_preprocessing() called without start_preprocessing()")
        self.time_preprocessing     += time.perf_counter() - self._pp_wall_start
        self.cpu_time_preprocessing += time.process_time() - self._pp_cpu_start
        self._pp_wall_start = None
        self._pp_cpu_start  = None

    # ─────────────────────────────────────────────────────────────────────────
    # RUN-SOURCE TIMER  (pure neutron transport)
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        """Call at the start of run_source(), after preprocessing is done."""
        self._rs_wall_start = time.perf_counter()
        self._rs_cpu_start  = time.process_time()
        self._rs_wall_end   = None
        self._rs_cpu_end    = None

    def stop(self):
        """Call at the end of run_source(). Idempotent."""
        if self._rs_wall_end is not None:
            return
        if self._rs_wall_start is None:
            raise RuntimeError("stop() called without start()")
        self._rs_wall_end = time.perf_counter()
        self._rs_cpu_end  = time.process_time()

    # ─────────────────────────────────────────────────────────────────────────
    # SUB-REGION CONTEXT MANAGERS
    # ─────────────────────────────────────────────────────────────────────────

    class _Region:
        __slots__ = ("_perf", "_wall_attr", "_cpu_attr", "_count_attr",
                     "_wall_t0", "_cpu_t0")

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
            setattr(self._perf, self._wall_attr, getattr(self._perf, self._wall_attr) + wall)
            setattr(self._perf, self._cpu_attr,  getattr(self._perf, self._cpu_attr)  + cpu)
            if self._count_attr is not None:
                setattr(self._perf, self._count_attr, getattr(self._perf, self._count_attr) + 1)
            return False

    def time_majorant_region(self):
        return self._Region(self, "time_majorant", "cpu_time_majorant", "n_majorant_updates")

    def time_xs_eval_region(self):
        return self._Region(self, "time_xs_eval", "cpu_time_xs_eval", "n_xs_evaluations")

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset_counters(self, keep_preprocessing: bool = True):
        saved_pp_wall = self.time_preprocessing
        saved_pp_cpu  = self.cpu_time_preprocessing

        self._rs_wall_start = None
        self._rs_wall_end   = None
        self._rs_cpu_start  = None
        self._rs_cpu_end    = None
        self._pp_wall_start = None
        self._pp_cpu_start  = None

        self.n_neutrons                  = 0
        self.n_real_collisions           = 0
        self.n_virtual_collisions        = 0
        self.n_xs_evaluations            = 0
        self.n_majorant_updates          = 0
        self.n_wrong_majorant            = 0
        self.n_wrong_majorant_mean_error = 0.0

        self.time_majorant     = 0.0
        self.cpu_time_majorant = 0.0
        self.time_xs_eval      = 0.0
        self.cpu_time_xs_eval  = 0.0

        if keep_preprocessing:
            self.time_preprocessing     = saved_pp_wall
            self.cpu_time_preprocessing = saved_pp_cpu
        else:
            self.time_preprocessing     = 0.0
            self.cpu_time_preprocessing = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # DERIVED PROPERTIES
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def time_run_source(self) -> float:
        """Wall time of run_source() — pure transport, no preprocessing."""
        if self._rs_wall_start is None or self._rs_wall_end is None:
            return float("nan")
        return self._rs_wall_end - self._rs_wall_start

    @property
    def cpu_time_run_source(self) -> float:
        if self._rs_cpu_start is None or self._rs_cpu_end is None:
            return float("nan")
        return self._rs_cpu_end - self._rs_cpu_start

    @property
    def time_total(self) -> float:
        """Preprocessing + run_source wall time."""
        rs = self.time_run_source
        if not math.isfinite(rs):
            return float("nan")
        return self.time_preprocessing + rs

    @property
    def cpu_time_total(self) -> float:
        rs = self.cpu_time_run_source
        if not math.isfinite(rs):
            return float("nan")
        return self.cpu_time_preprocessing + rs

    # backward-compatible aliases used by existing code
    @property
    def total_time(self) -> float:
        return self.time_run_source

    @property
    def total_cpu_time(self) -> float:
        return self.cpu_time_run_source

    @property
    def neutrons_per_second(self) -> float:
        """Throughput based on run-source time only (no preprocessing)."""
        rs = self.time_run_source
        if not math.isfinite(rs) or rs <= 0.0:
            return 0.0
        return self.n_neutrons / rs

    @property
    def rejection_fraction(self) -> float:
        total = self.n_real_collisions + self.n_virtual_collisions
        return self.n_virtual_collisions / total if total > 0 else 0.0

    @property
    def wrong_majorant_fraction(self) -> float:
        return (self.n_wrong_majorant / self.n_majorant_updates
                if self.n_majorant_updates > 0 else 0.0)

    @property
    def wrong_majorant_mean_error(self) -> float:
        return (self.n_wrong_majorant_mean_error / self.n_wrong_majorant
                if self.n_wrong_majorant > 0 else 0.0)

    @property
    def cpu_efficiency(self) -> float:
        rs = self.time_run_source
        ct = self.cpu_time_run_source
        if not (math.isfinite(rs) and rs > 0.0 and math.isfinite(ct)):
            return float("nan")
        return ct / rs

    # ─────────────────────────────────────────────────────────────────────────
    # SNAPSHOT
    # ─────────────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            # ── four timers ───────────────────────────────────────────────
            "time_preprocessing_s"     : self.time_preprocessing,
            "cpu_time_preprocessing_s" : self.cpu_time_preprocessing,
            "time_run_source_s"        : self.time_run_source,
            "cpu_time_run_source_s"    : self.cpu_time_run_source,
            "time_total_s"             : self.time_total,
            "cpu_time_total_s"         : self.cpu_time_total,
            # backward-compat keys expected by _compute_batch_stats / export
            "total_time_s"             : self.time_run_source,
            "total_cpu_time_s"         : self.cpu_time_run_source,
            # ── sub-region timers ─────────────────────────────────────────
            "time_majorant_s"          : self.time_majorant,
            "cpu_time_majorant_s"      : self.cpu_time_majorant,
            "n_majorant_updates"       : self.n_majorant_updates,
            "time_xs_eval_s"           : self.time_xs_eval,
            "cpu_time_xs_eval_s"       : self.cpu_time_xs_eval,
            "n_xs_evaluations"         : self.n_xs_evaluations,
            # ── throughput / counters ─────────────────────────────────────
            "neutrons_per_second"      : self.neutrons_per_second,
            "n_neutrons"               : self.n_neutrons,
            "n_real_collisions"        : self.n_real_collisions,
            "n_virtual_collisions"     : self.n_virtual_collisions,
            "rejection_fraction"       : self.rejection_fraction,
            "cpu_efficiency"           : self.cpu_efficiency,
            # ── majorant quality ──────────────────────────────────────────
            "n_wrong_majorant"         : self.n_wrong_majorant,
            "wrong_majorant_mean_error": self.wrong_majorant_mean_error,
            "wrong_majorant_fraction"  : self.wrong_majorant_fraction,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        W = 64

        def row(label, wall, cpu=""):
            cpu_str = f"{cpu:>10.4f}" if isinstance(cpu, float) else f"{'':>10}"
            return f"  {label:<34} {wall:>10.4f}  {cpu_str}"

        def srow(label, val):
            return f"  {label:<34} {val}"

        lines = [
            "=" * W,
            "  PERFORMANCE SUMMARY",
            "=" * W,
            f"  {'Metric':<34} {'Wall (s)':>10}  {'CPU (s)':>10}",
            "-" * W,
            row("Preprocessing",
                self.time_preprocessing, self.cpu_time_preprocessing),
            row("Run-source (transport only)",
                self.time_run_source,    self.cpu_time_run_source),
            row("Total  (preprocessing + run)",
                self.time_total,         self.cpu_time_total),
            "-" * W,
            row("  ↳ Majorant XS evaluations",
                self.time_majorant,      self.cpu_time_majorant),
            row("  ↳ Local XS evaluations",
                self.time_xs_eval,       self.cpu_time_xs_eval),
            "-" * W,
            f"  {'CPU efficiency (run-source)':<34} {self.cpu_efficiency:>10.3f}"
            f"    [cpu/wall, per-process]",
            f"  {'Neutrons / second':<34} {self.neutrons_per_second:>10.1f}"
            f"    [based on run-source time]",
            "-" * W,
            srow("Neutrons simulated",
                 f"{self.n_neutrons:>10,}"),
            srow("Majorant updates",
                 f"{self.n_majorant_updates:>10,}"),
            srow("XS evaluations",
                 f"{self.n_xs_evaluations:>10,}"),
            srow("Real collisions",
                 f"{self.n_real_collisions:>10,}"),
            srow("Virtual (rejected)",
                 f"{self.n_virtual_collisions:>10,}"),
            srow("Rejection fraction",
                 f"{self.rejection_fraction:>10.3%}"),
            srow("Wrong majorant fraction",
                 f"{self.wrong_majorant_fraction:>10.3%}"),
            srow("Wrong majorant mean error",
                 f"{self.wrong_majorant_mean_error:>10.3%}"),
            "=" * W,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MajorantRecord / NeutronHistory  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MajorantRecord:
    energy:            float
    value:             float
    limiting_material: str
    actual_max_xs:     float

    @property
    def margin(self) -> float:
        return self.value / self.actual_max_xs if self.actual_max_xs > 0 else float("inf")


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
        return (1.0 - self.final_energy / self.birth_energy) if self.birth_energy > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BatchTimer  (unchanged from prior version)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchTimer:
    label:            str           = "batch"
    n_workers:        Optional[int] = None
    n_batches:        Optional[int] = None
    n_neutrons_total: Optional[int] = None

    wall_total:         float = 0.0
    cpu_total_main:     float = 0.0
    sum_worker_wall:    float = 0.0
    sum_worker_cpu:     float = 0.0
    sum_worker_pp_wall: float = 0.0
    sum_worker_pp_cpu:  float = 0.0

    _wall_t0: Optional[float] = field(default=None, repr=False)
    _cpu_t0:  Optional[float] = field(default=None, repr=False)

    def __enter__(self):
        self._wall_t0 = time.perf_counter()
        self._cpu_t0  = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wall_total     = time.perf_counter() - self._wall_t0
        self.cpu_total_main = time.process_time() - self._cpu_t0
        return False

    def attach_worker_records(self, batch_results: list):
        self.n_batches          = len(batch_results)
        self.sum_worker_wall    = 0.0
        self.sum_worker_cpu     = 0.0
        self.sum_worker_pp_wall = 0.0
        self.sum_worker_pp_cpu  = 0.0
        n_neutrons_total        = 0
        for r in batch_results:
            p  = r.get("perf", {})
            wt = p.get("time_run_source_s", p.get("total_time_s", float("nan")))
            ct = p.get("cpu_time_run_source_s", p.get("total_cpu_time_s", float("nan")))
            if math.isfinite(wt):  self.sum_worker_wall += wt
            if math.isfinite(ct):  self.sum_worker_cpu  += ct
            self.sum_worker_pp_wall += p.get("time_preprocessing_s", 0.0)
            self.sum_worker_pp_cpu  += p.get("cpu_time_preprocessing_s", 0.0)
            n_neutrons_total        += int(p.get("n_neutrons", 0))
        self.n_neutrons_total = n_neutrons_total

    @property
    def parallel_efficiency(self) -> float:
        nw = self.n_workers
        if not nw or nw <= 0 or self.wall_total <= 0.0 or not math.isfinite(self.sum_worker_cpu):
            return float("nan")
        return self.sum_worker_cpu / (nw * self.wall_total)

    @property
    def overall_throughput(self) -> float:
        if self.n_neutrons_total is None or self.wall_total <= 0.0:
            return 0.0
        return self.n_neutrons_total / self.wall_total

    @classmethod
    def compute_speedup(cls, serial: "BatchTimer", parallel: "BatchTimer") -> float:
        if serial.wall_total <= 0.0 or parallel.wall_total <= 0.0:
            return float("nan")
        return serial.wall_total / parallel.wall_total

    def summary(self) -> str:
        nw = self.n_workers if self.n_workers else 1
        lines = [
            "=" * 60,
            f"  BATCH TIMER — {self.label}",
            "=" * 60,
            f"  n_workers          : {nw}",
            f"  n_batches          : {self.n_batches}",
            (f"  n_neutrons (total) : {self.n_neutrons_total:,}"
             if self.n_neutrons_total is not None else ""),
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
        return "\n".join(l for l in lines if l)