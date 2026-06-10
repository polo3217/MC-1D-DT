"""
performance_classes.py
======================
Timing structure:

    time_preprocessing_s   : wall time in set_maj_xs_method, set_access_method, etc.
    time_run_source_s      : wall time of run_source() — pure neutron transport.
                             Used for neutrons_per_second.
    time_total_s           : time_preprocessing_s + time_run_source_s

Energy-bin tracking
-------------------
Instead of a binary WMP / ENDF split, performance is classified by energy
into user-defined bins whose boundaries are automatically merged with the
WMP E_max values of all nuclides.  Each bin knows exactly which nuclides
are evaluated via WMP and which fall back to ENDF, giving a physically
meaningful breakdown of computational cost vs energy regime.

Setup (call once after set_maj_xs_method, before run_source):

    geom.perf.setup_bins(
        user_bounds  = [1e-5, 1e2, 1e4, 2e7],   # eV, optional
        nuclide_wmp  = {                          # from nuclide_objects
            "U238": (E_min_U238, E_max_U238),
            "Na23": (E_min_Na23, E_max_Na23),
            ...
        }
    )

After setup, all score_*() calls route to the correct bin automatically.
"""

import bisect
import math
import os
import time
import threading
import array # CHANGE: Added import for Option 1 (efficient array storage)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import tracemalloc

import numpy as np
import pandas as pd
import psutil


# ─────────────────────────────────────────────────────────────────────────────
# EnergyBinStats  — per-bin accumulator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnergyBinStats:
    """
    All tracked quantities for one energy bin.

    e_lo, e_hi        : bin boundaries in eV
    label             : human-readable label, e.g.
                        "1.00e-05–1.00e+02 eV  WMP:[U238,Na23]  ENDF:[Fe56]"
    nuclides_wmp      : nuclides evaluated via WMP in this bin
    nuclides_endf     : nuclides evaluated via ENDF (outside WMP range) in this bin
    """
    e_lo:             float
    e_hi:             float
    label:            str
    nuclides_wmp:     List[str] = field(default_factory=list)
    nuclides_endf:    List[str] = field(default_factory=list)

    # majorant evaluation
    time_majorant_s:      float = 0.0
    cpu_time_majorant_s:  float = 0.0
    n_majorant_updates:   int   = 0

    # local XS evaluation (acceptance step)
    time_xs_eval_s:       float = 0.0
    cpu_time_xs_eval_s:   float = 0.0
    n_xs_evaluations:     int   = 0

    # collisions
    n_real_collisions:    int   = 0
    n_virtual_collisions: int   = 0

    # wrong majorant
    n_wrong_majorant:     int   = 0
    wrong_majorant_error: float = 0.0

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def us_per_majorant(self) -> float:
        return (1e6 * self.time_majorant_s / self.n_majorant_updates
                if self.n_majorant_updates > 0 else float("nan"))

    @property
    def us_per_xs_eval(self) -> float:
        return (1e6 * self.time_xs_eval_s / self.n_xs_evaluations
                if self.n_xs_evaluations > 0 else float("nan"))

    @property
    def rejection_fraction(self) -> float:
        total = self.n_real_collisions + self.n_virtual_collisions
        return self.n_virtual_collisions / total if total > 0 else float("nan")

    @property
    def wrong_majorant_fraction(self) -> float:
        return (self.n_wrong_majorant / self.n_majorant_updates
                if self.n_majorant_updates > 0 else float("nan"))

    @property
    def wrong_majorant_mean_error(self) -> float:
        return (self.wrong_majorant_error / self.n_wrong_majorant
                if self.n_wrong_majorant > 0 else float("nan"))

    # ── accumulation ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset counters, keep bin definition (boundaries, labels, nuclide lists)."""
        self.time_majorant_s      = 0.0
        self.cpu_time_majorant_s  = 0.0
        self.n_majorant_updates   = 0
        self.time_xs_eval_s       = 0.0
        self.cpu_time_xs_eval_s   = 0.0
        self.n_xs_evaluations     = 0
        self.n_real_collisions    = 0
        self.n_virtual_collisions = 0
        self.n_wrong_majorant     = 0
        self.wrong_majorant_error = 0.0

    def merge(self, other: "EnergyBinStats") -> None:
        """Fold another bin's counters into this one (for parallel merge)."""
        self.time_majorant_s      += other.time_majorant_s
        self.cpu_time_majorant_s  += other.cpu_time_majorant_s
        self.n_majorant_updates   += other.n_majorant_updates
        self.time_xs_eval_s       += other.time_xs_eval_s
        self.cpu_time_xs_eval_s   += other.cpu_time_xs_eval_s
        self.n_xs_evaluations     += other.n_xs_evaluations
        self.n_real_collisions    += other.n_real_collisions
        self.n_virtual_collisions += other.n_virtual_collisions
        self.n_wrong_majorant     += other.n_wrong_majorant
        self.wrong_majorant_error += other.wrong_majorant_error

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for snapshot / parallel transport)."""
        return {
            "e_lo":                   self.e_lo,
            "e_hi":                   self.e_hi,
            "label":                  self.label,
            "nuclides_wmp":           list(self.nuclides_wmp),
            "nuclides_endf":          list(self.nuclides_endf),
            "time_majorant_s":        self.time_majorant_s,
            "cpu_time_majorant_s":    self.cpu_time_majorant_s,
            "n_majorant_updates":     self.n_majorant_updates,
            "time_xs_eval_s":         self.time_xs_eval_s,
            "cpu_time_xs_eval_s":     self.cpu_time_xs_eval_s,
            "n_xs_evaluations":       self.n_xs_evaluations,
            "n_real_collisions":      self.n_real_collisions,
            "n_virtual_collisions":   self.n_virtual_collisions,
            "n_wrong_majorant":       self.n_wrong_majorant,
            "wrong_majorant_error":   self.wrong_majorant_error,
            # derived — included so callers can read without re-computing
            "us_per_majorant":        self.us_per_majorant,
            "us_per_xs_eval":         self.us_per_xs_eval,
            "rejection_fraction":     self.rejection_fraction,
            "wrong_majorant_fraction":    self.wrong_majorant_fraction,
            "wrong_majorant_mean_error":  self.wrong_majorant_mean_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EnergyBinStats":
        b = cls(
            e_lo          = d["e_lo"],
            e_hi          = d["e_hi"],
            label         = d["label"],
            nuclides_wmp  = d.get("nuclides_wmp",  []),
            nuclides_endf = d.get("nuclides_endf", []),
        )
        b.time_majorant_s      = d.get("time_majorant_s",      0.0)
        b.cpu_time_majorant_s  = d.get("cpu_time_majorant_s",  0.0)
        b.n_majorant_updates   = d.get("n_majorant_updates",   0)
        b.time_xs_eval_s       = d.get("time_xs_eval_s",       0.0)
        b.cpu_time_xs_eval_s   = d.get("cpu_time_xs_eval_s",   0.0)
        b.n_xs_evaluations     = d.get("n_xs_evaluations",     0)
        b.n_real_collisions    = d.get("n_real_collisions",    0)
        b.n_virtual_collisions = d.get("n_virtual_collisions", 0)
        b.n_wrong_majorant     = d.get("n_wrong_majorant",     0)
        b.wrong_majorant_error = d.get("wrong_majorant_error", 0.0)
        return b


# ─────────────────────────────────────────────────────────────────────────────
# MemorySnapshot / MemoryTracker  (unchanged except for arrays and threshold)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    label:     str
    timestamp: float
    rss_mb:    float
    vms_mb:    float


class MemoryTracker:
    def __init__(self, poll_interval: float = 0.1, name: str = "MemoryTracker"):
        self._name          = name
        self._process       = psutil.Process(os.getpid())
        self._snapshots:    List[MemorySnapshot] = []
        self._poll_interval = poll_interval
        self._t0            = time.perf_counter()
        
        # CHANGE: Option 1 - Use array module for flat, efficient storage instead of List[float]
        self._poll_rss      = array.array('f')
        self._poll_times    = array.array('d')
        
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
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True,
                                         name="MemoryPoller")
        self._thread.start()
        self.snapshot("init")
        #print(f"\n [Memory] Tracker started (poll interval: "f"{self._poll_interval*1000:.0f} ms)")

    def stop(self):
        self._polling = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.snapshot("stop")
        #print(f"\n [Memory] Tracker stopped.")

    def _poll_loop(self):
        # CHANGE: Option 2 - Threshold variables to prevent logging redundant data points
        last_recorded_rss = -1.0
        threshold_mb = 0.1  # Record only if memory changes by at least 0.1 MB

        while self._polling:
            try:
                rss = self._process.memory_info().rss / 1e6
                t   = time.perf_counter() - self._t0
                
                # Always accurately track the peak regardless of the threshold
                if rss > self._peak_rss_mb:
                    self._peak_rss_mb = rss
                
                # CHANGE: Option 2 - Only append to arrays if the jump is significant
                if abs(rss - last_recorded_rss) >= threshold_mb:
                    self._poll_rss.append(rss)
                    self._poll_times.append(t)
                    last_recorded_rss = rss

            except psutil.NoSuchProcess:
                break
            time.sleep(self._poll_interval)

    def snapshot(self, label: str, external_rss_mb: float = None) -> MemorySnapshot:
        if external_rss_mb is not None:
            rss, vms = external_rss_mb, 0.0
        else:
            mem = self._process.memory_info()
            rss, vms = mem.rss / 1e6, mem.vms / 1e6
        snap = MemorySnapshot(label=label,
                              timestamp=time.perf_counter() - self._t0,
                              rss_mb=rss, vms_mb=vms)
        self._snapshots.append(snap)
        if rss > self._peak_rss_mb:
            self._peak_rss_mb = rss
        return snap

    def peak_mb(self) -> float:    return self._peak_rss_mb
    def current_mb(self) -> float: return self._process.memory_info().rss / 1e6

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
    Performance tracker with energy-bin breakdown.

    Energy bins replace the old binary WMP/ENDF split.  Each bin covers
    an energy interval and knows exactly which nuclides use WMP vs ENDF
    in that interval.

    Setup (call after nuclide_objects are loaded, before run_source):
        geom.perf.setup_bins(
            user_bounds = [1e-5, 1e2, 1e4, 2e7],  # optional
            nuclide_wmp = {"U238": (E_min, E_max), "Na23": (E_min, E_max), ...}
        )
    """

    # ── preprocessing timer ──────────────────────────────────────────────────
    _pp_wall_start: Optional[float] = field(default=None, repr=False)
    _pp_cpu_start:  Optional[float] = field(default=None, repr=False)

    time_preprocessing:     float = 0.0
    cpu_time_preprocessing: float = 0.0

    # ── run-source timer ─────────────────────────────────────────────────────
    _rs_wall_start: Optional[float] = field(default=None, repr=False)
    _rs_wall_end:   Optional[float] = field(default=None, repr=False)
    _rs_cpu_start:  Optional[float] = field(default=None, repr=False)
    _rs_cpu_end:    Optional[float] = field(default=None, repr=False)

    # ── totals (summed over all energy bins) ─────────────────────────────────
    time_majorant_s:     float = 0.0
    cpu_time_majorant_s: float = 0.0
    time_xs_eval_s:      float = 0.0
    cpu_time_xs_eval_s:  float = 0.0

    n_neutrons:           int = 0
    n_real_collisions:    int = 0
    n_virtual_collisions: int = 0
    n_majorant_updates:   int = 0
    n_xs_evaluations:     int = 0
    n_wrong_majorant:     int = 0
    wrong_majorant_error: float = 0.0

    wrong_majorant_energies: List[List[float]] = field(default_factory=list)

    # ── energy bins ──────────────────────────────────────────────────────────
    # Populated by setup_bins().  Empty until called.
    bins: List[EnergyBinStats] = field(default_factory=list)
    # Sorted list of upper boundaries for O(log n) bin lookup.
    # _bin_edges[i] is the upper boundary of bins[i].
    _bin_edges: List[float] = field(default_factory=list, repr=False)

    # ── memory fields (written by geometry_classes) ──────────────────────────
    baseline_rss_mb: float = 0.0
    setup_mem_mb:    float = 0.0
    peak_mem_mb:     float = 0.0

    # ── tracemalloc (opt-in) ─────────────────────────────────────────────────
    python_heap_setup_kb:     float = 0.0
    python_heap_transport_kb: float = 0.0
    _tracemalloc_enabled: bool = field(default=False, repr=False)
    _tm_pp_before:        int  = field(default=0,     repr=False)
    _tm_rs_before:        int  = field(default=0,     repr=False)

    # ─────────────────────────────────────────────────────────────────────────
    # ENERGY BIN SETUP
    # ─────────────────────────────────────────────────────────────────────────

    def setup_bins(
        self,
        nuclide_wmp: Dict[str, tuple],
        user_bounds: Optional[List[float]] = None,
    ) -> None:
        """
        Build the energy bin structure.

        Parameters
        ----------
        nuclide_wmp : dict mapping nuclide_name -> (E_min, E_max)
            WMP energy range for every nuclide in the geometry.
            Obtained from nuclide_objects after set_maj_xs_method() loads them.
            Example:
                {n: (nuclide_objects[n]['wmp'].E_min,
                     nuclide_objects[n]['wmp'].E_max)
                 for n in geometry.nuclides if nuclide_objects[n]['wmp']}

        user_bounds : list of floats, optional
            Additional energy boundaries to include. The WMP E_min and E_max
            of every nuclide are always added automatically.
            If None, only the WMP boundaries are used.

        After calling this, all score_*() methods route to the correct bin.
        The bin structure survives reset_counters(keep_preprocessing=True).
        """
        # Collect all boundaries: user-defined + WMP E_min + WMP E_max
        boundaries = set()
        if user_bounds:
            boundaries.update(user_bounds)
        for name, (e_min, e_max) in nuclide_wmp.items():
            boundaries.add(e_min)
            boundaries.add(e_max)

        # Sort and deduplicate
        edges = sorted(boundaries)

        # Need at least two boundaries to form one bin
        if len(edges) < 2:
            raise ValueError(
                "setup_bins requires at least 2 boundaries (got "
                f"{len(edges)}). Pass user_bounds or ensure nuclide_wmp "
                "contains at least one nuclide with E_min != E_max."
            )

        self.bins = []
        self._bin_edges = []

        for i in range(len(edges) - 1):
            e_lo = edges[i]
            e_hi = edges[i + 1]
            mid  = (e_lo + e_hi) / 2.0

            # Classify nuclides at the midpoint of this bin
            wmp_here  = [n for n, (mn, mx) in nuclide_wmp.items()
                         if mn <= mid <= mx]
            endf_here = [n for n in nuclide_wmp if n not in wmp_here]

            # Build a readable label
            def _short(lst):
                if not lst:        return "none"
                if len(lst) <= 3:  return ",".join(lst)
                return f"{','.join(lst[:3])}+{len(lst)-3}"

            lbl = (f"{e_lo:.2e}–{e_hi:.2e} eV  "
                   f"WMP:[{_short(wmp_here)}]  "
                   f"ENDF:[{_short(endf_here)}]")

            self.bins.append(EnergyBinStats(
                e_lo          = e_lo,
                e_hi          = e_hi,
                label         = lbl,
                nuclides_wmp  = wmp_here,
                nuclides_endf = endf_here,
            ))
            self._bin_edges.append(e_hi)

        print(f"[PerformanceTracker] {len(self.bins)} energy bins set up.")
        for b in self.bins:
            print(f"  {b.label}")

    def _find_bin(self, energy: float) -> int:
        """
        Return the index of the bin that contains `energy`.
        Uses bisect for O(log n) lookup.
        Returns -1 if no bins are set up or energy is out of range.
        """
        if not self._bin_edges:
            return -1
        idx = bisect.bisect_left(self._bin_edges, energy)
        if idx >= len(self.bins):
            idx = len(self.bins) - 1
        return idx

    # ─────────────────────────────────────────────────────────────────────────
    # SCORING METHODS  (called from geometry_classes hot path)
    # ─────────────────────────────────────────────────────────────────────────

    def score_majorant(self, energy: float, wall_s: float, cpu_s: float) -> None:
        """
        Record one majorant XS evaluation.
        Called from get_majorant_xs() after the timer block.
        """
        self.time_majorant_s     += wall_s
        self.cpu_time_majorant_s += cpu_s
        self.n_majorant_updates  += 1
        i = self._find_bin(energy)
        if i >= 0:
            b = self.bins[i]
            b.time_majorant_s     += wall_s
            b.cpu_time_majorant_s += cpu_s
            b.n_majorant_updates  += 1

    def score_xs_eval(self, energy: float, wall_s: float, cpu_s: float) -> None:
        """
        Record one local XS evaluation (cache miss in _evaluate_acceptance,
        or virtual-collision XS in _run_neutron).
        """
        self.time_xs_eval_s     += wall_s
        self.cpu_time_xs_eval_s += cpu_s
        self.n_xs_evaluations   += 1
        i = self._find_bin(energy)
        if i >= 0:
            b = self.bins[i]
            b.time_xs_eval_s     += wall_s
            b.cpu_time_xs_eval_s += cpu_s
            b.n_xs_evaluations   += 1

    def score_collision(self, energy: float, kind: str) -> None:
        """
        Record a real or virtual collision.

        Parameters
        ----------
        energy : float
        kind   : 'real' or 'virtual'
        """
        if kind == 'real':
            self.n_real_collisions += 1
            i = self._find_bin(energy)
            if i >= 0:
                self.bins[i].n_real_collisions += 1
        else:
            self.n_virtual_collisions += 1
            i = self._find_bin(energy)
            if i >= 0:
                self.bins[i].n_virtual_collisions += 1

    def score_wrong_majorant(self, energy: float, error: float) -> None:
        """
        Record a wrong-majorant event (acceptance_prob > 1).

        Parameters
        ----------
        energy : float
        error  : acceptance_prob - 1.0  (fractional overshoot)
        """
        self.n_wrong_majorant     += 1
        self.wrong_majorant_error += error
        i = self._find_bin(energy)
        if i >= 0:
            self.bins[i].n_wrong_majorant     += 1
            self.bins[i].wrong_majorant_error += error

    # ─────────────────────────────────────────────────────────────────────────
    # PARALLEL SUPPORT
    # ─────────────────────────────────────────────────────────────────────────

    def bins_to_snapshot(self) -> List[dict]:
        """
        Return a picklable list of bin dicts for shipping from worker to parent.
        Called in _run_single_batch_worker after geom.run_source().
        """
        return [b.to_dict() for b in self.bins]

    def merge_bins(self, worker_bin_snapshots: List[List[dict]]) -> None:
        """
        Fold per-worker bin snapshots into this instance.
        Called in run_batch (parent process) after pool.map() returns.

        Parameters
        ----------
        worker_bin_snapshots : list of lists
            Each element is the result of one worker's bins_to_snapshot() call.
            [ [bin0_dict, bin1_dict, ...],   # worker 0
              [bin0_dict, bin1_dict, ...],   # worker 1
              ... ]
        """
        if not self.bins:
            # Bins not set up in parent — reconstruct from first worker
            if worker_bin_snapshots and worker_bin_snapshots[0]:
                self.bins = [EnergyBinStats.from_dict(d)
                             for d in worker_bin_snapshots[0]]
                self._bin_edges = [b.e_hi for b in self.bins]
                # Merge the rest
                for snap in worker_bin_snapshots[1:]:
                    for i, d in enumerate(snap):
                        if i < len(self.bins):
                            self.bins[i].merge(EnergyBinStats.from_dict(d))
            return

        for snap in worker_bin_snapshots:
            for i, d in enumerate(snap):
                if i < len(self.bins):
                    self.bins[i].merge(EnergyBinStats.from_dict(d))

    # ─────────────────────────────────────────────────────────────────────────
    # TRACEMALLOC
    # ─────────────────────────────────────────────────────────────────────────

    def enable_tracemalloc(self, depth: int = 5) -> None:
        """
        Opt in to Python-heap allocation tracking.
        Call once before set_maj_xs_method().

        Parallel behaviour
        ------------------
        _tracemalloc_enabled is pickled with the geometry.
        In parallel.py, _run_single_batch_worker must start tracemalloc
        if this flag is True (see PATCH P1 in rigorous_profiler.py).
        """
        self._tracemalloc_enabled = True
        if not tracemalloc.is_tracing():
            tracemalloc.start(depth)

    def top_allocations(self, n: int = 15) -> str:
        """Top-N Python allocation sites. Requires enable_tracemalloc() first."""
        if not tracemalloc.is_tracing():
            return "tracemalloc not running — call enable_tracemalloc() first."
        stats = tracemalloc.take_snapshot().statistics("traceback")
        lines = [f"Top {n} Python allocation sites"]
        for i, stat in enumerate(stats[:n], 1):
            lines.append(f"\n#{i}  {stat.size/1024:.1f} kB  ({stat.count:,} blocks)")
            for frame in stat.traceback:
                lines.append(f"    {frame.filename}:{frame.lineno}")
        return "\n".join(lines)

    def current_heap_kb(self) -> float:
        """Total Python-heap bytes alive right now, in kB. 0 if tracemalloc off."""
        if not tracemalloc.is_tracing():
            return 0.0
        current, _peak = tracemalloc.get_traced_memory()
        return current / 1024

    # ─────────────────────────────────────────────────────────────────────────
    # PREPROCESSING TIMER
    # ─────────────────────────────────────────────────────────────────────────

    def start_preprocessing(self):
        self._pp_wall_start = time.perf_counter()
        self._pp_cpu_start  = time.process_time()
        if self._tracemalloc_enabled and tracemalloc.is_tracing():
            tracemalloc.reset_peak()
            self._tm_pp_before, _ = tracemalloc.get_traced_memory()
        else:
            self._tm_pp_before = 0

    def stop_preprocessing(self):
        if self._pp_wall_start is None:
            raise RuntimeError(
                "stop_preprocessing() called without start_preprocessing()")
        self.time_preprocessing     += time.perf_counter() - self._pp_wall_start
        self.cpu_time_preprocessing += time.process_time()  - self._pp_cpu_start
        self._pp_wall_start = None
        self._pp_cpu_start  = None
        if self._tracemalloc_enabled and tracemalloc.is_tracing():
            _, peak_setup = tracemalloc.get_traced_memory()
            self.python_heap_setup_kb += peak_setup / 1024
        self._tm_pp_before = 0

    # ─────────────────────────────────────────────────────────────────────────
    # RUN-SOURCE TIMER
    # ─────────────────────────────────────────────────────────────────────────

    def start(self):
        self._rs_wall_start = time.perf_counter()
        self._rs_cpu_start  = time.process_time()
        self._rs_wall_end   = None
        self._rs_cpu_end    = None
        if self._tracemalloc_enabled and tracemalloc.is_tracing():
            tracemalloc.reset_peak()
            self._tm_rs_before, _ = tracemalloc.get_traced_memory()
        else:
            self._tm_rs_before = 0

    def stop(self):
        if self._rs_wall_end is not None:
            return
        if self._rs_wall_start is None:
            raise RuntimeError("stop() called without start()")
        self._rs_wall_end = time.perf_counter()
        self._rs_cpu_end  = time.process_time()
        if self._tracemalloc_enabled and tracemalloc.is_tracing():
            _, peak_transport = tracemalloc.get_traced_memory()
            self.python_heap_transport_kb = peak_transport / 1024
        self._tm_rs_before = 0

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset_counters(self, keep_preprocessing: bool = True):
        # Save what needs to survive
        saved_pp_wall             = self.time_preprocessing
        saved_pp_cpu              = self.cpu_time_preprocessing
        saved_baseline_rss        = self.baseline_rss_mb
        saved_setup_mem           = self.setup_mem_mb
        saved_python_heap_setup   = self.python_heap_setup_kb

        # Clear run-source timers
        self._rs_wall_start = None
        self._rs_wall_end   = None
        self._rs_cpu_start  = None
        self._rs_cpu_end    = None
        self._pp_wall_start = None
        self._pp_cpu_start  = None

        # Reset totals
        self.time_majorant_s      = 0.0
        self.cpu_time_majorant_s  = 0.0
        self.time_xs_eval_s       = 0.0
        self.cpu_time_xs_eval_s   = 0.0
        self.n_neutrons           = 0
        self.n_real_collisions    = 0
        self.n_virtual_collisions = 0
        self.n_majorant_updates   = 0
        self.n_xs_evaluations     = 0
        self.n_wrong_majorant     = 0
        self.wrong_majorant_error = 0.0
        self.wrong_majorant_energies = []
        self.peak_mem_mb             = 0.0
        self.python_heap_transport_kb = 0.0

        # Reset per-bin counters (keep bin structure — boundaries and labels)
        for b in self.bins:
            b.reset()

        if keep_preprocessing:
            self.time_preprocessing   = saved_pp_wall
            self.cpu_time_preprocessing = saved_pp_cpu
            self.baseline_rss_mb      = saved_baseline_rss
            self.setup_mem_mb         = saved_setup_mem
            self.python_heap_setup_kb = saved_python_heap_setup
        else:
            self.time_preprocessing     = 0.0
            self.cpu_time_preprocessing = 0.0
            self.baseline_rss_mb        = 0.0
            self.setup_mem_mb           = 0.0
            self.python_heap_setup_kb   = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # DERIVED PROPERTIES
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def time_run_source(self) -> float:
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
        rs = self.time_run_source
        return float("nan") if not math.isfinite(rs) else self.time_preprocessing + rs

    @property
    def cpu_time_total(self) -> float:
        rs = self.cpu_time_run_source
        return float("nan") if not math.isfinite(rs) else self.cpu_time_preprocessing + rs

    # backward-compat aliases
    @property
    def total_time(self) -> float:    return self.time_run_source
    @property
    def total_cpu_time(self) -> float: return self.cpu_time_run_source

    @property
    def neutrons_per_second(self) -> float:
        rs = self.time_run_source
        if not math.isfinite(rs) or rs <= 0.0:
            return 0.0
        return self.n_neutrons / rs

    @property
    def rejection_fraction(self) -> float:
        total = self.n_real_collisions + self.n_virtual_collisions
        return self.n_virtual_collisions / total if total > 0 else float("nan")

    @property
    def wrong_majorant_fraction(self) -> float:
        return (self.n_wrong_majorant / self.n_majorant_updates
                if self.n_majorant_updates > 0 else float("nan"))

    @property
    def wrong_majorant_mean_error(self) -> float:
        return (self.wrong_majorant_error / self.n_wrong_majorant
                if self.n_wrong_majorant > 0 else float("nan"))

    @property
    def us_per_majorant(self) -> float:
        return (1e6 * self.time_majorant_s / self.n_majorant_updates
                if self.n_majorant_updates > 0 else float("nan"))

    @property
    def us_per_xs_eval(self) -> float:
        return (1e6 * self.time_xs_eval_s / self.n_xs_evaluations
                if self.n_xs_evaluations > 0 else float("nan"))

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
        d = {
            # timing
            "time_preprocessing_s":    self.time_preprocessing,
            "cpu_time_preprocessing_s": self.cpu_time_preprocessing,
            "time_run_source_s":        self.time_run_source,
            "cpu_time_run_source_s":    self.cpu_time_run_source,
            "time_total_s":             self.time_total,
            "cpu_time_total_s":         self.cpu_time_total,
            "total_time_s":             self.time_run_source,   # backward compat
            "total_cpu_time_s":         self.cpu_time_run_source,
            # sub-region totals
            "time_majorant_s":          self.time_majorant_s,
            "cpu_time_majorant_s":      self.cpu_time_majorant_s,
            "time_xs_eval_s":           self.time_xs_eval_s,
            "cpu_time_xs_eval_s":       self.cpu_time_xs_eval_s,
            # throughput
            "neutrons_per_second":      self.neutrons_per_second,
            "n_neutrons":               self.n_neutrons,
            # collision totals
            "n_real_collisions":        self.n_real_collisions,
            "n_virtual_collisions":     self.n_virtual_collisions,
            "n_majorant_updates":       self.n_majorant_updates,
            "n_xs_evaluations":         self.n_xs_evaluations,
            # quality totals
            "rejection_fraction":       self.rejection_fraction,
            "wrong_majorant_fraction":  self.wrong_majorant_fraction,
            "wrong_majorant_mean_error": self.wrong_majorant_mean_error,
            "n_wrong_majorant":         self.n_wrong_majorant,
            "wrong_majorant_energies":  self.wrong_majorant_energies,
            # derived µs/op
            "us_per_majorant_s":        self.us_per_majorant,
            "us_per_xs_eval_s":         self.us_per_xs_eval,
            # cpu
            "cpu_efficiency":           self.cpu_efficiency,
            # memory
            "baseline_rss_mb":          self.baseline_rss_mb,
            "setup_mem_mb":             self.setup_mem_mb,
            "peak_mem_mb":              self.peak_mem_mb,
            "python_heap_setup_kb":     self.python_heap_setup_kb,
            "python_heap_transport_kb": self.python_heap_transport_kb,
            # energy bins (list of dicts, one per bin)
            "energy_bins":              self.bins_to_snapshot(),
        }
        return d

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def _flt(self, v, fmt=".4f") -> str:
        return (f"{'N/A':>10}"
                if v is None or (isinstance(v, float) and math.isnan(v))
                else f"{v:>{10}{fmt}}")

    def pct(self, v) -> str:
        return (f"{'N/A':>10}"
                if v is None or math.isnan(v)
                else f"{v:>10.3%}")

    def summary(self) -> str:
        W = 74

        def row(label, wall, cpu=""):
            cpu_str = (f"{cpu:>10.4f}" if isinstance(cpu, float)
                       else f"{'':>10}")
            return f"  {label:<36} {wall:>10.4f}  {cpu_str}"

        def srow(label, val):
            return f"  {label:<36} {val}"

        lines = [
            "=" * W, "  PERFORMANCE SUMMARY", "=" * W, "",
            "=" * W, "  PREPROCESSING", "=" * W,
            f"  {'Metric':<36} {'Wall (s)':>10}  {'CPU (s)':>10}",
            "-" * W,
            row("Preprocessing",
                self.time_preprocessing, self.cpu_time_preprocessing),
            "",
            "=" * W, "  MEMORY  (RSS)", "=" * W,
            srow("Baseline RSS (MB)",
                 self._flt(self.baseline_rss_mb, ".1f")),
            srow("Setup memory peak (MB)",
                 self._flt(self.setup_mem_mb, ".1f")),
            srow("Peak during transport (MB)",
                 self._flt(self.peak_mem_mb, ".1f")),
            srow("Python heap: setup peak (kB)",
                 self._flt(self.python_heap_setup_kb, ".1f")
                 + "  [parent process only]"),
            srow("Python heap: transport peak (kB)",
                 self._flt(self.python_heap_transport_kb, ".1f")
                 + "  [per-worker in parallel mode]"),
            "",
            "=" * W, "  RUN-SOURCE", "=" * W,
            f"  {'Metric':<36} {'Wall (s)':>10}  {'CPU (s)':>10}",
            "-" * W,
            row("Run-source (transport only)",
                self.time_run_source, self.cpu_time_run_source),
            row("Total  (preprocessing + run)",
                self.time_total, self.cpu_time_total),
            "-" * W,
            row("  -> Majorant XS evaluations",
                self.time_majorant_s, self.cpu_time_majorant_s),
            row("  -> Local XS evaluations",
                self.time_xs_eval_s, self.cpu_time_xs_eval_s),
            "-" * W,
            f"  {'CPU efficiency (run-source)':<36} {self.cpu_efficiency:>10.3f}"
            f"    [cpu/wall, per-process]",
            f"  {'Neutrons / second':<36} {self.neutrons_per_second:>10.1f}"
            f"    [based on run-source time]",
            "",
            "=" * W, "  TOTALS", "=" * W,
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
                 self.pct(self.rejection_fraction)),
            srow("Wrong majorant fraction",
                 self.pct(self.wrong_majorant_fraction)),
            srow("Wrong majorant mean error",
                 self.pct(self.wrong_majorant_mean_error)),
            srow("µs per majorant lookup",
                 self._flt(self.us_per_majorant, ".4f")),
            srow("µs per XS eval",
                 self._flt(self.us_per_xs_eval, ".4f")),
        ]

        # ── Per-bin table ────────────────────────────────────────────────────
        if self.bins:
            lines += ["", "=" * W, "  PER-ENERGY-BIN BREAKDOWN", "=" * W]
            col_w = max(len(b.label) for b in self.bins) + 2
            hdr = (f"  {'Bin label':<{col_w}} {'t_maj(ms)':>10} "
                   f"{'t_xs(ms)':>10} {'µs/maj':>8} {'µs/xs':>8} "
                   f"{'rej%':>7} {'wrong%':>7} {'n_real':>9} {'n_virt':>9}")
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))
            for b in self.bins:
                def _f(v, fmt):
                    return f"{v:{fmt}}" if math.isfinite(v) else "  N/A"
                lines.append(
                    f"  {b.label:<{col_w}}"
                    f" {b.time_majorant_s*1000:>10.3f}"
                    f" {b.time_xs_eval_s*1000:>10.3f}"
                    f" {_f(b.us_per_majorant, '8.3f')}"
                    f" {_f(b.us_per_xs_eval,  '8.3f')}"
                    f" {_f(b.rejection_fraction*100, '7.2f')}"
                    f" {_f(b.wrong_majorant_fraction*100 if not math.isnan(b.wrong_majorant_fraction) else float('nan'), '7.4f')}"
                    f" {b.n_real_collisions:>9,}"
                    f" {b.n_virtual_collisions:>9,}"
                )

        lines.append("=" * W)
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
    def n_steps(self) -> int:        return len(self.events)
    @property
    def total_path_length(self) -> float:
        return float(np.sum(self.distances))
    @property
    def final_energy(self) -> float:
        return self.energies[-1] if self.energies else self.birth_energy
    @property
    def energy_loss_fraction(self) -> float:
        return (1.0 - self.final_energy / self.birth_energy
                if self.birth_energy > 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# BatchTimer  (unchanged)
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
            if math.isfinite(wt): self.sum_worker_wall += wt
            if math.isfinite(ct): self.sum_worker_cpu  += ct
            self.sum_worker_pp_wall += p.get("time_preprocessing_s", 0.0)
            self.sum_worker_pp_cpu  += p.get("cpu_time_preprocessing_s", 0.0)
            n_neutrons_total        += int(p.get("n_neutrons", 0))
        self.n_neutrons_total = n_neutrons_total

    @property
    def parallel_efficiency(self) -> float:
        nw = self.n_workers
        if not nw or nw <= 0 or self.wall_total <= 0.0:
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
            "=" * 60, f"  BATCH TIMER — {self.label}", "=" * 60,
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