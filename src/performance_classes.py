import sys
import os
sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')

from dataclasses import dataclass, field
import time
from datetime import datetime
from pathlib import Path
from itertools import accumulate

from typing import List, Optional
import numpy as np
import pandas as pd

import majorant_multipole as maj
import openmc
import reconr
import psutil

#==========================================
# --- Memory usage tracking (optional) ---
#==========================================
# src/memory_tracker.py

import os
import time
import threading
import psutil
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemorySnapshot:
    label       : str
    timestamp   : float
    rss_mb      : float
    vms_mb      : float


class MemoryTracker:
    """
    Tracks process memory usage over time using psutil.

    Two complementary mechanisms:
      1. Manual snapshots  — point-in-time, labeled, called explicitly
      2. Background poller — samples RSS every `poll_interval` seconds,
                             captures the true peak even between snapshots

    Usage
    -----
    tracker = MemoryTracker(poll_interval=0.1)
    tracker.start()

    geom.set_access_method("reconr", ...)
    tracker.snapshot("after_reconr")

    geom.run_batch(src, n_batches=10)
    tracker.snapshot("after_run")

    tracker.stop()
    print(tracker.summary())
    """

    def __init__(self, poll_interval: float = 0.1):
        """
        Parameters
        ----------
        poll_interval : float
            Seconds between background RSS samples. 
            0.01 = 10ms  (high resolution, negligible overhead)
            0.1  = 100ms (good default)
            1.0  = 1s    (low overhead, may miss short spikes)
        """
        self._process       = psutil.Process(os.getpid())
        self._snapshots     : List[MemorySnapshot] = []
        self._poll_interval = poll_interval
        self._t0            = time.perf_counter()

        # Background poller state
        self._poll_rss      : List[float] = []   # all sampled RSS values
        self._poll_times    : List[float] = []   # corresponding timestamps
        self._peak_rss_mb   = 0.0
        self._polling       = False
        self._thread        : Optional[threading.Thread] = None

    # make MemoryTracker picklable for multiprocessing (worker processes won't start the poller thread)
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove thread and event — not picklable
        state['_thread']  = None
        state['_polling'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore thread as None — worker never starts polling
        self._thread  = None
        self._polling = False

    # ── Background poller ─────────────────────────────────────────────────────

    def start(self):
        """Start the background memory polling thread."""
        if self._polling:
            return
        self._polling = True
        
        self._thread  = threading.Thread(
            target   = self._poll_loop,
            daemon   = True,    # dies automatically if main process exits
            name     = "MemoryPoller",
        )
        self._thread.start()
        self.snapshot("init")
        print(f"Memory tracker started (poll interval: {self._poll_interval*1000:.0f} ms)")

    def stop(self):
        """Stop the background polling thread."""
        self._polling = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.snapshot("stop")
        print("Memory tracker stopped.")

    def _poll_loop(self):
        """Runs in background thread — samples RSS continuously."""
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

    # ── Manual snapshots ──────────────────────────────────────────────────────

    def snapshot(self, label: str,
             external_rss_mb: float = None) -> MemorySnapshot:
        """
        Record memory usage with a label.

        Parameters
        ----------
        external_rss_mb : float, optional
            If provided, use this RSS value instead of reading from the
            current process. Used to record worker-process memory in
            parallel runs.
        """
        if external_rss_mb is not None:
            rss = external_rss_mb
            vms = 0.0   # not available from worker report
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
    
    # ── Accessors ─────────────────────────────────────────────────────────────

    def peak_mb(self) -> float:
        """True peak RSS — includes all background samples."""
        return self._peak_rss_mb

    def current_mb(self) -> float:
        """Current RSS without storing a snapshot."""
        return self._process.memory_info().rss / 1e6

    def delta_mb(self, label_a: str, label_b: str) -> float:
        """RSS delta in MB between two labeled snapshots."""
        a = self._get(label_a)
        b = self._get(label_b)
        if a is None or b is None:
            raise KeyError(f"Label not found: {label_a!r} or {label_b!r}")
        return b.rss_mb - a.rss_mb

    # ── DataFrames ────────────────────────────────────────────────────────────

    def snapshots_to_dataframe(self) -> pd.DataFrame:
        """Manual snapshots only."""
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
        """Full continuous RSS trace from background poller."""
        
        return pd.DataFrame({
            "time_s" : [round(t, 7) for t in self._poll_times],
            "rss_mb" : [round(r, 7) for r in self._poll_rss],
        })

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if not self._snapshots:
            return "No snapshots recorded."

        base  = self._snapshots[0].rss_mb
        df    = self.snapshots_to_dataframe()
        lines = [
            "=" * 65,
            "  MEMORY TRACKER SUMMARY",
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
# --- PerformanceTracker dataclass ---
# ==========================================
@dataclass
class PerformanceTracker:
    """
    Wall-clock AND CPU-time performance diagnostics.

    Wall time  : time.perf_counter()   — real elapsed time (includes I/O, sleep …)
    CPU time   : time.process_time()   — on-CPU time only (no I/O wait, no sleep)

    Both are measured for every timed region:
        • majorant XS evaluation
        • total simulation run

    Efficiency indicator
    -----------------------------------
    Calculate CPU and Wall time

    Usage
    -----
        tracker = PerformanceTracker()
        tracker.start()
        # ... run simulation ...
        tracker.stop()
        print(tracker.summary())
    """
    # ── internal start snapshots (not shown in repr) ──────────────────────────
    _wall_start: float = field(default=0.0, repr=False)
    _wall_end:   float = field(default=0.0, repr=False)
    _cpu_start:  float = field(default=0.0, repr=False)
    _cpu_end:    float = field(default=0.0, repr=False)

    # ── event counts ──────────────────────────────────────────────────────────
    n_neutrons:           int = 0
    n_real_collisions:    int = 0
    n_virtual_collisions: int = 0
    n_xs_evaluations:     int = 0
    n_majorant_updates:   int = 0
    n_wrong_majorant:      int = 0
    n_wrong_majorant_mean_error: float = 0.0


    # ── wall-clock timers [seconds] ───────────────────────────────────────────
    time_majorant:        float = 0.0
    time_xs_eval:         float = 0.0

    # ── CPU-time timers [seconds] ─────────────────────────────────────────────
    cpu_time_majorant:        float = 0.0
    cpu_time_xs_eval:         float = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        """Record wall and CPU start times."""
        self._wall_start = time.perf_counter()
        self._cpu_start  = time.process_time()

    def stop(self):
        """Record wall and CPU stop times."""
        self._wall_end = time.perf_counter()
        self._cpu_end  = time.process_time()

    # ── derived properties ────────────────────────────────────────────────────
    @property
    def total_time(self) -> float:
        """Total wall-clock time [s]."""
        return self._wall_end - self._wall_start

    @property
    def total_cpu_time(self) -> float:
        """Total CPU time [s]."""
        return self._cpu_end - self._cpu_start

    @property
    def cpu_efficiency(self) -> float:
        """CPU time / wall time.  >1 → multi-threaded;  <1 → I/O bound."""
        return self.total_cpu_time / self.total_time if self.total_time > 0 else 0.0

    @property
    def neutrons_per_second(self) -> float:
        return self.n_neutrons / self.total_time if self.total_time > 0 else 0.0

    @property
    def rejection_fraction(self) -> float:
        total = self.n_virtual_collisions + self.n_real_collisions
        return self.n_virtual_collisions / total if total > 0 else 0.0
    
    @property
    def wrong_majorant_mean_error(self) -> float:
        return self.n_wrong_majorant_mean_error / self.n_wrong_majorant if self.n_wrong_majorant > 0 else 0.0
    
    @property
    def wrong_majorant_fraction(self) -> float:
        return self.n_wrong_majorant / self.n_majorant_updates if self.n_majorant_updates > 0 else 0.0

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PERFORMANCE SUMMARY",
            "=" * 60,
            f"  {'Metric':<30} {'Wall (s)':>10}  {'CPU (s)':>10}",
            "-" * 60,
            f"  {'Total simulation':<30} {self.total_time:>10.4f}  {self.total_cpu_time:>10.4f}",
            f"  {'  Majorant evaluations':<30} {self.time_majorant:>10.4f}  {self.cpu_time_majorant:>10.4f}",
            "-" * 60,
            f"  CPU efficiency (cpu/wall)    : {self.cpu_efficiency:>10.3f}",
            f"  Neutrons / second            : {self.neutrons_per_second:>10.1f}",
            "-" * 60,
            f"  Neutrons simulated           : {self.n_neutrons:>10,}",
            f"  Majorant updates             : {self.n_majorant_updates:>10,}",
            f"  Real collisions              : {self.n_real_collisions:>10,}",
            f"  Virtual (rejected)           : {self.n_virtual_collisions:>10,}",
            f"  Rejection fraction           : {self.rejection_fraction:>10.3%}",
            f"  Wrong majorant fraction      : {self.wrong_majorant_fraction:>10.3%}",
            f"  Wrong majorant mean error      : {self.wrong_majorant_mean_error:>10.3%}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # [NEW] snapshot() — collect scalar performance metrics into a plain dict
    # so run_batch() can store one record per batch without keeping the full
    # PerformanceTracker object alive.
    def snapshot(self) -> dict:
        """Return a dict of scalar performance metrics for batch bookkeeping."""
        return {
            "total_time_s"         : self.total_time,
            "total_cpu_time_s"     : self.total_cpu_time,
            "cpu_efficiency"       : self.cpu_efficiency,
            "neutrons_per_second"  : self.neutrons_per_second,
            "n_neutrons"           : self.n_neutrons,
            "n_real_collisions"    : self.n_real_collisions,
            "n_virtual_collisions" : self.n_virtual_collisions,
            "n_majorant_updates"   : self.n_majorant_updates,
            "rejection_fraction"   : self.rejection_fraction,
            "time_majorant_s"      : self.time_majorant,
            "time_xs_eval_s"       : self.time_xs_eval,
            "n_wrong_majorant"     : self.n_wrong_majorant,
            "wrong_majorant_mean_error" : self.wrong_majorant_mean_error,
            "wrong_majorant_fraction" : self.wrong_majorant_fraction
        }

# ==========================================
# --- MajorantRecord dataclass ---
# ==========================================
@dataclass
class MajorantRecord:
    """
    Stores one snapshot of the majorant cross section at the moment it was
    (re-)computed — either at neutron birth or after a scattering event.

    Attributes
    ----------
    energy              : neutron energy [eV] at which the majorant was evaluated
    value               : majorant Σ_maj [cm⁻¹]
    limiting_material   : name of the material that set the maximum real XS
    actual_max_xs       : the largest real total XS across all materials [cm⁻¹]
    margin              : value / actual_max_xs  (efficiency indicator; 1.0 = tight)
    """
    energy: float
    value: float
    limiting_material: str
    actual_max_xs: float

    @property
    def margin(self) -> float:
        return self.value / self.actual_max_xs if self.actual_max_xs > 0 else float("inf")


# ==========================================
# --- NeutronHistory dataclass ---
# ==========================================
@dataclass
class NeutronHistory:
    """
    Complete per-neutron transport history.

    Every step (real or virtual) appends one entry to each list so that
    list indices are aligned:
        positions[i], energies[i], events[i],
        majorant_xs_at_step[i], local_xs_at_step[i],
        material_at_step[i], distances[i]

    Events
    ------
    "birth"      : initial point (distance[0] = 0)
    "virtual"    : delta-tracking rejection (fictitious collision)
    "scatter"    : real scattering collision
    "absorption" : real absorption collision  → terminal
    "leakage"    : neutron left the geometry → terminal
    """
    neutron_id:    int
    birth_energy:  float
    birth_position: List[float]

    positions:            List[List[float]] = field(default_factory=list)
    energies:             List[float]       = field(default_factory=list)
    events:               List[str]         = field(default_factory=list)
    majorant_xs_at_step:  List[float]       = field(default_factory=list)
    local_xs_at_step:     List[float]       = field(default_factory=list)
    material_at_step:     List[str]         = field(default_factory=list)
    distances:            List[float]       = field(default_factory=list)

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