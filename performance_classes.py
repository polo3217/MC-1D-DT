import sys
import os
sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')

from dataclasses import dataclass, field
import time
from datetime import datetime
from pathlib import Path

from typing import List, Optional
import numpy as np
import majorant_multipole as maj
import openmc
import reconr



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