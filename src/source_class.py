import sys
sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')

from dataclasses import dataclass, field
import time
from datetime import datetime
from pathlib import Path

from typing import List, Optional

import pandas as pd
import numpy as np
import openmc
from scipy import stats
import bisect
import random
import math

import majorant_multipole as maj
from src.performance_classes import PerformanceTracker, MajorantRecord, NeutronHistory, MemoryTracker
from src.tally_classes import FluxTallyTLE, VerificationTally
from src.neutron_class import Neutron



@dataclass
class SourceRegion:
    """
    Defines a source term within a spatial region.

    Parameters
    ----------
    region_name    : str    Must match a Region.name in the Geometry.
    weight         : float  Relative sampling weight (default 1.0).
                            Neutrons are distributed across regions
                            proportionally to weight.
    energy_dist    : str    "flat", "mono", or "log_normal"
    energy_range   : tuple  (E_min, E_max) in eV
    direction_dist : str    "isotropic" or "forward"
    direction      : list   Fixed direction for "forward" dist [ux, uy, uz]
    mu, sigma      : float  Parameters for log_normal energy distribution
    """
    region_name    : str
    weight         : float = 1.0
    energy_dist    : str   = "flat"
    energy_range   : tuple = (0.1, 1e6)
    direction_dist : str   = "isotropic"
    direction      : list  = field(default_factory=lambda: [1.0, 0.0, 0.0])
    mu             : Optional[float] = None
    sigma          : Optional[float] = None


# ==========================================
# --- Source class ---
# ==========================================
class Source:
    def __init__(self,
                 neutron_nbr    : int,
                 geometry,
                 source_regions : List[SourceRegion]):

        if not source_regions:
            raise ValueError("At least one SourceRegion must be provided.")

        self.neutron_nbr    = neutron_nbr
        self.geometry       = geometry
        self.source_regions = source_regions

        # Validate that every region name exists in the geometry
        for sr in self.source_regions:
            geometry.get_region(sr.region_name)

        # Normalise weights and precompute cumulative array for O(log N) sampling
        total_weight         = sum(sr.weight for sr in source_regions)
        self._norm_weights   = [sr.weight / total_weight for sr in source_regions]
        self._cumulative_weights = list(np.cumsum(self._norm_weights))

        # No upfront neutron allocation — neutrons are sampled on demand
        # via run_source() or generate_batch()

    # ------------------------------------------------------------------
    @classmethod
    def point(cls, neutron_nbr: int, geometry,
              region_name   : str,
              position      : list,
              energy_dist   : str   = "flat",
              energy_range  : tuple = (0.1, 1e6),
              direction_dist: str   = "isotropic",
              direction     : list  = None,
              mu            : float = None,
              sigma         : float = None) -> "Source":

        sr = SourceRegion(
            region_name    = region_name,
            weight         = 1.0,
            energy_dist    = energy_dist,
            energy_range   = energy_range,
            direction_dist = direction_dist,
            direction      = direction or [1.0, 0.0, 0.0],
            mu             = mu,
            sigma          = sigma,
        )
        src                      = cls.__new__(cls)
        src.neutron_nbr          = neutron_nbr
        src.geometry             = geometry
        src.source_regions       = [sr]
        src._norm_weights        = [1.0]
        src._cumulative_weights  = [1.0]
        src._point_position      = np.array(position, dtype=float)
        return src
    
    @classmethod
    def plane(cls, neutron_nbr: int, geometry, x_coord: float, region_name: str, **kwargs):
        """Factory for a source located at a specific x-plane."""
        src = cls.point(neutron_nbr, geometry, region_name, [x_coord, 0.0, 0.0], **kwargs)
        return src

    # ------------------------------------------------------------------
    def _sample_region(self) -> SourceRegion:
        idx = bisect.bisect_left(self._cumulative_weights, random.random())
        return self.source_regions[min(idx, len(self.source_regions) - 1)]

    # ------------------------------------------------------------------
    def _sample_position(self, sr: SourceRegion) -> np.ndarray:
        if hasattr(self, "_point_position"):
            return self._point_position.copy()
        region = self.geometry.get_region(sr.region_name)
        x = random.uniform(region.x_min, region.x_max)
        return np.array([x, 0.0, 0.0], dtype=float)

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_energy(sr: SourceRegion) -> float:
        if sr.energy_dist == "flat":
            return float(random.uniform(*sr.energy_range))
        elif sr.energy_dist == "mono":
            return float(sr.energy_range[0])
        elif sr.energy_dist == "log_normal":
            if sr.mu is None or sr.sigma is None:
                raise ValueError(
                    f"SourceRegion '{sr.region_name}': mu and sigma required "
                    f"for log_normal energy distribution."
                )
            while True:
                energy = np.random.lognormal(mean=sr.mu, sigma=sr.sigma)
                if sr.energy_range[0] <= energy <= sr.energy_range[1]:
                    return float(energy)
        else:
            raise ValueError(
                f"SourceRegion '{sr.region_name}': unknown energy_dist "
                f"'{sr.energy_dist}'."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_direction(sr: SourceRegion) -> np.ndarray:
        if sr.direction_dist == "isotropic":
            mu  = 2.0 * random.random() - 1.0
            phi = 2.0 * math.pi * random.random()
            sin_theta = math.sqrt(max(0.0, 1.0 - mu**2))
            return np.array([
                sin_theta * math.cos(phi),
                sin_theta * math.sin(phi),
                mu,
            ], dtype=float)
        elif sr.direction_dist == "forward":
            d    = np.array(sr.direction, dtype=float)
            norm = np.linalg.norm(d)
            if norm == 0:
                raise ValueError(
                    f"SourceRegion '{sr.region_name}': direction vector "
                    f"cannot be zero."
                )
            return d / norm
        else:
            raise ValueError(
                f"SourceRegion '{sr.region_name}': unknown direction_dist "
                f"'{sr.direction_dist}'."
            )

    # ------------------------------------------------------------------
    def _sample_neutron(self) -> Neutron:
        sr        = self._sample_region()
        position  = self._sample_position(sr)
        energy    = self._sample_energy(sr)
        direction = self._sample_direction(sr)
        return Neutron(energy, position, direction)

    # ------------------------------------------------------------------
    def generate_batch(self, n: int) -> "_BatchSource":
        if n <= 0:
            raise ValueError(f"generate_batch() called with n={n}. Must be > 0.")
        return _BatchSource(n, [self._sample_neutron() for _ in range(n)])

    # ------------------------------------------------------------------
    def run_source(self, geom) -> None:
        """
        Stream neutrons directly into the geometry one at a time —
        no list allocation at all. Used by Geometry.run_source() when
        called without batching.
        """
        geom.perf.n_neutrons = self.neutron_nbr
        for nid in range(self.neutron_nbr):
            n = self._sample_neutron()
            geom._run_neutron(n, neutron_id=nid)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['geometry'] = None  # strip before pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

# ==========================================
# [NEW] --- _BatchSource helper class ---
# ==========================================
class _BatchSource:
    """
    Minimal source-like object returned by source.generate_batch().

    Has the same interface expected by geometry.run_source():
        .neutron_nbr  : int
        .neutrons     : list[Neutron]

    Not intended to be constructed directly by user code.
    """
    def __init__(self, neutron_nbr: int, neutrons: List[Neutron]):
        self.neutron_nbr = neutron_nbr
        self.neutrons    = neutrons