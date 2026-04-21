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


import majorant_multipole as maj
from src.performance_classes import PerformanceTracker, MajorantRecord, NeutronHistory, MemoryTracker
from src.tally_classes import FluxTallyTLE, VerificationTally
from src.neutron_class import Neutron






# ==========================================
# --- Source class ---
# ==========================================
class Source:
    """
    --- CHANGES FROM ORIGINAL ---
    [NEW] generate_batch(n) method: samples n neutrons on-the-fly and returns
          a lightweight _BatchSource object understood by run_source / run_batch.
          This avoids allocating all neutrons upfront when only a fraction are
          needed for a single batch.
    [NO CHANGE] __init__ and existing neutrons list are identical.
    """

    def __init__(self, neutron_nbr: int,
                 energy_range=(0.1, 1e6),
                 energy_dist: str = "flat",
                 position=None,
                 direction=None):
        self.neutron_nbr  = neutron_nbr
        self.energy_range = list(energy_range)
        self.energy_dist  = energy_dist
        self.position     = position.copy()  if position  else [0, 0, 0]
        self.direction    = direction.copy() if direction else [1, 0, 0]
        self.neutrons: List[Neutron] = []

        for _ in range(neutron_nbr):
            if energy_dist == "flat":
                energy = float(np.random.uniform(*energy_range))
            elif energy_dist == "mono":
                energy = float(energy_range[0])
            else:
                raise ValueError(f"Unknown energy_dist: '{energy_dist}'")

            self.neutrons.append(
                Neutron(energy, self.position.copy(), self.direction.copy())
            )

    # [NEW] generate_batch()
    def generate_batch(self, n: int) -> "_BatchSource":
        """
        Sample n neutrons from the same distribution as __init__ and return
        a lightweight _BatchSource that looks like a source to run_source().

        Parameters
        ----------
        n : number of neutrons for this batch

        Returns
        -------
        _BatchSource with .neutron_nbr == n and .neutrons list of length n
        """


        if n <= 0:
            raise ValueError(f"generate_batch() called with n={n}. Must be > 0.")
            

        batch_neutrons = []
        for _ in range(n):
            if self.energy_dist == "flat":
                energy = float(np.random.uniform(*self.energy_range))
            elif self.energy_dist == "mono":
                energy = float(self.energy_range[0])
            else:
                raise ValueError(f"Unknown energy_dist: '{self.energy_dist}'")
            batch_neutrons.append(
                Neutron(energy, self.position.copy(), self.direction.copy())
            )
        return _BatchSource(n, batch_neutrons)


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