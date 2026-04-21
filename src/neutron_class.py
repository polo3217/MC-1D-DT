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



# ==========================================
# --- Neutron class ---
# ==========================================
class Neutron:
    def __init__(self, energy=1e6, position=None, direction=None):
        self.energy    = energy
        self.position  = position.copy()  if position  else [0, 0, 0]
        self.direction = direction.copy() if direction else [1, 0, 0]
        self.xs        = None
        self.material  = None

        # [MODIFIED] Optimization 1: Cache for Virtual Collision XS Loop
        # These store the state of the last expensive cross-section evaluation
        self.last_eval_energy = -1.0
        self.last_eval_mat    = None
        self.last_eval_xs     = None

        # [MODIFIED] Optimization 2 & 3: Tally caching variables
        # Stores the current index to avoid O(log N) lookups on every step
        self.current_energy_bin = -1
        # Accumulates track length locally to avoid array overhead during virtual steps
        self.accumulated_distance = 0.0