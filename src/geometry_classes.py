"""
neutron_transport.py
====================
Simplified Monte Carlo Delta-Tracking neutron transport code.
Includes physics tallies, per-neutron history storage, majorant diagnostics,
and computational performance tracking (wall time + CPU time).

CPU time  : time.process_time()  — counts only time the process was on-CPU
            (excludes I/O waits, sleep, and time spent in other processes).
Wall time : time.perf_counter()  — real elapsed ("clock on the wall") time.

Both are tracked for every timed region so the dashboard can show
parallelism efficiency = CPU time / wall time.

"""
import sys
import os
import math
import random
import bisect
from multiprocessing import Pool
import functools

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
from src.performance_classes import PerformanceTracker, MajorantRecord, NeutronHistory
from src.tally_classes import FluxTallyTLE, VerificationTally
import src.group_xs as group_xs
import src.vectfit as vf
import src.serpent_evaluation as srp
import src.reconr_v2 as reconr

global  valid_nuclides_name 
global  valid_nuclides_list 
global  xs_dir

valid_nuclides_name = [ "U238", "U235", "Pu239","Pu240", "Cnat", "O16", "H1", "Fe56", "Xe135", "Na23" ]    
valid_nuclides_list = [ '092238', '092235', '094239', '094240', '006000',  '008016', '001001','026056','054135', '011023']
xs_dir = '/home/paule/open_mc_projects/xs_lib/endfb-vii.1-hdf5/wmp'



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


# ==========================================
# --- Material class ---
# ==========================================
class Material:
    def __init__(self, name, nuclides=None, T=293.6):
        self.name     = name

        #nuclides are defined by name and than the multipole library is loaded
        self.nuclides = nuclides if nuclides is not None else []
        for i, pair in enumerate(self.nuclides):
            print(pair)
            nuclide_name = pair[0]
            nuclide_density = pair[1]
            
                
            if nuclide_name not in valid_nuclides_name:
                raise ValueError(f"Nuclide '{nuclide_name}' not in valid nuclides list.")
            
            index = valid_nuclides_name.index(nuclide_name)
            nuclide_id = valid_nuclides_list[index]                         
            file_h5 = str(nuclide_id) + '.h5'
            
            nuclide_obj = openmc.data.WindowedMultipole.from_hdf5(os.path.join(xs_dir, file_h5))
            self.nuclides[i] = (nuclide_obj, nuclide_density*1.0e-24)
    
        self.T        = T
        self.total_density = float(np.sum([n[1] for n in self.nuclides]))
        self.xs       = None

    def _xs_evaluation(self, energy: float) -> np.ndarray:
        """Return [scatter, absorb, fission] in cm⁻¹."""
        xs_total = np.zeros(3)
        for nuclide_obj, density in self.nuclides:
            xs = np.array(nuclide_obj(energy, self.T))
            xs_total += density * xs
        self.xs = xs_total
        return xs_total


# ==========================================
# --- Geometry class ---
# ==========================================
class Geometry:
    """
    Infinite-slab geometry with delta-tracking transport.

    Attributes
    ----------
    perf          : PerformanceTracker
    histories     : list[NeutronHistory]   populated after run_source()
    majorant_log  : list[MajorantRecord]   one record per majorant evaluation
    flux_tally    : FluxTallyTLE | None    set by attach_flux_tally()
    verbose       : bool                   print per-step diagnostics

    --- CHANGES FROM ORIGINAL ---
    [NEW] batch_results : list — populated by run_batch(); each entry is a dict
          with keys "flux", "verif", "perf" containing snapshots from that batch.
    """

    def __init__(self,
                 flux_tally=True,
                 verification_tally=True,
                 perf_tracker=True,
                 majorant_log=True,
                 history_flag=False, 
                 verbose: bool = False):
        
        self._mode                = "analysis"  # "analysis" or "validation"
        self._maj_xs_method       = "serpent"      # "vectfit", "sqrtE_T" or "serpent" or "point"
        self._maj_mat_method      = "simple"     # "simple" or "maj_mat"
        self._access_method        = "fly"        # "fly" or "reconr"

        #specific for serpent majorant_xs_method
        self.T_array        = None

        self._materials      = []
        self._nuclides       = {}
        self.boundaries      = [0]
        self.material_array  = []
        self.source          = None
        self.verbose         = verbose
        
        # specific for the maj_mat_method 
        self.maj_mat  : Material       = None
        self.xs_maj_tables   = {}

        self.reconr_e_grid          = None
        self.reconr_maj_xs_grid     = None
        self.reconr_sqrt_E_min      = None
        self.reconr_e_spacing       = None
        self.reconr_window_pointers = None

        self._df_group_xs    = None

        # Legacy score accumulators
        self.absorption_score        = 0
        self.scattering_score        = 0
        self.leakage_score           = 0
        self.acception_score         = 0
        self.rejection_score         = 0
        self.distance_score          = 0
        self.distance_sampling_score = 0

        # Flags to control which features are active
        self.flux_tally_flag         = flux_tally
        self.verif_tally_flag        = verification_tally
        self.perf_tracker_flag       = perf_tracker
        self.majorant_log_flag       = majorant_log
        self.history_flag            = history_flag  # always store histories; can be filtered later based on mode or other criteria

        # Initialise objects (always created; flags control whether they are used)
        self.perf:         PerformanceTracker          = PerformanceTracker()
        self.histories:    List[NeutronHistory]        = []
        self._majorant_log: List[MajorantRecord]       = []
        self.flux_tally:   Optional[FluxTallyTLE]      = None
        self.verif_tally:  Optional[VerificationTally] = None
        
        self.batch_results: list = []

    def reset(self):
        """Reset all tallies, histories, and performance metrics."""
        self.histories.clear()
        self._majorant_log.clear()
        if self.flux_tally:
            self.flux_tally._flux_tally.reset()
        if self.verif_tally:
            self.verif_tally._absorption.reset()
            self.verif_tally._scatter.reset()
            self.verif_tally._current_fwd.reset()
            self.verif_tally._current_bwd.reset()
            self.verif_tally._leak_left.reset()
            self.verif_tally._leak_right.reset()

        self.perf = PerformanceTracker()
        self.absorption_score = 0
        self.scattering_score = 0
        self.leakage_score = 0
        self.acception_score = 0
        self.rejection_score = 0
        self.distance_score = 0
        self.distance_sampling_score = 0

    @property
    def mode(self):
        return self._mode
    
    def set_mode(self, value: str, filename: str = None):
        if value not in ["analysis", "validation"]:
            raise ValueError("Mode must be 'analysis' or 'validation'")
        if value == "validation":
            if filename is None:
                raise ValueError("filename must be provided when setting validation mode")
            self._mode = value
            self.set_maj_xs_method("group")
            self.maj_method = "simple"
            self.df_group_xs = group_xs.get_group_xs(filename, verbose=self.verbose)
        else:
            self._mode = value

    @property 
    def maj_xs_method(self):
        return self._maj_xs_method
    
    def set_maj_xs_method(self, method: str = "serpent", verbose: bool = False,
                          T_array: Optional[np.ndarray] = None,
                          xs_maj_file_dir : Optional[str] = None):
        if method not in ["serpent", "vectfit", "sqrtE_T", "group"]:
            raise ValueError("maj_xs_method must be 'serpent', 'vectfit', 'sqrtE_T' or 'group'")
        self._maj_xs_method = method

        if method == "vectfit":
            if xs_maj_file_dir is None:
                raise ValueError("file_dir must be provided when setting maj_xs_method to 'vectfit'")
            self.xs_maj_tables = vf.xs_majorant_tables(file_dir=xs_maj_file_dir, verbose=verbose)

        if method == "serpent":
            if T_array is None:
                raise ValueError("T_array must be provided when setting maj_xs_method to 'serpent'")
            self.T_array = T_array

        if method == "sqrtE_T":
            raise NotImplementedError("sqrtE_T majorant method not implemented yet")
        
    @property
    def maj_mat_method(self):
        return self._maj_mat_method

    @maj_mat_method.setter
    def maj_mat_method(self, value):
        self._maj_mat_method = value
        if value == "maj_mat":
            self.maj_mat    = self.build_majorant_all_material(verbose_maj=True)
        
    @property
    def access_method(self):
        return self._access_method
    
    def set_access_method(self, value, err_lim=0.001, err_max=0.01, err_int=None,
                        last_window=None, last_energy=None):
        if value not in ["fly", "reconr"]:
            raise ValueError("access_method must be 'fly' or 'reconr'")
        self._access_method = value

        if value == "reconr":
            print("Building majorant XS grid for RECONR access method with settings:")
            print(f"xs_method:      {self.maj_xs_method}")
            print(f"maj_mat_method: {self.maj_mat_method}")
            print(f"materials:      {[mat.name for mat in self.materials]}")

            self.reconr_e_grid, self.reconr_maj_xs_grid, self.reconr_sqrt_E_min, self.reconr_e_spacing, self.reconr_window_pointers = reconr.build_majorant_xs_grid(self, err_lim, err_max, err_int, last_window, last_energy)
            

    @property
    def majorant_log(self) -> List[MajorantRecord]:
        return self._majorant_log

    # ------------------------------------------------------------------

    @property
    def df_group_xs(self):
        return self._df_group_xs

    @df_group_xs.setter
    def df_group_xs(self, df):
        self._df_group_xs = df

    @property
    def materials(self):
        return self._materials

    def add_material(self, mat: Material):
        if any(m.name == mat.name for m in self._materials):
            raise ValueError(f"Material with name '{mat.name}' already exists.")
        self._materials.append(mat)
        for nuclide in mat.nuclides:
            print(nuclide[0].name)
            if nuclide[0].name not in self._nuclides:
                self._nuclides[nuclide[0].name] = nuclide
            
    @property
    def nuclides(self):
        return self._nuclides

    # ------------------------------------------------------------------
    #  Majorant helper methods

    def build_majorant_all_material(self, verbose_maj: bool = False) -> Material:
        # collect max density per nuclide across all materials
        nuclides = {}   # {name: (nuclide_obj, max_density)}
        for mat in self.materials:
            for nuclide_obj, density in mat.nuclides:
                if verbose_maj:
                    print(f"Material: {mat.name:12s} | "
                        f"Nuclide: {nuclide_obj.name:10s} | "
                        f"Density: {density*1e24:.4e} atoms/cm³")
                if nuclide_obj.name not in nuclides:
                    nuclides[nuclide_obj.name] = (nuclide_obj, density)
                else:
                    _, existing_density = nuclides[nuclide_obj.name]
                    nuclides[nuclide_obj.name] = (nuclide_obj,
                                                max(existing_density, density))

        if verbose_maj:
            print("Majorant material composition:")
            for name, (_, density) in nuclides.items():
                print(f"  {name:10s} : {density*1e24:.4e} atoms/cm³")

        # build the Material object bypassing __init__ loading logic
        mat_maj       = object.__new__(Material)
        mat_maj.name  = "maj_mat"
        mat_maj.xs    = None
        mat_maj.nuclides = [(nuclide_obj, density)
                            for nuclide_obj, density in nuclides.values()]
        mat_maj.total_density = float(sum(d for _, d in mat_maj.nuclides))

        return mat_maj

    # ------------------------------------------------------------------
    def attach_flux_tally(self, energy_bins: List[float],
                          transverse_area: float = 1.0):
        self.flux_tally = FluxTallyTLE(self.boundaries, energy_bins,
                                       transverse_area)

    # ------------------------------------------------------------------
    def attach_verification_tally(self, energy_bins: List[float],
                                  surface_xs: List[float]):
        self.verif_tally = VerificationTally(
            self.boundaries, energy_bins, surface_xs
        )

    # ------------------------------------------------------------------
    def calculate_nuclide_majorant_xs(self, energy: float, nuclide, temperature= None)-> float:
        if self.maj_xs_method == "vectfit":
            if self.xs_maj_tables is None:
                raise ValueError("Majorant tables not loaded. Set maj_xs_method to 'vectfit' and ensure tables are loaded.")
            if self.xs_maj_tables[nuclide.name] is None:
                raise ValueError(f"Majorant table for nuclide {nuclide.name} not found. Ensure tables are loaded and contain the nuclide.")
            
            maj_table = self.xs_maj_tables[nuclide.name]
            return maj.evaluate_sig_maj(nuclide, energy, maj_table)
        
        elif self.maj_xs_method == "sqrtE_T":
            raise NotImplementedError("sqrtE_T majorant method not implemented yet")
        
        elif self.maj_xs_method == "serpent":
            if self.maj_mat_method == "maj_mat":
                raise ValueError ("Serpent majorant method is not compatile with the maj_mat method -- There is no way to know the temperature at wich to evaluate the majorant.")
            if self.T_array is None:
                raise ValueError("T_array must be provided for serpent majorant method")
            return srp.serpent_majorant(energy, T_eval=temperature, T_array=self.T_array, nuclide=nuclide)

        return
    
    def caculate_mat_majorant_xs(self, energy: float) -> float:
        ## Case maj_mat = simply caculate a majorant material and evaluate it.
        if self.maj_mat_method == "maj_mat":
            assert self.maj_mat.name == "maj_mat", "maj_mat method requires a majorant material to be defined and set as maj_mat"
            majorant_xs = 0.0
            
            for nuclide_obj, density in self.maj_mat.nuclides:
                maj_xs_nuclide = self.calculate_nuclide_majorant_xs(energy, nuclide_obj)
                majorant_xs += density * maj_xs_nuclide

        ## Case simple : iterate through each material and select the maximum majorant xs
        elif self.maj_mat_method == "simple":
            majorant_xs = 0.0
            for mat in self.materials:
                mat_majorant_xs = 0.0
                for nuclide_obj, density in mat.nuclides:
                    maj_xs_nuclide = self.calculate_nuclide_majorant_xs(energy, nuclide_obj, mat.T)
                    mat_majorant_xs += density * maj_xs_nuclide
                if mat_majorant_xs > majorant_xs:
                    majorant_xs = mat_majorant_xs
        
        return majorant_xs
    
    def access_majorant_xs(self, energy: float) -> float:
        if self.access_method == "fly":
            return self.caculate_mat_majorant_xs(energy)
        
        elif self.access_method == "reconr":
            if energy <= self.reconr_e_grid[0]:
                return self.reconr_maj_xs_grid[0]
            if energy >= self.reconr_e_grid[-1]:
                return self.reconr_maj_xs_grid[-1]

            # O(1) window identification
            w = int((math.sqrt(energy) - self.reconr_sqrt_E_min) / self.reconr_e_spacing)
            

            lo = self.reconr_window_pointers[w]
            hi = self.reconr_window_pointers[w + 1]
            
            # Bisect within window slice — no list copy
            i = bisect.bisect_right(self.reconr_e_grid, energy, lo, hi)
        
            

            E1  = self.reconr_e_grid[i - 1]
            E2  = self.reconr_e_grid[i]
            xs1 = self.reconr_maj_xs_grid[i - 1]
            xs2 = self.reconr_maj_xs_grid[i]

            
            xs = xs1 + (xs2 - xs1) * (energy - E1) / (E2 - E1)
            
            return xs

    def get_majorant_xs(self, energy: float) -> float:

        if self.perf_tracker_flag:
            wall_t0 = time.perf_counter()
            cpu_t0  = time.process_time()

        if self.mode == "validation":
            actual_max_xs     = 0.0
            limiting_mat_name = ""
            if self.df_group_xs is None:
                raise ValueError("...")
            for cell in self.df_group_xs['cell'].unique():
                cell_df = self.df_group_xs[self.df_group_xs['cell'] == cell]
                total = cell_df['scatter'].sum() + cell_df['absorption'].sum()
                if total > actual_max_xs:
                    actual_max_xs     = total
                    limiting_mat_name = f"Cell {cell}"
            majorant_xs = 2.0 * actual_max_xs

        else:
            majorant_xs = self.access_majorant_xs(energy)
            if self.majorant_log_flag:
                actual_max_xs     = 0.0
                limiting_mat_name = ""
                for mat in self.materials:
                    xs_arr = mat._xs_evaluation(energy)
                    total  = float(xs_arr[0] + xs_arr[1])
                    if total > actual_max_xs:
                        actual_max_xs     = total
                        limiting_mat_name = mat.name

        # shared bookkeeping for both modes
        if self.majorant_log_flag:
            self._majorant_log.append(MajorantRecord(
                energy=energy,
                value=majorant_xs,
                limiting_material=limiting_mat_name,
                actual_max_xs=actual_max_xs,
            ))

        if self.perf_tracker_flag:
            wall_elapsed = time.perf_counter() - wall_t0
            cpu_elapsed  = time.process_time()  - cpu_t0
            self.perf.n_majorant_updates  += 1
            self.perf.time_majorant       += wall_elapsed
            self.perf.cpu_time_majorant   += cpu_elapsed
        
        return majorant_xs

    # ------------------------------------------------------------------
    def _sample_neutron_distance(self, majorant_xs: float) -> float:
        if majorant_xs is None or majorant_xs <= 0:
            print(f"Error: Invalid majorant_xs value: {majorant_xs}")
            raise ValueError("Majorant cross section must be positive")
        # [MODIFIED] Scalar Optimization: Avoid numpy scalar dispatch overhead
        return -math.log(random.random()) / majorant_xs

    # ------------------------------------------------------------------
    def _move_neutron(self, n: Neutron, distance: float):
        # [MODIFIED] Scalar Optimization: Unrolled loop to avoid Python 'for' overhead
        n.position[0] += n.direction[0] * distance
        n.position[1] += n.direction[1] * distance
        n.position[2] += n.direction[2] * distance

    # ------------------------------------------------------------------
    def _get_material_at(self, x: float) -> Optional[Material]:
        # [MODIFIED] Scalar Optimization: Use compiled bisect instead of pure Python loops
        idx = bisect.bisect_right(self.boundaries, x) - 1
        if 0 <= idx < len(self.material_array):
            return self.material_array[idx]
        return None

    # ------------------------------------------------------------------
    def _evaluate_acceptance(self, n: Neutron, majorant_xs: float) -> bool:
        if majorant_xs is None:
            raise ValueError("Majorant cross section is not defined")

        mat = self._get_material_at(n.position[0])

        if mat is None:
            return False
        
        if self.mode == "analysis":

            wall_t0 = time.perf_counter()
            cpu_t0  = time.process_time()

            # [MODIFIED] Optimization 1: Virtual Collision Caching Loop
            # Instead of re-evaluating the WMP Faddeeva function on every step,
            # we check if the neutron's energy and material are unchanged since the last evaluation.
            if n.energy == n.last_eval_energy and mat == n.last_eval_mat:
                n.xs = n.last_eval_xs
            else:
                n.xs = mat._xs_evaluation(n.energy)
                # Cache the results for the next virtual collision
                n.last_eval_energy = n.energy
                n.last_eval_mat = mat
                n.last_eval_xs = n.xs

            wall_elapsed = time.perf_counter() - wall_t0
            cpu_elapsed  = time.process_time()  - cpu_t0

            self.perf.time_xs_eval     += wall_elapsed
            self.perf.cpu_time_xs_eval += cpu_elapsed
            self.perf.n_xs_evaluations += 1

            local_xs        = float(n.xs[0] + n.xs[1])
            acceptance_prob = local_xs / majorant_xs

            if self.verbose:
                print(
                    f"  Material: {mat.name:12s} | "
                    f"local_xs: {local_xs:.4f} | "
                    f"majorant: {majorant_xs:.4f} | "
                    f"ratio: {acceptance_prob:.4f}"
                )

            # [MODIFIED] Scalar Optimization: Replace np.random.uniform() with random.random()
            if random.random() < acceptance_prob:
                n.material = mat
                return True
            return False
        
        elif self.mode == "validation":
            wall_t0 = time.perf_counter()
            cpu_t0  = time.process_time()
            if self.df_group_xs is None:
                raise ValueError("Group-wise XS data frame not set. Set df_group_xs before using 'group' method.")
            if mat.name == "cell 1":
                mat_name = 15
            elif mat.name == "cell 2":
                mat_name = 16
            else:
                raise ValueError(f"Unexpected material name: {mat.name}. Expected 'cell 1' or 'cell 2' for group-wise evaluation.")

            cell_df = self.df_group_xs[self.df_group_xs['cell'] == mat_name]
            scatter_xs = cell_df['scatter'].sum()
            absorption_xs = cell_df['absorption'].sum()
            fission_xs = cell_df['fission'].sum() if 'fission' in cell_df.columns else 0.0
            n.xs = np.array([scatter_xs, absorption_xs, fission_xs])

            wall_elapsed = time.perf_counter() - wall_t0
            cpu_elapsed  = time.process_time()  - cpu_t0

            self.perf.time_xs_eval     += wall_elapsed
            self.perf.cpu_time_xs_eval += cpu_elapsed
            self.perf.n_xs_evaluations += 1

            local_xs        = float(n.xs[0] + n.xs[1])
            acceptance_prob = local_xs / majorant_xs

            if self.verbose:
                print(
                    f"  Material: {mat.name:12s} | "
                    f"local_xs: {local_xs:.4f} | "
                    f"majorant: {majorant_xs:.4f} | "
                    f"ratio: {acceptance_prob:.4f}"
                )

            # [MODIFIED] Scalar Optimization: Replace np.random.uniform() with random.random()
            if random.random() < acceptance_prob:
                n.material = mat
                return True
            return False

    # ------------------------------------------------------------------
    def _sample_collision(self, n: Neutron) -> str:
        if n.energy < 10.0:
            return "absorption"
        scatter_prob = n.xs[0] / (n.xs[0] + n.xs[1])
        # [MODIFIED] Scalar Optimization: Replace np.random.uniform() with random.random()
        return "scatter" if random.random() < scatter_prob else "absorption"

    # ------------------------------------------------------------------
    def _scattering_neutron(self, n: Neutron):
        if self.mode == "analysis":
            # [MODIFIED] Scalar Optimization: Use pure Python list math instead of np.array and np.random.choice
            contributions = [density * nuclide_obj(n.energy, n.material.T)[0]
                             for nuclide_obj, density in n.material.nuclides]
            total_prob = sum(contributions)
            rand_val = random.random() * total_prob
            
            cum_sum = 0.0
            idx = 0
            for i, prob in enumerate(contributions):
                cum_sum += prob
                if rand_val <= cum_sum:
                    idx = i
                    break

            A = n.material.nuclides[idx][0].sqrtAWR ** 2

        elif self.mode == "validation":
            if len(n.material.nuclides) != 1:
                raise ValueError("Group-wise scattering/validation mode only implemented for materials with a single nuclide.")
            nuclide_obj = n.material.nuclides[0][0]
            A = nuclide_obj.sqrtAWR ** 2

        # [MODIFIED] Scalar Optimization: Replace np.random and np.pi with random/math modules
        mu_cm = 2.0 * random.random() - 1.0
        phi   = 2.0 * math.pi * random.random()

        new_energy = n.energy * (A**2 + 2*A*mu_cm + 1) / (A + 1)**2
        n.energy   = max(new_energy, 1e-10)

        # [MODIFIED] Scalar Optimization: Replace np.sqrt, np.cos, np.sin with native math functions
        denom  = math.sqrt(A**2 + 2*A*mu_cm + 1)
        mu_lab = (A * mu_cm + 1.0) / denom if denom != 0 else 0.0

        u, v, w   = n.direction
        sin_theta = math.sqrt(max(0.0, 1.0 - mu_lab**2))
        cos_phi   = math.cos(phi)
        sin_phi   = math.sin(phi)

        if abs(w) < 0.999999:
            temp  = math.sqrt(1.0 - w**2)
            new_u = u*mu_lab + sin_theta*(u*w*cos_phi - v*sin_phi)/temp
            new_v = v*mu_lab + sin_theta*(v*w*cos_phi + u*sin_phi)/temp
            new_w = w*mu_lab - sin_theta*temp*cos_phi
        else:
            new_u = sin_theta * cos_phi
            new_v = sin_theta * sin_phi
            # [MODIFIED] Scalar Optimization: Inline sign check instead of np.sign
            new_w = (1.0 if w >= 0 else -1.0) * mu_lab

        norm           = math.sqrt(new_u**2 + new_v**2 + new_w**2)
        n.direction[0] = new_u / norm
        n.direction[1] = new_v / norm
        n.direction[2] = new_w / norm

    # ------------------------------------------------------------------
    def _run_neutron(self, n: Neutron, neutron_id: int = 0,
                    track_neutron: bool = False):
        
        if self.history_flag:

            hist = NeutronHistory(
                neutron_id     = neutron_id,
                birth_energy   = n.energy,
                birth_position = n.position.copy(),
            )
            hist.positions.append(n.position.copy())
            hist.energies.append(n.energy)
            hist.events.append("birth")
            hist.majorant_xs_at_step.append(0.0)
            hist.local_xs_at_step.append(0.0)
            hist.material_at_step.append("")
            hist.distances.append(0.0)

        if track_neutron:
            pos_out = [n.position.copy()]
            dir_out = [n.direction.copy()]
            eng_out = [n.energy]
            evt_out = ["birth"]

        # [MODIFIED] Optimization 2: Pre-calculate energy bin at birth to avoid O(log N) lookups on every step
        if self.flux_tally_flag and self.flux_tally is not None:
            n.current_energy_bin = self.flux_tally._energy_bin(n.energy)
            n.accumulated_distance = 0.0

        majorant_xs = self.get_majorant_xs(n.energy)

        while True:
            distance = self._sample_neutron_distance(majorant_xs)
            x_start  = n.position[0]
       
            # move neutron
            self._move_neutron(n, distance)
            
            self.distance_sampling_score += 1

            # ---------- Leakage — checked BEFORE any scoring ----------
            if (n.position[0] < self.boundaries[0] or
                    n.position[0] >= self.boundaries[-1]):

                if n.position[0] < self.boundaries[0]:
                    x_boundary = self.boundaries[0]
                else:
                    x_boundary = self.boundaries[-1]

                if abs(n.direction[0]) > 0.0:
                    distance_inside = (x_boundary - x_start) / n.direction[0]
                else:
                    distance_inside = 0.0
                distance_inside = max(0.0, min(distance_inside, distance))

                self.distance_score          += distance_inside

                # [FIX] score surface crossing up to the boundary only, not to
                # the out-of-bounds position — prevents spurious current scores
                if self.verif_tally_flag and self.verif_tally is not None:
                    self.verif_tally.score_surface_crossing(x_start, x_boundary)
                    self.verif_tally.score_leakage(n.position[0])

                # [MODIFIED] Optimization 3: Accumulate track length and flush on leakage
                if self.flux_tally_flag and self.flux_tally is not None:
                    n.accumulated_distance += distance_inside
                    if n.current_energy_bin >= 0:
                        # Direct scoring via _flux_tally array bypasses the bin binary search
                        self.flux_tally._flux_tally.score(n.current_energy_bin, n.accumulated_distance)
                    n.accumulated_distance = 0.0

                self.leakage_score += 1
                if self.history_flag:
                    hist.fate = "leakage"
                    hist.positions.append(n.position.copy())
                    hist.energies.append(n.energy)
                    hist.events.append("leakage")
                    hist.majorant_xs_at_step.append(majorant_xs)
                    hist.local_xs_at_step.append(0.0)
                    hist.material_at_step.append("outside")
                    hist.distances.append(distance_inside)

                if track_neutron:
                    pos_out.append(n.position.copy())
                    dir_out.append(n.direction.copy())
                    eng_out.append(n.energy)
                    evt_out.append("leakage")
                break

            # ---------- Neutron stayed inside ----------
            # [FIX] surface crossing scored here with actual in-bounds x_end
            self.distance_score          += distance
            
            if self.verif_tally_flag and self.verif_tally is not None:
                self.verif_tally.score_surface_crossing(x_start, n.position[0])

            # [MODIFIED] Optimization 3: Accumulate track length locally instead of scoring immediately
            if self.flux_tally_flag and self.flux_tally is not None:
                n.accumulated_distance += distance

            # ---------- Acceptance check ----------
            if self._evaluate_acceptance(n, majorant_xs):
                self.acception_score        += 1
                self.perf.n_real_collisions += 1

                local_xs  = float(n.xs[0] + n.xs[1])
                mat_name  = n.material.name if n.material else ""
                coll_type = self._sample_collision(n)

                if coll_type == "absorption":
                    self.absorption_score += 1
                    if self.history_flag:
                        hist.fate = "absorption"
                        

                    if self.verif_tally_flag and self.verif_tally is not None:
                        self.verif_tally.score_collision(
                            n.position[0], n.energy, 'absorption')
                        
                    # [MODIFIED] Optimization 3: Flush accumulated track length on absorption
                    if self.flux_tally_flag and self.flux_tally is not None and n.current_energy_bin >= 0:
                        self.flux_tally._flux_tally.score(n.current_energy_bin, n.accumulated_distance)
                        n.accumulated_distance = 0.0
                    
                    if self.history_flag:
                        hist.positions.append(n.position.copy())
                        hist.energies.append(n.energy)
                        hist.events.append("absorption")
                        hist.majorant_xs_at_step.append(majorant_xs)
                        hist.local_xs_at_step.append(local_xs)
                        hist.material_at_step.append(mat_name)
                        hist.distances.append(distance)

                    if track_neutron:
                        pos_out.append(n.position.copy())
                        dir_out.append(n.direction.copy())
                        eng_out.append(n.energy)
                        evt_out.append("absorption")
                    break

                else:  # scatter
                    self.scattering_score += 1
                    if self.history_flag:
                        hist.n_scatters       += 1

                    if self.verif_tally_flag and self.verif_tally is not None:
                        self.verif_tally.score_collision(
                            n.position[0], n.energy, 'scatter')

                    if self.history_flag:
                        hist.positions.append(n.position.copy())
                        hist.energies.append(n.energy)
                        hist.events.append("scatter")
                        hist.majorant_xs_at_step.append(majorant_xs)
                        hist.local_xs_at_step.append(local_xs)
                        hist.material_at_step.append(mat_name)
                        hist.distances.append(distance)

                    if track_neutron:
                        pos_out.append(n.position.copy())
                        dir_out.append(n.direction.copy())
                        eng_out.append(n.energy)
                        evt_out.append("scatter")

                    # [MODIFIED] Optimization 3: Flush track length BEFORE scatter changes energy
                    if self.flux_tally_flag and self.flux_tally is not None and n.current_energy_bin >= 0:
                        self.flux_tally._flux_tally.score(n.current_energy_bin, n.accumulated_distance)
                        n.accumulated_distance = 0.0

                    self._scattering_neutron(n)

                    # [MODIFIED] Optimization 2: Update cached energy bin AFTER scatter
                    if self.flux_tally_flag and self.flux_tally is not None:
                        n.current_energy_bin = self.flux_tally._energy_bin(n.energy)

                    majorant_xs = self.get_majorant_xs(n.energy)

            else:
                # ---------- Virtual collision ----------
                self.rejection_score           += 1
                self.perf.n_virtual_collisions += 1
                if self.history_flag:
                    hist.n_virtual                 += 1

                mat_at_pos = self._get_material_at(n.position[0])
                local_xs   = 0.0
                mat_name   = ""
                if mat_at_pos is not None:
                    if self.mode == "analysis":
                        xs_arr   = mat_at_pos._xs_evaluation(n.energy)
                    elif self.mode == "validation":
                        if mat_at_pos.name == "cell 1":
                            xs_scatter    = self.df_group_xs[(self.df_group_xs['cell'] == 15)]['scatter'].sum()
                            xs_absorption = self.df_group_xs[(self.df_group_xs['cell'] == 15)]['absorption'].sum()
                            xs_fission    = self.df_group_xs[(self.df_group_xs['cell'] == 15)]['fission'].sum() if 'fission' in self.df_group_xs.columns else 0.0
                            xs_arr = np.array([xs_scatter, xs_absorption, xs_fission])
                        elif mat_at_pos.name == "cell 2":
                            xs_scatter    = self.df_group_xs[(self.df_group_xs['cell'] == 16)]['scatter'].sum()
                            xs_absorption = self.df_group_xs[(self.df_group_xs['cell'] == 16)]['absorption'].sum()
                            xs_fission    = self.df_group_xs[(self.df_group_xs['cell'] == 16)]['fission'].sum() if 'fission' in self.df_group_xs.columns else 0.0
                            xs_arr = np.array([xs_scatter, xs_absorption, xs_fission])

                    local_xs = float(xs_arr[0] + xs_arr[1])
                    mat_name = mat_at_pos.name

                if self.history_flag:
                    hist.positions.append(n.position.copy())
                    hist.energies.append(n.energy)
                    hist.events.append("virtual")
                    hist.majorant_xs_at_step.append(majorant_xs)
                    hist.local_xs_at_step.append(local_xs)
                    hist.material_at_step.append(mat_name)
                    hist.distances.append(distance)

                if track_neutron:
                    pos_out.append(n.position.copy())
                    dir_out.append(n.direction.copy())
                    eng_out.append(n.energy)
                    evt_out.append("virtual")

        if self.history_flag:
            self.histories.append(hist)

        if self.flux_tally_flag and self.flux_tally is not None:
            self.flux_tally.end_history()
        if self.verif_tally_flag and self.verif_tally is not None:
            self.verif_tally.end_history()

        if track_neutron:
            return pos_out, dir_out, eng_out, evt_out
        return None, None, None, None

    # ------------------------------------------------------------------
    def run_source(self, src, track_neutron: bool = False):
        self.source          = src
        self.perf.n_neutrons = src.neutron_nbr

        print(f"Running source with following setttings:")
        print(f"  Mode: {self.mode}")
        print(f"  Majorant XS method: {self.maj_xs_method}")
        print(f"  Material majorant method: {self.maj_mat_method}")
        print(f"  Access method: {self.access_method}")
        self.perf.start()

        tracks = []
        for nid, n in enumerate(src.neutrons):
            active_neutron = Neutron(energy=n.energy, position=n.position, direction=n.direction)
            result = self._run_neutron(active_neutron, neutron_id=nid,
                                       track_neutron=track_neutron)
            if track_neutron:
                tracks.append(result)

        self.perf.stop()
        return tracks if track_neutron else None

    # ------------------------------------------------------------------
    # [NEW] run_batch()
    # ─────────────────────────────────────────────────────────────────
    def run_batch(self, src, n_batches: int, track_neutron: bool = False):

        if n_batches < 2:
            raise ValueError("n_batches must be >= 2 to compute cross-batch statistics.")

        total_neutrons     = src.neutron_nbr          # capture before any reset()
        neutrons_per_batch = total_neutrons

        self.batch_results = []

        for batch_idx in range(n_batches):

            batch_n = neutrons_per_batch 

            if batch_n <= 0:
                raise ValueError(
                    f"Batch {batch_idx} has {batch_n} neutrons. "
                    f"Check that N_NEUTRONS ({total_neutrons}) >= N_BATCHES ({n_batches})."
                )

            batch_src = src.generate_batch(batch_n)

            self.reset()
            self.run_source(batch_src, track_neutron=track_neutron)

            snap = {
                "batch"     : batch_idx,
                "n_neutrons": batch_n,
                "perf"      : self.perf.snapshot(),
            }
            if self.flux_tally_flag and self.flux_tally is not None:
                snap["flux"] = {
                    "energy_bins" : self.flux_tally.energy_bins.tolist(),
                    "flux": {
                        "mean"          : self.flux_tally.mean.tolist(),
                        "std"           : self.flux_tally.std.tolist(),
                        "relative_error": self.flux_tally.relative_error.tolist(),
                    }
                }
            if self.verif_tally_flag and self.verif_tally is not None:
                snap["verif"] = self.verif_tally.snapshot()

            self.batch_results.append(snap)
            print(f"  Batch {batch_idx+1:>3}/{n_batches}  "
                f"({batch_n} neutrons)  "
                f"wall={self.perf.total_time:.3f}s")

        self.batch_stats = self._compute_batch_stats()
        return self.batch_stats

    # ------------------------------------------------------------------
    # [NEW] _compute_batch_stats()
    # ─────────────────────────────────────────────────────────────────
    def _compute_batch_stats(self) -> dict:
        """
        Aggregate per-batch snapshots into cross-batch mean ± std.

        For every tally key present in batch_results the method stacks the
        per-batch mean arrays and computes:
            cross_mean = mean over batches
            cross_std  = std  over batches / sqrt(n_batches)   (std of the mean)

        Returns a dict with keys "flux", "verif", "perf" (whichever are present).
        Each sub-dict has "mean" and "std" (and "relative_error" for array tallies).
        """
        B = len(self.batch_results)
        stats = {"n_batches": B}

        # ── flux tally ────────────────────────────────────────────────────
        if "flux" in self.batch_results[0]:
            # shape: (B, n_energy)
            flux_means = np.array([r["flux"]["flux"]["mean"]
                                   for r in self.batch_results])
            cross_mean = flux_means.mean(axis=0)
            # std of the mean = sample_std / sqrt(B)
            cross_std  = flux_means.std(axis=0, ddof=1) / np.sqrt(B)
            with np.errstate(divide="ignore", invalid="ignore"):
                re = np.where(cross_mean != 0.0,
                              cross_std / np.abs(cross_mean),
                              np.inf)
            stats["flux"] = {
                "energy_bins"   : self.batch_results[0]["flux"]["energy_bins"],
                "mean"          : cross_mean.tolist(),
                "std"           : cross_std.tolist(),
                "relative_error": re.tolist(),
            }

        # ── verification tally ────────────────────────────────────────────
        if "verif" in self.batch_results[0]:
            verif_stats = {}
            # quantities that are 2-D (n_space × n_energy)
            for key in ("absorption", "scatter"):
                arr = np.array([r["verif"][key]["mean"]
                                for r in self.batch_results])   # (B, ns, ne)
                cm  = arr.mean(axis=0)
                cs  = arr.std(axis=0, ddof=1) / np.sqrt(B)
                with np.errstate(divide="ignore", invalid="ignore"):
                    re = np.where(cm != 0.0, cs / np.abs(cm), np.inf)
                verif_stats[key] = {
                    "mean"          : cm.tolist(),
                    "std"           : cs.tolist(),
                    "relative_error": re.tolist(),
                }
            # quantities that are 1-D (n_surf,)
            for key in ("current_fwd", "current_bwd"):
                arr = np.array([r["verif"][key]["mean"]
                                for r in self.batch_results])   # (B, n_surf)
                cm  = arr.mean(axis=0)
                cs  = arr.std(axis=0, ddof=1) / np.sqrt(B)
                with np.errstate(divide="ignore", invalid="ignore"):
                    re = np.where(cm != 0.0, cs / np.abs(cm), np.inf)
                verif_stats[key] = {
                    "mean"          : cm.tolist(),
                    "std"           : cs.tolist(),
                    "relative_error": re.tolist(),
                }
            # scalar leakage quantities
            for key in ("leak_left", "leak_right"):
                vals = np.array([r["verif"][key]["mean"]
                                 for r in self.batch_results])  # (B,)
                cm   = float(vals.mean())
                cs   = float(vals.std(ddof=1) / np.sqrt(B))
                re   = cs / abs(cm) if cm != 0.0 else float("inf")
                verif_stats[key] = {"mean": cm, "std": cs, "relative_error": re}

            verif_stats["energy_bins"] = self.batch_results[0]["verif"]["energy_bins"]
            verif_stats["surface_xs"]  = self.batch_results[0]["verif"]["surface_xs"]
            stats["verif"] = verif_stats

        # ── performance metrics ───────────────────────────────────────────
        perf_keys = ["total_time_s", "neutrons_per_second",
                     "rejection_fraction", "cpu_efficiency"]
        perf_stats = {}
        for k in perf_keys:
            vals = np.array([r["perf"][k] for r in self.batch_results])
            perf_stats[k] = {
                "mean": float(vals.mean()),
                "std" : float(vals.std(ddof=1)),
            }
        # summed (not averaged) counters
        for k in ("n_neutrons", "n_real_collisions",
                  "n_virtual_collisions", "n_majorant_updates"):
            perf_stats[k] = int(sum(r["perf"][k] for r in self.batch_results))
        stats["perf"] = perf_stats

        return stats

    # ------------------------------------------------------------------
    def summary(self) -> str:

        # [FIX] Use the actual number of simulated neutrons, not the history list length
        n = max(self.perf.n_neutrons, 1)

        # [FIX] Calculate fractions using global counters instead of history objects
        abs_frac      = self.absorption_score / n
        leak_frac     = self.leakage_score    / n
        mean_scatters = self.scattering_score / n
        mean_virtual  = self.rejection_score  / n
        mean_path     = self.distance_score   / n

        if self.majorant_log_flag and self._majorant_log:
            maj_values  = [r.value  for r in self._majorant_log]
            maj_margins = [r.margin for r in self._majorant_log]
            mat_counts: dict = {}
            for r in self._majorant_log:
                mat_counts[r.limiting_material] = \
                    mat_counts.get(r.limiting_material, 0) + 1
            top_mat = max(mat_counts, key=mat_counts.get)
        else:
            maj_values  = [0.0]
            maj_margins = [0.0]
            top_mat     = "N/A"

        lines = [
            "=" * 60,
            "  SIMULATION SUMMARY",
            "=" * 60,
            f"  Neutrons simulated      : {n:>10,}",
            f"  Absorbed fraction       : {abs_frac:>10.3%}",
            f"  Leaked fraction         : {leak_frac:>10.3%}",
            "-" * 60,
            f"  Mean scatters/neutron   : {mean_scatters:>10.2f}",
            f"  Mean virtual/neutron    : {mean_virtual:>10.2f}",
            f"  Mean path length  [cm]  : {mean_path:>10.4f}",
            "-" * 60,
            "  MAJORANT XS DIAGNOSTICS",
            f"  Majorant updates        : {len(self._majorant_log):>10,}",
            f"  Mean majorant [cm⁻¹]    : {np.mean(maj_values):>10.4f}",
            f"  Max  majorant [cm⁻¹]    : {np.max(maj_values):>10.4f}",
            f"  Mean margin (maj/real)  : {np.mean(maj_margins):>10.3f}",
            f"  Min margin (maj/real)  : {np.min(maj_margins):>10.3f}",
            f"  Max margin (maj/real)  : {np.max(maj_margins):>10.3f}",
            f"  Most-limiting material  : {top_mat}",
            "=" * 60,
        ]

        if self.flux_tally_flag and self.flux_tally is not None:
            lines.append(self.flux_tally.summary())
            lines.append("=" * 60)

        if self.verif_tally_flag and self.verif_tally is not None:
            lines.append(self.verif_tally.summary())
            lines.append("=" * 60)

        lines.append(self.perf.summary())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def histories_to_dataframe(self):
        
        rows = []
        for h in self.histories:
            for i in range(h.n_steps):
                rows.append({
                    "neutron_id"  : h.neutron_id,
                    "birth_energy": h.birth_energy,
                    "step"        : i,
                    "event"       : h.events[i],
                    "fate"        : h.fate,
                    "x"           : h.positions[i][0],
                    "y"           : h.positions[i][1],
                    "z"           : h.positions[i][2],
                    "energy"      : h.energies[i],
                    "distance"    : h.distances[i],
                    "majorant_xs" : h.majorant_xs_at_step[i],
                    "local_xs"    : h.local_xs_at_step[i],
                    "material"    : h.material_at_step[i],
                })
        return pd.DataFrame(rows)

    def majorant_log_to_dataframe(self):
        
        rows = [
            {
                "energy"            : r.energy,
                "majorant_xs"       : r.value,
                "actual_max_xs"     : r.actual_max_xs,
                "margin"            : r.margin,
                "limiting_material" : r.limiting_material,
            }
            for r in self._majorant_log
        ]
        return pd.DataFrame(rows)


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