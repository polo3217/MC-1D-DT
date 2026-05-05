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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

import majorant_multipole as maj
from src.source_class import Source
from src.neutron_class import Neutron
from src.performance_classes import PerformanceTracker, MajorantRecord, NeutronHistory, MemoryTracker, BatchTimer
from src.tally_classes import FluxTallyTLE, VerificationTally
import src.group_xs as group_xs
import src.vectfit as vf
import src.discrete_evaluation as discrete
import src.reconr_v2 as reconr
import src.serpent_rothenstein as srp
import src.parallel as parallel
import src.sqrtT_E as sqrtT_E
import src.reconr_parallel as reconr_parallel

global  valid_nuclides_name 
global  valid_nuclides_list 
global  xs_dir
global xs_dir_sqrtT_E
global xs_dir_vectfit

valid_nuclides_name = [ "U238", "U235", "Pu239","Pu240", "Cnat", "O16", "H1", "Fe56", "Xe135", "Na23" ]    
valid_nuclides_list = [ '092238', '092235', '094239', '094240', '006000',  '008016', '001001','026056','054135', '011023']
xs_dir = '/home/paule/open_mc_projects/xs_lib/endfb-vii.1-hdf5/wmp'
xs_dir_sqrtT_E = '/home/paule/open_mc_projects/MC-1D_DT/structured_code/src/sqrtT_E_data'
xs_dir_vectfit = '/home/paule/open_mc_projects/MC-1D_DT/structured_code/src/vectfit_data'


# ==========================================
# --- Material class ---
# ==========================================
class Material:
    def __init__(self, name, nuclides=None, T=293.6):
        self.name     = name
        self.nuclides = nuclides if nuclides is not None else []
        for i, pair in enumerate(self.nuclides):
            print(f"[Material] Processing nuclide pair: {pair[0]} (Density: {pair[1]:.2e})")
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

@dataclass
class Region:
    name: str
    material: Material
    x_min: float
    x_max: float

# ==========================================
# --- Geometry class ---
# ==========================================
class Geometry:
    def __init__(self,
                 flux_tally=True,
                 verification_tally=True,
                 perf_tracker=True,
                 majorant_log=False,
                 history_flag=False, 
                 memory_flag=True, poll_interval : Optional[float] = 0.001,
                 verbose: bool = False):
        
        self._mode                = "analysis"
        self._maj_xs_method       = "discrete"
        self._maj_mat_method      = "simple" 
        self._access_method       = "fly"    

        self.T_array = None
        self.Q = 0
        self.n_points_per_window = 0
        self.T_min = 0
        self.T_max = 0
        self.cutoff_energy = 1.0e-5  

        self._materials      = []
        self._nuclides       = {}
        self.boundaries      = [0.0]  
        self._regions        = []
        self.material_array  = []
        self.source          = None
        self.verbose         = verbose

        self.boundary_conditions = {"left": "vacuum", "right": "vacuum"}
        
        self.maj_mat  : Material = None
        self.xs_maj_tables   = {}

        self.reconr_e_grid          = None
        self.reconr_maj_xs_grid     = None
        self.reconr_sqrt_E_min      = None
        self.reconr_e_spacing       = None
        self.reconr_window_pointers = None
        self._df_group_xs           = None

        # Score accumulators
        self.absorption_score        = 0
        self.scattering_score        = 0
        self.leakage_score           = 0
        self.acception_score         = 0
        self.rejection_score         = 0
        self.distance_score          = 0
        self.distance_sampling_score = 0
        self.wrong_majorant_score    = 0
        self.wrong_majorant_error_score = 0.0

        # Flags
        self.flux_tally_flag         = flux_tally
        self.verif_tally_flag        = verification_tally
        self.perf_tracker_flag       = perf_tracker
        self.majorant_log_flag       = majorant_log
        self.history_flag            = history_flag
        self.memory_tracker_flag     = memory_flag

        self.perf:         PerformanceTracker          = PerformanceTracker()
        self.memory:       MemoryTracker               = MemoryTracker(poll_interval=poll_interval)
        
        self.histories:    List[NeutronHistory]        = []
        self._majorant_log: List[MajorantRecord]       = []
        self.flux_tally:   Optional[FluxTallyTLE]      = None
        self.verif_tally:  Optional[VerificationTally] = None
        
        self.batch_results: list = []

    def reset(self):
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

        self.perf.reset_counters(keep_preprocessing=True)
        self.absorption_score = 0
        self.scattering_score = 0
        self.leakage_score = 0
        self.acception_score = 0
        self.rejection_score = 0
        self.distance_score = 0
        self.distance_sampling_score = 0
        self.wrong_majorant_score = 0
        self.wrong_majorant_error_score = 0.0

    def __getstate__(self):
        state = self.__dict__.copy()
        state['memory'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from src.performance_classes import MemoryTracker
        self.memory = MemoryTracker()

    @property
    def mode(self):
        return self._mode
    
    def set_mode(self, value: str, filename: str = None):
        if self.perf_tracker_flag: self.perf.start_preprocessing()
        if self.memory_tracker_flag: self.memory.start()
        
        if value not in ["analysis", "validation"]:
            raise ValueError("Mode must be 'analysis' or 'validation'")
        if value == "validation":
            if filename is None: raise ValueError("filename required for validation")
            self._mode = value
            self.set_maj_xs_method("group")
            self.maj_method = "simple"
            self.df_group_xs = group_xs.get_group_xs(filename, verbose=self.verbose)
        else:
            self._mode = value
        
        if self.memory_tracker_flag: self.memory.stop()
        if self.perf_tracker_flag: self.perf.stop_preprocessing()

    @property 
    def maj_xs_method(self):
        return self._maj_xs_method
    
    def set_maj_xs_method(self, method: str = "serpent", verbose: bool = False,
                          T_array: Optional[np.ndarray] = None,
                          xs_maj_file_dir : Optional[str] = None,
                          Q: Optional[int] = 2, n_points_per_window: Optional[int] = 100,
                          T_bound_method: Optional[str] = "from_materials", 
                          T_bounds: Optional[List[float]] = None):
        if self.perf_tracker_flag: self.perf.start_preprocessing()
        if self.memory_tracker_flag: self.memory.start()
        
        if method not in ["serpent", "vectfit", "sqrtT_E", "group", "discrete"]:
            raise ValueError("Invalid maj_xs_method")
        self._maj_xs_method = method

        if method == "vectfit":
            xs_maj_file_dir = xs_maj_file_dir or xs_dir_vectfit
            print(f"Loading tables from default directory: {xs_maj_file_dir}")
            self.xs_maj_tables = vf.xs_majorant_tables(file_dir=xs_maj_file_dir, verbose=verbose)

        if method == "discrete":
            if T_array is None: raise ValueError("T_array required for discrete method")
            self.T_array = T_array

        if method == "serpent":
            if self.verbose:
                print("  [Warning] Serpent/Rothenstein should be used with the reconr access method.")
            print("[Setup] Setting T bounds for Serpent/Rothenstein...")
            if T_bound_method == "user_defined":
                self.T_min, self.T_max = T_bounds
            elif T_bound_method == "from_materials":
                self.T_min = min(mat.T for mat in self.materials)
                self.T_max = max(mat.T for mat in self.materials)
            print(f"  [Setup] Serpent/Rothenstein T bounds set: {self.T_min} K to {self.T_max} K")
            self.Q = Q
            self.n_points_per_window = n_points_per_window

        if method == "sqrtT_E":
            xs_maj_file_dir = xs_maj_file_dir or xs_dir_sqrtT_E
            print("Loading table from:", xs_maj_file_dir)
            self.xs_maj_tables = sqrtT_E.xs_majorant_tables(file_dir=xs_maj_file_dir)
            if T_bound_method == "user_defined":
                self.T_min, self.T_max = T_bounds
            elif T_bound_method == "from_materials":
                self.T_min = min(mat.T for mat in self.materials)
                self.T_max = max(mat.T for mat in self.materials)
            print(f"  [Setup] T Bounds for sqrtT_E set: {self.T_min} K to {self.T_max} K")
            
        if self.memory_tracker_flag: self.memory.snapshot("maj_xs_method_setup")
        if self.memory_tracker_flag: self.memory.stop()
        if self.perf_tracker_flag: self.perf.stop_preprocessing()
        
    @property
    def maj_mat_method(self):
        return self._maj_mat_method

    @maj_mat_method.setter
    def maj_mat_method(self, value):
        if self.perf_tracker_flag: self.perf.start_preprocessing()
        if self.memory_tracker_flag: self.memory.start()
        
        self._maj_mat_method = value
        if value == "maj_mat":
            self.maj_mat = self.build_majorant_all_material(verbose_maj=True)
            
        if self.memory_tracker_flag: self.memory.snapshot("maj_mat_method_setup")
        if self.memory_tracker_flag: self.memory.stop()
        if self.perf_tracker_flag: self.perf.stop_preprocessing()

    @property
    def access_method(self):
        return self._access_method
    
    def set_access_method(self, value, err_lim=0.001, err_max=0.01, err_int=None,
                        last_window=None, last_energy=None):
        if self.perf_tracker_flag: self.perf.start_preprocessing()
        if self.memory_tracker_flag: self.memory.start()

        if value not in ["fly", "reconr"]: raise ValueError("Invalid access_method")
        self._access_method = value

        if value == "reconr":
            print("[Setup] Building majorant XS grid for RECONR access method...")
            self.reconr_e_grid, self.reconr_maj_xs_grid, self.reconr_sqrt_E_min, self.reconr_e_spacing, self.reconr_window_pointers = reconr_parallel.build_majorant_xs_grid(self, err_lim, err_max, err_int, last_window, last_energy)
        
        if self.memory_tracker_flag: self.memory.snapshot("access_method_setup")
        if self.memory_tracker_flag: self.memory.stop()
        if self.perf_tracker_flag: self.perf.stop_preprocessing()

    @property
    def majorant_log(self) -> List[MajorantRecord]:
        return self._majorant_log

    @property
    def df_group_xs(self):
        return self._df_group_xs

    @df_group_xs.setter
    def df_group_xs(self, df):
        self._df_group_xs = df

    @property
    def materials(self):
        return self._materials
    
    def get_region(self, name: str) -> "Region":
        for r in self._regions:
            if r.name == name: return r
        raise KeyError(f"[Geometry] No region named '{name}'.")

    def get_regions(self) -> list:
        return sorted(self._regions, key=lambda r: r.x_min)
    
    def add_region(self, name: str, material: Material, x_min: float, x_max: float):
        if x_max <= x_min: raise ValueError(f"x_max ({x_max}) must be > x_min ({x_min})")
        if any(r.name == name for r in self._regions): raise ValueError(f"Region named '{name}' exists")

        # Auto-split check
        for existing in self._regions:
            if existing.x_min < x_min and existing.x_max > x_max:
                left_region  = Region(name=f"{existing.name}_left", material=existing.material, x_min=existing.x_min, x_max=x_min)
                right_region = Region(name=f"{existing.name}_right", material=existing.material, x_min=x_max, x_max=existing.x_max)
                self._regions.remove(existing)
                self._regions.extend([left_region, right_region])
                break

        self._regions.append(Region(name=name, material=material, x_min=x_min, x_max=x_max))
        self._rebuild_boundaries()
        self.rebuild_materials()
        self.rebuild_nuclides()
        self._validate_no_gaps_or_overlaps()

    def rebuild_materials(self):
        if not self._regions:
            self.material_array = []
            self._materials = []
            return
        sorted_regions = sorted(self._regions, key=lambda r: r.x_min)
        self.material_array = [r.material for r in sorted_regions]
        unique_mats = []
        for r in sorted_regions:
            if r.material not in unique_mats: unique_mats.append(r.material)
        self._materials = unique_mats

    def rebuild_nuclides(self):
        self._nuclides = {}
        for mat in self.materials:
            for nuclide_obj, density in mat.nuclides:
                if nuclide_obj.name not in self._nuclides:
                    self._nuclides[nuclide_obj.name] = (nuclide_obj, density)

    def _rebuild_boundaries(self):
        if not self._regions:
            self.boundaries = [0.0]
            self.material_array = []
            return
        sorted_regions = sorted(self._regions, key=lambda r: r.x_min)
        self.boundaries = [sorted_regions[0].x_min] + [r.x_max for r in sorted_regions]
        self.material_array = [r.material for r in sorted_regions]
    
    def _validate_no_gaps_or_overlaps(self):
        sorted_regions = sorted(self._regions, key=lambda r: r.x_min)
        for i in range(1, len(sorted_regions)):
            prev, curr = sorted_regions[i-1], sorted_regions[i]
            if curr.x_min < prev.x_max: raise ValueError(f"Overlap: '{prev.name}' & '{curr.name}'")
            if curr.x_min > prev.x_max: raise ValueError(f"Gap: '{prev.name}' & '{curr.name}'")

    def add_material(self, mat: Material):
        if self.memory_tracker_flag: self.memory.start()
        if any(m.name == mat.name for m in self._materials): raise ValueError("Material exists")
        
        self._materials.append(mat)
        for nuclide in mat.nuclides:
            if nuclide[0].name not in self._nuclides:
                self._nuclides[nuclide[0].name] = nuclide
                
        if self.memory_tracker_flag:
            self.memory.snapshot(f"add_material_{mat.name}")
            self.memory.stop()

    @property
    def nuclides(self):
        return self._nuclides

    def draw(self, figsize: tuple = (12, 4), show_material_legend: bool = True):
        if not self._regions:
            print("[Geometry] No regions to draw.")
            return None, None

        sorted_regions = sorted(self._regions, key=lambda r: r.x_min)
        total_width = sorted_regions[-1].x_max - sorted_regions[0].x_min
        unique_materials = list({r.material.name: r.material for r in sorted_regions}.keys())
        colour_cycle = plt.cm.tab10.colors
        mat_colours = {name: colour_cycle[i % len(colour_cycle)] for i, name in enumerate(unique_materials)}

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(sorted_regions[0].x_min - 0.05 * total_width, sorted_regions[-1].x_max + 0.05 * total_width)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("x  [cm]", fontsize=11)
        ax.set_title("Geometry cross-section", fontsize=12, pad=10)

        for r in sorted_regions:
            colour = mat_colours[r.material.name]
            width = r.x_max - r.x_min
            rect = mpatches.FancyBboxPatch((r.x_min, 0.05), width, 0.90, boxstyle="square,pad=0",
                                           facecolor=colour, edgecolor="black", linewidth=0.8, alpha=0.55)
            ax.add_patch(rect)
            cx = r.x_min + width / 2
            ax.text(cx, 0.72, r.name, ha="center", va="center", fontsize=9, fontweight="bold", color="black")
            ax.text(cx, 0.38, r.material.name, ha="center", va="center", fontsize=8, color="black", style="italic")
            ax.text(cx, 0.20, f"T = {r.material.T:.0f} K", ha="center", va="center", fontsize=7.5, color="#444444")
            for xb in (r.x_min, r.x_max):
                ax.axvline(xb, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

        _bc_symbol = {"vacuum": "✕  vacuum", "reflective": "↔  reflective"}
        for side, xpos, ha in (("left", sorted_regions[0].x_min, "right"), ("right", sorted_regions[-1].x_max, "left")):
            bc = self.boundary_conditions[side]
            xoff = -0.01 * total_width if side == "left" else 0.01 * total_width
            ax.text(xpos + xoff, 0.50, _bc_symbol[bc], ha=ha, va="center", fontsize=8,
                    color="#c0392b" if bc == "vacuum" else "#2980b9", fontweight="bold")

        if show_material_legend:
            legend_handles = [mpatches.Patch(facecolor=mat_colours[name], edgecolor="black",
                              linewidth=0.6, alpha=0.7, label=name) for name in unique_materials]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.7, title="Materials", title_fontsize=8)

        ax.spines[["top", "left", "right"]].set_visible(False)
        fig.tight_layout()
        plt.show()
        return fig, ax

    def set_cutoff_energy(self, energy: float):
        if energy <= 0: raise ValueError("Cutoff energy must be positive.")
        self.cutoff_energy = energy
        print(f"[Setup] Cutoff energy set to {self.cutoff_energy:.2e} eV")

    def build_majorant_all_material(self, verbose_maj: bool = False) -> Material:
        nuclides = {}
        for mat in self.materials:
            for nuclide_obj, density in mat.nuclides:
                if nuclide_obj.name not in nuclides:
                    nuclides[nuclide_obj.name] = (nuclide_obj, density)
                else:
                    _, existing_density = nuclides[nuclide_obj.name]
                    nuclides[nuclide_obj.name] = (nuclide_obj, max(existing_density, density))
        
        mat_maj = object.__new__(Material)
        mat_maj.name = "maj_mat"
        mat_maj.xs = None
        mat_maj.nuclides = [(nuclide_obj, density) for nuclide_obj, density in nuclides.values()]
        mat_maj.total_density = float(sum(d for _, d in mat_maj.nuclides))
        return mat_maj

    def attach_flux_tally(self, energy_bins: List[float], transverse_area: float = 1.0, boundaries: Optional[List[float]] = None):
        if boundaries == None:
            boundaries = self.boundaries
        self.flux_tally = FluxTallyTLE(boundaries, energy_bins, transverse_area)

    def attach_verification_tally(self, energy_bins: List[float], surface_xs: List[float], boundaries: Optional[List[float]] = None):
        if boundaries == None:
            boundaries = self.boundaries
            
        self.verif_tally = VerificationTally(boundaries, energy_bins, surface_xs)

    def set_boundary_conditions(self, left: str = "vacuum", right: str = "vacuum"):
        valid = {"vacuum", "reflective"}
        if left not in valid or right not in valid: raise ValueError("Invalid Boundary Condition")
        self.boundary_conditions["left"]  = left
        self.boundary_conditions["right"] = right
        print(f"[Setup] Boundary conditions set: left='{left}', right='{right}'")

    def calculate_nuclide_majorant_xs(self, energy: float, nuclide, temperature=None) -> float:
        if self.maj_xs_method == "vectfit":
            maj_table = self.xs_maj_tables[nuclide.name]
            return maj.evaluate_sig_maj(nuclide, energy, maj_table)
        elif self.maj_xs_method == "sqrtT_E":
            maj_table = self.xs_maj_tables[nuclide.name]
            return sqrtT_E.evaluate_majorant_cross_section(multipole_data=nuclide, E=energy, data_table=maj_table, T_range=(self.T_min, self.T_max))
        elif self.maj_xs_method == "serpent":
            return srp.find_majorant_xs_rothenstein(nuclide, energy, T_min=self.T_min, T_max=self.T_max, Q=self.Q, n_points_per_window=self.n_points_per_window)
        elif self.maj_xs_method == "discrete":
            return discrete.discrete_majorant(energy, T_eval=temperature, T_array=self.T_array, nuclide=nuclide)
        return 0.0

    def caculate_mat_majorant_xs(self, energy: float) -> float:
        if self.maj_mat_method == "maj_mat":
            majorant_xs = 0.0
            for nuclide_obj, density in self.maj_mat.nuclides:
                majorant_xs += density * self.calculate_nuclide_majorant_xs(energy, nuclide_obj)
        elif self.maj_mat_method == "simple":
            majorant_xs = 0.0
            for mat in self.materials:
                mat_majorant_xs = 0.0
                for nuclide_obj, density in mat.nuclides:
                    mat_majorant_xs += density * self.calculate_nuclide_majorant_xs(energy, nuclide_obj, mat.T)
                if mat_majorant_xs > majorant_xs: majorant_xs = mat_majorant_xs
        return majorant_xs

    def access_majorant_xs(self, energy: float) -> float:
        if self.access_method == "fly":
            return self.caculate_mat_majorant_xs(energy)
        elif self.access_method == "reconr":
            if energy <= self.reconr_e_grid[0]: return self.reconr_maj_xs_grid[0]
            if energy >= self.reconr_e_grid[-1]: return self.reconr_maj_xs_grid[-1]
            w = int((math.sqrt(energy) - self.reconr_sqrt_E_min) / self.reconr_e_spacing)
            lo = self.reconr_window_pointers[w]
            hi = self.reconr_window_pointers[w + 1]
            i = bisect.bisect_right(self.reconr_e_grid, energy, lo, hi)
            E1, E2 = self.reconr_e_grid[i - 1], self.reconr_e_grid[i]
            xs1, xs2 = self.reconr_maj_xs_grid[i - 1], self.reconr_maj_xs_grid[i]
            return xs1 + (xs2 - xs1) * (energy - E1) / (E2 - E1)

    def get_majorant_xs(self, energy: float) -> float:
        if self.perf_tracker_flag:
            wall_t0 = time.perf_counter()
            cpu_t0  = time.process_time()

        if self.mode == "validation":
            actual_max_xs = 0.0
            limiting_mat_name = ""
            for cell in self.df_group_xs['cell'].unique():
                cell_df = self.df_group_xs[self.df_group_xs['cell'] == cell]
                total = cell_df['scatter'].sum() + cell_df['absorption'].sum()
                if total > actual_max_xs:
                    actual_max_xs = total
                    limiting_mat_name = f"Cell {cell}"
            majorant_xs = 2.0 * actual_max_xs
        else:
            majorant_xs = self.access_majorant_xs(energy)
            if self.majorant_log_flag:
                actual_max_xs = 0.0
                limiting_mat_name = ""
                for mat in self.materials:
                    xs_arr = mat._xs_evaluation(energy)
                    total  = float(xs_arr[0] + xs_arr[1])
                    if total > actual_max_xs:
                        actual_max_xs = total
                        limiting_mat_name = mat.name

        if self.majorant_log_flag:
            self._majorant_log.append(MajorantRecord(energy=energy, value=majorant_xs, limiting_material=limiting_mat_name, actual_max_xs=actual_max_xs))

        if self.perf_tracker_flag:
            self.perf.n_majorant_updates += 1
            self.perf.time_majorant += time.perf_counter() - wall_t0
            self.perf.cpu_time_majorant += time.process_time() - cpu_t0
        
        return majorant_xs

    def _sample_neutron_distance(self, majorant_xs: float) -> float:
        return -math.log(random.random()) / majorant_xs

    def _move_neutron(self, n: Neutron, distance: float):
        n.position[0] += n.direction[0] * distance
        n.position[1] += n.direction[1] * distance
        n.position[2] += n.direction[2] * distance

    def _get_material_at(self, x: float) -> Optional[Material]:
        if x < self.boundaries[0] or x >= self.boundaries[-1]: return None
        idx = bisect.bisect_right(self.boundaries, x) - 1
        return self.material_array[idx]

    def _evaluate_acceptance(self, n: Neutron, majorant_xs: float) -> bool:
        mat = self._get_material_at(n.position[0])
        if mat is None: return False
        
        if self.mode == "analysis":
            wall_t0 = time.perf_counter()
            cpu_t0  = time.process_time()

            if n.energy == n.last_eval_energy and mat == n.last_eval_mat:
                n.xs = n.last_eval_xs
            else:
                n.xs = mat._xs_evaluation(n.energy)
                n.last_eval_energy = n.energy
                n.last_eval_mat = mat
                n.last_eval_xs = n.xs

            self.perf.time_xs_eval += time.perf_counter() - wall_t0
            self.perf.cpu_time_xs_eval += time.process_time() - cpu_t0
            self.perf.n_xs_evaluations += 1

            local_xs = float(n.xs[0] + n.xs[1])
            acceptance_prob = local_xs / majorant_xs
            if acceptance_prob > 1.0:
                self.wrong_majorant_score += 1
                self.wrong_majorant_error_score += acceptance_prob - 1.0
                acceptance_prob = 1.0

            if random.random() < acceptance_prob:
                n.material = mat
                return True
            return False
        
        elif self.mode == "validation":
            wall_t0 = time.perf_counter()
            cpu_t0  = time.process_time()

            mat_name = 15 if mat.name == "cell 1" else 16
            cell_df = self.df_group_xs[self.df_group_xs['cell'] == mat_name]
            scatter_xs = cell_df['scatter'].sum()
            absorption_xs = cell_df['absorption'].sum()
            fission_xs = cell_df['fission'].sum() if 'fission' in cell_df.columns else 0.0
            n.xs = np.array([scatter_xs, absorption_xs, fission_xs])

            self.perf.time_xs_eval += time.perf_counter() - wall_t0
            self.perf.cpu_time_xs_eval += time.process_time() - cpu_t0
            self.perf.n_xs_evaluations += 1

            local_xs = float(n.xs[0] + n.xs[1])
            acceptance_prob = local_xs / majorant_xs

            if random.random() < acceptance_prob:
                n.material = mat
                return True
            return False

    def _sample_collision(self, n: Neutron) -> str:
        if n.energy < self.cutoff_energy: return "absorption"
        scatter_prob = n.xs[0] / (n.xs[0] + n.xs[1])
        return "scatter" if random.random() < scatter_prob else "absorption"

    def _scattering_neutron(self, n: Neutron):
        if self.mode == "analysis":
            contributions = [density * nuclide_obj(n.energy, n.material.T)[0] for nuclide_obj, density in n.material.nuclides]
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
            A = n.material.nuclides[0][0].sqrtAWR ** 2

        mu_cm = 2.0 * random.random() - 1.0
        phi   = 2.0 * math.pi * random.random()

        new_energy = n.energy * (A**2 + 2*A*mu_cm + 1) / (A + 1)**2
        n.energy   = max(new_energy, 1e-10)

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
            new_w = (1.0 if w >= 0 else -1.0) * mu_lab

        norm           = math.sqrt(new_u**2 + new_v**2 + new_w**2)
        n.direction[0] = new_u / norm
        n.direction[1] = new_v / norm
        n.direction[2] = new_w / norm

    def _run_neutron(self, n: Neutron, neutron_id: int = 0, track_neutron: bool = False):
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

        if self.flux_tally_flag and self.flux_tally is not None:
            n.current_energy_bin = self.flux_tally._energy_bin(n.energy)
            n.accumulated_distance = 0.0

        majorant_xs = self.get_majorant_xs(n.energy)
        
        # Sample the FIRST distance outside the loop
        distance = self._sample_neutron_distance(majorant_xs)

        while True:
            x_start = n.position[0]
            
            # --- UNIFIED BOUNDARY HANDLING (Piecewise Flight) ---
            # Calculate geometric distance to left and right boundaries
            tol = 1e-13 # Tolerance to prevent getting stuck exactly on the wall
            
            if n.direction[0] < -tol:
                dist_left = (self.boundaries[0] - n.position[0]) / n.direction[0]
            else:
                dist_left = float('inf')
                
            if n.direction[0] > tol:
                dist_right = (self.boundaries[-1] - n.position[0]) / n.direction[0]
            else:
                dist_right = float('inf')
                
            dist_to_wall = min(dist_left, dist_right)
            dist_to_wall = max(0.0, dist_to_wall) # Guard against floating point drift
            
            # Check if the sampled distance carries the neutron past the wall
            if dist_to_wall < distance:
                # --- WALL COLLISION ---
                # Move neutron exactly TO the boundary
                self._move_neutron(n, dist_to_wall)
                
                bc_side = "left" if dist_left < dist_right else "right"
                x_boundary = self.boundaries[0] if bc_side == "left" else self.boundaries[-1]
                
                # Snap exactly to boundary coordinate to remove floating point errors
                n.position[0] = x_boundary
                
                if self.flux_tally_flag and self.flux_tally is not None:
                    n.accumulated_distance += dist_to_wall
                self.distance_score += dist_to_wall
                
                bc_type = self.boundary_conditions[bc_side]

                if bc_type == "vacuum":
                    # TERMINATE
                    if self.verif_tally_flag and self.verif_tally is not None:
                        self.verif_tally.score_surface_crossing(x_start, x_boundary)
                        self.verif_tally.score_leakage(x_boundary)
                    
                    if self.flux_tally_flag and self.flux_tally is not None and n.current_energy_bin >= 0:
                        self.flux_tally._flux_tally.score(n.current_energy_bin, n.accumulated_distance)
                    
                    self.leakage_score += 1
                    
                    if self.history_flag:
                        hist.fate = "leakage"
                        hist.positions.append(n.position.copy())
                        hist.energies.append(n.energy)
                        hist.events.append("leakage")
                        hist.majorant_xs_at_step.append(majorant_xs)
                        hist.local_xs_at_step.append(0.0)
                        hist.material_at_step.append("outside")
                        hist.distances.append(dist_to_wall)

                    if track_neutron:
                        pos_out.append(n.position.copy())
                        dir_out.append(n.direction.copy())
                        eng_out.append(n.energy)
                        evt_out.append("leakage")
                    break 
                
                elif bc_type == "reflective":
                    # REDIRECT: Flip direction and deduct traveled distance
                    n.direction[0] *= -1.0
                    distance -= dist_to_wall  # The remaining distance to travel
                    
                    if self.history_flag:
                        hist.positions.append(n.position.copy())
                        hist.energies.append(n.energy)
                        hist.events.append("reflect")
                        hist.majorant_xs_at_step.append(majorant_xs)
                        hist.local_xs_at_step.append(0.0)
                        hist.material_at_step.append("boundary")
                        hist.distances.append(dist_to_wall)

                    if track_neutron:
                        pos_out.append(n.position.copy())
                        dir_out.append(n.direction.copy())
                        eng_out.append(n.energy)
                        evt_out.append("reflect")

                    # Loop around to simulate the rest of the flight (handles multiple bounces!)
                    continue

            # --- NO WALL COLLISION: NEUTRON STAYED INSIDE ---
            # Move the full sampled distance to the collision site
            self._move_neutron(n, distance) 
            self.distance_score += distance
            
            if self.verif_tally_flag and self.verif_tally is not None:
                self.verif_tally.score_surface_crossing(x_start, n.position[0])
            
            if self.flux_tally_flag and self.flux_tally is not None:
                n.accumulated_distance += distance

            # Evaluate Delta-Tracking Acceptance
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
                        self.verif_tally.score_collision(n.position[0], n.energy, 'absorption')
                        
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
                        self.verif_tally.score_collision(n.position[0], n.energy, 'scatter')

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

                    if self.flux_tally_flag and self.flux_tally is not None and n.current_energy_bin >= 0:
                        self.flux_tally._flux_tally.score(n.current_energy_bin, n.accumulated_distance)
                        n.accumulated_distance = 0.0

                    self._scattering_neutron(n)

                    if self.flux_tally_flag and self.flux_tally is not None:
                        n.current_energy_bin = self.flux_tally._energy_bin(n.energy)

                    majorant_xs = self.get_majorant_xs(n.energy)
                    distance = self._sample_neutron_distance(majorant_xs)

            else:
                # Virtual collision
                self.rejection_score            += 1
                self.perf.n_virtual_collisions  += 1
                if self.history_flag:
                    hist.n_virtual                 += 1

                mat_at_pos = self._get_material_at(n.position[0])
                local_xs   = 0.0
                mat_name   = ""
                if mat_at_pos is not None:
                    if self.mode == "analysis":
                        xs_arr = mat_at_pos._xs_evaluation(n.energy)
                    elif self.mode == "validation":
                        m_id = 15 if mat_at_pos.name == "cell 1" else 16
                        xs_scatter = self.df_group_xs[(self.df_group_xs['cell'] == m_id)]['scatter'].sum()
                        xs_absorption = self.df_group_xs[(self.df_group_xs['cell'] == m_id)]['absorption'].sum()
                        xs_arr = np.array([xs_scatter, xs_absorption, 0.0])
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
                    
                distance = self._sample_neutron_distance(majorant_xs)

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
    def run_source(self, src, track_neutron: bool = False, from_batch: bool = False):
        self.source          = src
        self.perf.n_neutrons = src.neutron_nbr

        if self.memory_tracker_flag:
            if not from_batch: self.memory.start()

        print(f"\n[Simulation] Running source (Mode: {self.mode})")
        if self.verbose:
            print(f"  -> Majorant XS method:       {self.maj_xs_method}")
            print(f"  -> Material majorant method: {self.maj_mat_method}")
            print(f"  -> Access method:            {self.access_method}")
        self.perf.start()

        tracks = []

        if hasattr(src, "neutrons"):
            neutron_iter = src.neutrons
            for nid, n in enumerate(neutron_iter):
                active_neutron = Neutron(energy=n.energy, position=n.position, direction=n.direction)
                result = self._run_neutron(active_neutron, neutron_id=nid, track_neutron=track_neutron)
                if track_neutron: tracks.append(result)
        else:
            for nid in range(src.neutron_nbr):
                n = src._sample_neutron()
                active_neutron = Neutron(energy=n.energy, position=n.position, direction=n.direction)
                result = self._run_neutron(active_neutron, neutron_id=nid, track_neutron=track_neutron)
                if track_neutron: tracks.append(result)

        self.perf.n_wrong_majorant            = self.wrong_majorant_score
        self.perf.n_wrong_majorant_mean_error = self.wrong_majorant_error_score
        self.perf.stop()

        if self.memory_tracker_flag:
            if not from_batch:
                self.memory.snapshot("post_source_run")
                self.memory.stop()
        
        return tracks if track_neutron else None

    # ------------------------------------------------------------------
    def _compute_batch_stats(self) -> dict:
        B = len(self.batch_results)
        stats = {"n_batches": B}

        if "flux" in self.batch_results[0]:
            flux_means = np.array([r["flux"]["flux"]["mean"] for r in self.batch_results])
            cross_mean = flux_means.mean(axis=0)
            cross_std  = flux_means.std(axis=0, ddof=1) / np.sqrt(B)
            with np.errstate(divide="ignore", invalid="ignore"):
                re = np.where(cross_mean != 0.0, cross_std / np.abs(cross_mean), np.inf)
            stats["flux"] = {
                "energy_bins"   : self.batch_results[0]["flux"]["energy_bins"],
                "mean"          : cross_mean.tolist(),
                "std"           : cross_std.tolist(),
                "relative_error": re.tolist(),
            }

        if "verif" in self.batch_results[0]:
            verif_stats = {}
            for key in ("absorption", "scatter"):
                arr = np.array([r["verif"][key]["mean"] for r in self.batch_results])
                cm  = arr.mean(axis=0)
                cs  = arr.std(axis=0, ddof=1) / np.sqrt(B)
                with np.errstate(divide="ignore", invalid="ignore"):
                    re = np.where(cm != 0.0, cs / np.abs(cm), np.inf)
                verif_stats[key] = {"mean": cm.tolist(), "std": cs.tolist(), "relative_error": re.tolist()}

            for key in ("current_fwd", "current_bwd"):
                arr = np.array([r["verif"][key]["mean"] for r in self.batch_results])
                cm  = arr.mean(axis=0)
                cs  = arr.std(axis=0, ddof=1) / np.sqrt(B)
                with np.errstate(divide="ignore", invalid="ignore"):
                    re = np.where(cm != 0.0, cs / np.abs(cm), np.inf)
                verif_stats[key] = {"mean": cm.tolist(), "std": cs.tolist(), "relative_error": re.tolist()}

            for key in ("leak_left", "leak_right"):
                vals = np.array([r["verif"][key]["mean"] for r in self.batch_results])
                cm   = float(vals.mean())
                cs   = float(vals.std(ddof=1) / np.sqrt(B))
                re   = cs / abs(cm) if cm != 0.0 else float("inf")
                verif_stats[key] = {"mean": cm, "std": cs, "relative_error": re}

            verif_stats["energy_bins"] = self.batch_results[0]["verif"]["energy_bins"]
            verif_stats["surface_xs"]  = self.batch_results[0]["verif"]["surface_xs"]
            verif_stats["boundaries"]  = self.batch_results[0]["verif"]["boundaries"]
            stats["verif"] = verif_stats

        # ── performance ───────────────────────────────────────────────────────
        # Float keys: compute mean ± std across batches
        perf_float_keys = [
            "total_time_s",          "total_cpu_time_s",
            "time_preprocessing_s",  "cpu_time_preprocessing_s",
            "time_run_source_s",     "cpu_time_run_source_s",
            "time_total_s",          "cpu_time_total_s",
            "time_majorant_s",       "cpu_time_majorant_s",
            "time_xs_eval_s",        "cpu_time_xs_eval_s",
            "neutrons_per_second",   "rejection_fraction",
            "cpu_efficiency",
        ]
        perf_stats = {}
        for k in perf_float_keys:
            vals = np.array([r["perf"][k] for r in self.batch_results
                             if k in r["perf"]])
            if len(vals) == 0:
                perf_stats[k] = {"mean": float("nan"), "std": float("nan")}
            else:
                perf_stats[k] = {
                    "mean": float(vals.mean()),
                    "std":  float(vals.std(ddof=1)),
                }

        # Integer keys: sum across all batches
        for k in ("n_neutrons", "n_real_collisions", "n_virtual_collisions",
                  "n_majorant_updates", "n_xs_evaluations"):
            perf_stats[k] = int(sum(r["perf"].get(k, 0) for r in self.batch_results))

        # Wrong majorant: mean ± std + min/max + total count
        for k in ("wrong_majorant_fraction", "wrong_majorant_mean_error"):
            vals = np.array([r["perf"][k] for r in self.batch_results])
            perf_stats[k] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std(ddof=1)),
                "min":  float(vals.min()),
                "max":  float(vals.max()),
            }
        perf_stats["n_wrong_majorant"] = int(
            sum(r["perf"]["n_wrong_majorant"] for r in self.batch_results)
        )

        stats["perf"] = perf_stats
        return stats

    # ------------------------------------------------------------------
    def summary(self) -> str:
        n = max(self.perf.n_neutrons, 1)
        abs_frac      = self.absorption_score / n
        leak_frac     = self.leakage_score    / n
        mean_scatters = self.scattering_score / n
        mean_virtual  = self.rejection_score  / n
        mean_path     = self.distance_score   / n
        mean_wrong_maj_error = (self.wrong_majorant_error_score / self.wrong_majorant_score) if self.wrong_majorant_score > 0 else 0.0
        wrong_majorant_Rate = self.wrong_majorant_score / len(self._majorant_log) if len(self._majorant_log) > 0 else 0.0

        if self.majorant_log_flag and self._majorant_log:
            maj_values  = [r.value  for r in self._majorant_log]
            maj_margins = [r.margin for r in self._majorant_log]
            mat_counts: dict = {}
            for r in self._majorant_log:
                mat_counts[r.limiting_material] = mat_counts.get(r.limiting_material, 0) + 1
            top_mat = max(mat_counts, key=mat_counts.get)
        else:
            maj_values  = [0.0]
            maj_margins = [0.0]
            top_mat     = "N/A"

        lines = [
            "=" * 60, "  SIMULATION SUMMARY", "=" * 60,
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
            f"  Min margin (maj/real)   : {np.min(maj_margins):>10.3f}",
            f"  Max margin (maj/real)   : {np.max(maj_margins):>10.3f}",
            f"  Most-limiting material  : {top_mat}",
            f"  Wrong majorant updates  : {self.wrong_majorant_score:>10,}",
            f"  Wrong majorant mean error sum : {mean_wrong_maj_error:.4f}",
            f"  Wrong majorant rate     : {wrong_majorant_Rate:.4f}",
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
        rows = [{"energy" : r.energy, "majorant_xs" : r.value, "actual_max_xs" : r.actual_max_xs,
                 "margin" : r.margin, "limiting_material" : r.limiting_material} for r in self._majorant_log]
        return pd.DataFrame(rows)
    
    def memory_summary(self) -> str:
        return self.memory.summary()
    
    def memory_log_to_dataframe(self):
        return self.memory.to_dataframe()
    

Geometry.run_batch          = parallel.run_batch
Geometry.run_batch_serial   = parallel.run_batch_serial
Geometry.run_batch_parallel = parallel.run_batch_parallel