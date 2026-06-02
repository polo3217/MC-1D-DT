"""
reconr_parallel.py
==================
Parallelised drop-in replacement for build_majorant_xs_grid().

Strategy
--------
The RECONR stacking algorithm is embarrassingly parallel at the window level:
each WMP window [E_w, E_{w+1}] is independent — no point inserted in window w
affects window w+1.  We therefore:

  1. Split the initial point_grid into per-window slices.
  2. Run both the err_max pass AND the err_lim pass inside a single worker
     function (_process_window) for each window.
  3. Collect results in window order and concatenate.
  4. Rebuild the O(1) window pointer table from the merged grid.

Parallelism is provided by concurrent.futures.ProcessPoolExecutor so that
the GIL is not a bottleneck (calculate_mat_majorant_xs is CPU-bound).

Because worker processes cannot share the geometry object directly (it is not
picklable in general), we extract the XS evaluation callable into a
top-level function that receives only serialisable arguments.  The geometry
is passed as a picklable proxy if it supports __getstate__, otherwise we fall
back to multiprocessing with initializer-based sharing via a global.

WMP / ENDF preprocessing counts
---------------------------------
Both public entry points now return a 6-tuple whose last element is a dict:

    pp_counts = {
        "n_wmp"    : int,   # grid points evaluated via WMP
        "n_endf"   : int,   # grid points evaluated via ENDF fallback
        "time_wmp" : float, # cumulative wall-clock time for WMP evals (s)
        "time_endf": float, # cumulative wall-clock time for ENDF evals (s)
    }

These are collected from workers in the main process after the pool closes
and are therefore always accurate regardless of parallelism.
"""

from __future__ import annotations

import math
import os
import time
import numpy as np
import src.geometry_classes as geom
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Module-level geometry holder (used by worker processes after initialisation)
# ─────────────────────────────────────────────────────────────────────────────

_GEOMETRY = None   # set by _worker_init in each subprocess


def _worker_init(geometry):
    global _GEOMETRY
    _GEOMETRY = geometry


def _eval_xs(e: float) -> float:
    """Worker-side XS evaluation — uses the process-local geometry."""
    return _GEOMETRY.calculate_mat_majorant_xs(e)


# ─────────────────────────────────────────────────────────────────────────────
# Truncation helpers  (identical to original)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_midpoint(
    e_last: float, e_next: float
) -> Tuple[float, float, float, bool]:
    """
    Return (e_half, e_last_t, e_next_t, converged) using the same
    7-digit / 9-digit truncation logic as the original RECONR implementation.
    """
    e_half = (e_last + e_next) / 2.0
    e_h7   = float(f"{e_half:.7e}")
    e_l7   = float(f"{e_last:.7e}")
    e_n7   = float(f"{e_next:.7e}")

    if e_h7 == e_l7 or e_h7 == e_n7:
        e_h7 = float(f"{e_half:.9e}")
        e_l7 = float(f"{e_last:.9e}")
        e_n7 = float(f"{e_next:.9e}")
        if e_h7 == e_l7 or e_h7 == e_n7:
            return e_h7, e_l7, e_n7, True   # converged — cannot refine further

    return e_h7, e_l7, e_n7, False


# ─────────────────────────────────────────────────────────────────────────────
# WMP range helper
# ─────────────────────────────────────────────────────────────────────────────

def _is_in_wmp_range(e: float) -> bool:
    """
    Return True if energy e falls inside the WMP range of the geometry
    stored in _GEOMETRY.  Called inside workers — uses the process-local copy.

    For the mat-majorant path we check all nuclides in the geometry; if at
    least one covers e via WMP the evaluation is classified as WMP.
    For the nuclide path (_NUCLIDE_NAME is set) we check that single nuclide.
    """
    if _NUCLIDE_NAME is not None:
        # per-nuclide worker — check only the active nuclide
        wmp = geom.nuclide_objects[_NUCLIDE_NAME]['wmp']
        return wmp is not None and wmp.E_min <= e <= wmp.E_max

    # mat-majorant worker — check all nuclides in the geometry
    for name, _ in _GEOMETRY._nuclides.values():
        wmp = geom.nuclide_objects[name]['wmp']
        if wmp is not None and wmp.E_min <= e <= wmp.E_max:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-window worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_window(
    window_idx:  int,
    window_points: List[float],
    err_lim:     float,
    err_max:     float,
    err_int:     float,
) -> Tuple[int, List[float], List[float], int, int, float, float]:
    """
    Run both RECONR passes for a single WMP window.

    Returns
    -------
    (window_idx, energy_grid, xs_grid, n_wmp, n_endf, time_wmp, time_endf)

    n_wmp / n_endf   : number of unique XS evaluations in each regime.
    time_wmp / _endf : cumulative wall time spent in each regime (seconds).
    """
    # ── XS cache ─────────────────────────────────────────────────────────────
    _cache: dict[float, float] = {}

    # CHANGE: local counters for WMP vs ENDF evaluations within this window.
    # Keyed by energy so cache hits are not double-counted.
    _wmp_energies:  set = set()
    _endf_energies: set = set()
    _time_wmp:  float = 0.0
    _time_endf: float = 0.0

    def eval_xs_cached(e: float) -> float:
        nonlocal _time_wmp, _time_endf
        if e not in _cache:
            t0 = time.perf_counter()
            _cache[e] = _eval_xs(e)
            dt = time.perf_counter() - t0
            # CHANGE: classify and time each unique evaluation
            if _is_in_wmp_range(e):
                _wmp_energies.add(e)
                _time_wmp += dt
            else:
                _endf_energies.add(e)
                _time_endf += dt
        return _cache[e]

    # ── Pass 1 : err_max (coarse refinement) ─────────────────────────────────
    point_grid = list(window_points)
    energy_grid: List[float] = []
    xs_grid:     List[float] = []

    i = 0
    last_e = point_grid[-1]

    while i < len(point_grid) - 1:
        e_last = point_grid[i]
        e_next = point_grid[i + 1]

        if e_last >= last_e:
            break

        sigma_last = eval_xs_cached(e_last)
        sigma_next = eval_xs_cached(e_next)

        e_half, _, _, converged = _truncate_midpoint(e_last, e_next)
        sigma_half   = eval_xs_cached(e_half)
        sigma_interp = (sigma_last + sigma_next) / 2.0

        err = (abs(sigma_half - sigma_interp) / sigma_half
               if sigma_half != 0 else 0.0)

        if err > err_max and not converged:
            point_grid.insert(i + 1, e_half)
        else:
            energy_grid.append(e_last)
            xs_grid.append(sigma_last)
            i += 1

    energy_grid.append(point_grid[-1])
    xs_grid.append(eval_xs_cached(point_grid[-1]))

    # ── Pass 2 : err_lim (fine refinement) ───────────────────────────────────
    i = 0
    last_e = energy_grid[-1]

    while i < len(energy_grid) - 1:
        e      = energy_grid[i]
        e_next = energy_grid[i + 1]

        if e >= last_e:
            break

        sigma      = xs_grid[i]
        sigma_next = xs_grid[i + 1]

        e_half, _, _, converged = _truncate_midpoint(e, e_next)
        sigma_half   = eval_xs_cached(e_half)
        sigma_interp = (sigma + sigma_next) / 2.0

        err = (abs(sigma_half - sigma_interp) / sigma_half
               if sigma_half != 0 else 0.0)

        area = 0.5 * abs(sigma_half - sigma_interp) * (e_next - e)

        if err > err_lim and area > err_int and not converged:
            energy_grid.insert(i + 1, e_half)
            xs_grid.insert(i + 1, sigma_half)
        else:
            i += 1

    # CHANGE: return per-regime counts and times alongside the grid slices
    return (window_idx, energy_grid[:-1], xs_grid[:-1],
            len(_wmp_energies), len(_endf_energies), _time_wmp, _time_endf)


# ─────────────────────────────────────────────────────────────────────────────
# Window splitter
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_windows(
    point_grid: List[float],
    E_min:      float,
    E_spacing:  float,
) -> List[List[float]]:
    """
    Partition point_grid into per-window sublists.
    Each sublist contains the points whose window index equals w, PLUS the
    first point of window w+1 as a right boundary (so every sublist has ≥ 2
    points and the worker never needs to look outside its window).
    """
    window_of = [
        int((math.sqrt(e) - math.sqrt(E_min)) / E_spacing)
        for e in point_grid
    ]
    max_w = window_of[-1]

    windows: dict[int, List[int]] = {}
    for idx, w in enumerate(window_of):
        windows.setdefault(w, []).append(idx)

    slices: List[List[float]] = []
    for w in range(max_w + 1):
        if w not in windows:
            continue
        idxs = windows[w]
        pts  = [point_grid[i] for i in idxs]

        for w_next in range(w + 1, max_w + 2):
            if w_next in windows:
                pts.append(point_grid[windows[w_next][0]])
                break
            elif w_next > max_w:
                break

        if len(pts) >= 2:
            slices.append(pts)

    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — mat-majorant
# ─────────────────────────────────────────────────────────────────────────────

def build_majorant_xs_grid(
    geometry,
    err_lim:     float = 0.001,
    err_max:     float = 0.01,
    err_int:     float | None = None,
    last_window: int   | None = None,
    last_energy: float | None = None,
    n_workers:   int   | None = None,
) -> Tuple[List[float], List[float], float, float, List[int], Dict]:
    """
    Parallelised drop-in replacement for the serial build_majorant_xs_grid.

    Returns a 6-tuple: the original 5 values plus pp_counts (see module doc).
    """
    if err_int is None:
        err_int = err_lim / 20_000

    if n_workers is None:
        n_workers = os.cpu_count() - 2

    # ── Collect nuclides ──────────────────────────────────────────────────────
    if geometry.maj_mat_method == "maj_mat":
        materials = [geometry.maj_mat]
    elif geometry.maj_mat_method == "simple":
        materials = geometry.materials
    else:
        materials = geometry.materials

    nuclides: dict = {}
    for mat in materials:
        for nuclide_name, density in mat.nuclides:
            nuclides[nuclide_name] = geom.nuclide_objects[nuclide_name]['wmp']

    nuclide_list = list(nuclides.values())

    # ── Determine energy bounds and window structure ──────────────────────────
    E_min     = -np.inf
    E_max_nuc =  np.inf
    E_spacing =  np.inf
    minimum_spacing_nuclide = None

    for nuc in nuclide_list:
        if nuc.E_min > E_min:
            E_min = nuc.E_min
        if nuc.E_max < E_max_nuc:
            E_max_nuc = nuc.E_max
        if nuc.spacing <= E_spacing:
            E_spacing = nuc.spacing
            minimum_spacing_nuclide = nuc.name

    if last_energy is not None:
        E_max = last_energy
    else:
        E_max = E_max_nuc

    if E_max == E_max_nuc:
        n_windows = nuclides[minimum_spacing_nuclide].n_windows
    else:
        n_windows = int(math.ceil((math.sqrt(E_max) - math.sqrt(E_min)) / E_spacing)) + 1

    print(f"[reconr_parallel] {n_windows} WMP windows, "
          f"E ∈ [{E_min:.3e}, {E_max:.3e}] eV, "
          f"spacing = {E_spacing:.6f} √eV")
    print(f"[reconr_parallel] err_max={err_max}, err_lim={err_lim}, "
          f"err_int={err_int:.3e}")
    print(f"[reconr_parallel] workers = {n_workers}")

    # ── Build initial point grid ──────────────────────────────────────────────
    point_grid: List[float] = []
    for i in range(n_windows):
        e = (math.sqrt(E_min) + i * E_spacing) ** 2
        if e > E_max:
            break
        point_grid.append(e)

    if point_grid[-1] < E_max:
        point_grid.append(E_max)

    # ── Split into per-window slices ──────────────────────────────────────────
    window_slices    = _split_into_windows(point_grid, E_min, E_spacing)
    n_actual_windows = len(window_slices)
    print(f"[reconr_parallel] {n_actual_windows} non-empty windows to process")

    # ── Dispatch workers ──────────────────────────────────────────────────────
    results: dict[int, Tuple[List[float], List[float]]] = {}

    # CHANGE: accumulators for preprocessing WMP/ENDF counts collected from workers
    pp_n_wmp   = 0
    pp_n_endf  = 0
    pp_t_wmp   = 0.0
    pp_t_endf  = 0.0

    if n_workers == 1:
        _worker_init(geometry)
        for w_idx, pts in enumerate(window_slices):
            w_idx_out, e_local, xs_local, n_wmp, n_endf, t_wmp, t_endf = _process_window(
                w_idx, pts, err_lim, err_max, err_int
            )
            results[w_idx] = (e_local, xs_local)
            # CHANGE: accumulate per-window counts into preprocessing totals
            pp_n_wmp  += n_wmp;  pp_n_endf  += n_endf
            pp_t_wmp  += t_wmp;  pp_t_endf  += t_endf
            if w_idx % max(1, n_actual_windows // 10) == 0:
                print(f"  window {w_idx:4d}/{n_actual_windows}  "
                      f"({100*w_idx/n_actual_windows:.0f}%)")
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(geometry,),
        ) as pool:
            futures = {
                pool.submit(_process_window, w_idx, pts, err_lim, err_max, err_int): w_idx
                for w_idx, pts in enumerate(window_slices)
            }
            done = 0
            for future in as_completed(futures):
                w_idx, e_local, xs_local, n_wmp, n_endf, t_wmp, t_endf = future.result()
                results[w_idx] = (e_local, xs_local)
                # CHANGE: accumulate per-window counts into preprocessing totals
                pp_n_wmp  += n_wmp;  pp_n_endf  += n_endf
                pp_t_wmp  += t_wmp;  pp_t_endf  += t_endf
                done += 1
                if done % max(1, n_actual_windows // 10) == 0:
                    print(f"  {done:4d}/{n_actual_windows} windows done  "
                          f"({100*done/n_actual_windows:.0f}%)")

    # ── Merge in window order ─────────────────────────────────────────────────
    energy_grid: List[float] = []
    xs_grid:     List[float] = []

    for w_idx in range(n_actual_windows):
        e_local, xs_local = results[w_idx]
        energy_grid.extend(e_local)
        xs_grid.extend(xs_local)

    # Re-append the very last point (dropped by all workers to avoid duplicates)
    last_e = window_slices[-1][-1]
    energy_grid.append(last_e)
    # CHANGE: classify and time this final evaluation too
    t0 = time.perf_counter()
    xs_last = geometry.calculate_mat_majorant_xs(last_e)
    dt = time.perf_counter() - t0
    xs_grid.append(xs_last)
    if any(
        geom.nuclide_objects[name]['wmp'] is not None
        and geom.nuclide_objects[name]['wmp'].E_min <= last_e <= geom.nuclide_objects[name]['wmp'].E_max
        for name, _ in geometry._nuclides.values()
    ):
        pp_n_wmp += 1; pp_t_wmp += dt
    else:
        pp_n_endf += 1; pp_t_endf += dt

    print(f"[reconr_parallel] merged grid: {len(energy_grid)} points")
    # CHANGE: print preprocessing split summary
    print(f"[reconr_parallel] preprocessing evaluations — "
          f"WMP: {pp_n_wmp:,} ({pp_t_wmp:.2f}s)  "
          f"ENDF: {pp_n_endf:,} ({pp_t_endf:.2f}s)")

    # ── Safety margin ─────────────────────────────────────────────────────────
    e_arr  = np.array(energy_grid)
    xs_arr = np.array(xs_grid) * (1.0 + err_max)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    diffs = np.diff(e_arr)
    if np.any(diffs <= 0):
        n_inv = (diffs <= 0).sum()
        print(f"[reconr_parallel] WARNING: {n_inv} inversions before dedup "
              f"(worst: {diffs[diffs<=0].min():.3e} eV)")
    mask   = np.concatenate(([True], diffs > 0))
    e_arr  = e_arr[mask]
    xs_arr = xs_arr[mask]
    print(f"[reconr_parallel] dedup removed {(~mask).sum()} points, "
          f"final grid: {len(e_arr)} points")

    e_grid_list  = e_arr.tolist()
    xs_grid_list = xs_arr.tolist()

    # ── O(1) window pointer table ─────────────────────────────────────────────
    print("[reconr_parallel] building window pointer table...")
    window_pointers = [0]
    current_window  = 0

    for idx, e in enumerate(e_grid_list):
        w = int((math.sqrt(e) - math.sqrt(E_min)) / E_spacing)
        while w > current_window:
            window_pointers.append(idx)
            current_window += 1

    window_pointers.append(len(e_grid_list))

    print(f"[reconr_parallel] window pointers: {len(window_pointers)-1} windows")
    print(f"[reconr_parallel] E_first={e_grid_list[0]:.4e} eV, "
          f"E_last={e_grid_list[-1]:.4e} eV")

    # CHANGE: bundle preprocessing counts for the caller (geometry_classes)
    pp_counts = {
        "n_wmp"    : pp_n_wmp,
        "n_endf"   : pp_n_endf,
        "time_wmp" : pp_t_wmp,
        "time_endf": pp_t_endf,
    }

    return e_grid_list, xs_grid_list, math.sqrt(E_min), E_spacing, window_pointers, pp_counts


# ─────────────────────────────────────────────────────────────────────────────
# Per-nuclide worker globals
# ─────────────────────────────────────────────────────────────────────────────

_NUCLIDE_NAME = None   # set by _nuclide_worker_init in each subprocess


def _nuclide_worker_init(geometry, nuclide_name):
    global _GEOMETRY, _NUCLIDE_NAME
    _GEOMETRY     = geometry
    _NUCLIDE_NAME = nuclide_name


def _eval_xs_nuclide(e: float) -> float:
    """Worker-side per-nuclide XS evaluation."""
    return _GEOMETRY.calculate_nuclide_majorant_xs(energy=e, nuclide_name=_NUCLIDE_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Per-window worker — nuclide variant
# ─────────────────────────────────────────────────────────────────────────────

def _process_window_nuclide(
    window_idx:    int,
    window_points: List[float],
    err_lim:       float,
    err_max:       float,
    err_int:       float,
) -> Tuple[int, List[float], List[float], int, int, float, float]:
    """
    Identical logic to _process_window but calls _eval_xs_nuclide.

    Returns
    -------
    (window_idx, energy_grid, xs_grid, n_wmp, n_endf, time_wmp, time_endf)
    """
    _cache: dict[float, float] = {}

    # CHANGE: same local WMP/ENDF accounting as _process_window
    _wmp_energies:  set = set()
    _endf_energies: set = set()
    _time_wmp:  float = 0.0
    _time_endf: float = 0.0

    def eval_xs_cached(e: float) -> float:
        nonlocal _time_wmp, _time_endf
        if e not in _cache:
            t0 = time.perf_counter()
            _cache[e] = _eval_xs_nuclide(e)
            dt = time.perf_counter() - t0
            # CHANGE: classify using the nuclide-specific WMP range
            if _is_in_wmp_range(e):
                _wmp_energies.add(e)
                _time_wmp += dt
            else:
                _endf_energies.add(e)
                _time_endf += dt
        return _cache[e]

    # ── Pass 1 : err_max ─────────────────────────────────────────────────────
    point_grid = list(window_points)
    energy_grid: List[float] = []
    xs_grid:     List[float] = []

    i = 0
    last_e = point_grid[-1]

    while i < len(point_grid) - 1:
        e_last = point_grid[i]
        e_next = point_grid[i + 1]
        if e_last >= last_e:
            break

        sigma_last   = eval_xs_cached(e_last)
        sigma_next   = eval_xs_cached(e_next)
        e_half, _, _, converged = _truncate_midpoint(e_last, e_next)
        sigma_half   = eval_xs_cached(e_half)
        sigma_interp = (sigma_last + sigma_next) / 2.0
        err = (abs(sigma_half - sigma_interp) / sigma_half
               if sigma_half != 0 else 0.0)

        if err > err_max and not converged:
            point_grid.insert(i + 1, e_half)
        else:
            energy_grid.append(e_last)
            xs_grid.append(sigma_last)
            i += 1

    energy_grid.append(point_grid[-1])
    xs_grid.append(eval_xs_cached(point_grid[-1]))

    # ── Pass 2 : err_lim ─────────────────────────────────────────────────────
    i = 0
    last_e = energy_grid[-1]

    while i < len(energy_grid) - 1:
        e      = energy_grid[i]
        e_next = energy_grid[i + 1]
        if e >= last_e:
            break

        sigma      = xs_grid[i]
        sigma_next = xs_grid[i + 1]
        e_half, _, _, converged = _truncate_midpoint(e, e_next)
        sigma_half   = eval_xs_cached(e_half)
        sigma_interp = (sigma + sigma_next) / 2.0
        err  = (abs(sigma_half - sigma_interp) / sigma_half
                if sigma_half != 0 else 0.0)
        area = 0.5 * abs(sigma_half - sigma_interp) * (e_next - e)

        if err > err_lim and area > err_int and not converged:
            energy_grid.insert(i + 1, e_half)
            xs_grid.insert(i + 1, sigma_half)
        else:
            i += 1

    # CHANGE: return per-regime counts and times
    return (window_idx, energy_grid[:-1], xs_grid[:-1],
            len(_wmp_energies), len(_endf_energies), _time_wmp, _time_endf)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — per-nuclide
# ─────────────────────────────────────────────────────────────────────────────

def build_majorant_xs_nuclide(
    geometry,
    nuclide_name: str,
    err_lim:     float = 0.001,
    err_max:     float = 0.01,
    err_int:     float | None = None,
    last_energy: float | None = None,
    last_window: int   | None = None,
    n_workers:   int   | None = None,
) -> Tuple[List[float], List[float], float, float, List[int], Dict]:
    """
    Parallelised per-nuclide majorant grid builder.

    Returns a 6-tuple: the original 5 values plus pp_counts (see module doc).
    """
    if err_int is None:
        err_int = err_lim / 20_000

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)

    nuclide   = geom.nuclide_objects[nuclide_name]['wmp']
    E_min     = nuclide.E_min
    E_max     = nuclide.E_max
    E_spacing = nuclide.spacing
    n_windows = nuclide.n_windows

    if last_energy is not None:
        E_max = last_energy
    
    if E_max == nuclide.E_max:
        n_windows = nuclide.n_windows
    else:
        n_windows = int(math.ceil((math.sqrt(E_max) - math.sqrt(E_min)) / E_spacing))

    print(f"[reconr_parallel | {nuclide.name}] "
          f"{n_windows} windows, E ∈ [{E_min:.3e}, {E_max:.3e}] eV")
    print(f"[reconr_parallel | {nuclide.name}] "
          f"err_max={err_max}, err_lim={err_lim}, err_int={err_int:.3e}, "
          f"workers={n_workers}")

    # ── Initial point grid ────────────────────────────────────────────────────
    point_grid: List[float] = []
    for i in range(n_windows):
        e = (math.sqrt(E_min) + i * E_spacing) ** 2
        if e > E_max:
            print(f"Last point {e:.3e} eV exceeds E_max, stopping initial grid generation")
            break
        point_grid.append(e)

    if point_grid[-1] < E_max:
        point_grid.append(E_max)

    # ── Split into per-window slices ──────────────────────────────────────────
    window_slices    = _split_into_windows(point_grid, E_min, E_spacing)
    n_actual_windows = len(window_slices)
    print(f"[reconr_parallel | {nuclide.name}] "
          f"{n_actual_windows} non-empty windows to process")

    # ── Dispatch workers ──────────────────────────────────────────────────────
    results: dict[int, Tuple[List[float], List[float]]] = {}

    # CHANGE: accumulators for preprocessing WMP/ENDF counts
    pp_n_wmp  = 0;  pp_n_endf  = 0
    pp_t_wmp  = 0.0; pp_t_endf = 0.0

    if n_workers == 1:
        _nuclide_worker_init(geometry, nuclide_name)
        for w_idx, pts in enumerate(window_slices):
            _, e_local, xs_local, n_wmp, n_endf, t_wmp, t_endf = _process_window_nuclide(
                w_idx, pts, err_lim, err_max, err_int
            )
            results[w_idx] = (e_local, xs_local)
            # CHANGE: accumulate
            pp_n_wmp += n_wmp;  pp_n_endf += n_endf
            pp_t_wmp += t_wmp;  pp_t_endf += t_endf
            if w_idx % max(1, n_actual_windows // 10) == 0:
                print(f"  [{nuclide.name}] window {w_idx:4d}/{n_actual_windows} "
                      f"({100*w_idx/n_actual_windows:.0f}%)")
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_nuclide_worker_init,
            initargs=(geometry, nuclide_name),
        ) as pool:
            futures = {
                pool.submit(
                    _process_window_nuclide, w_idx, pts, err_lim, err_max, err_int
                ): w_idx
                for w_idx, pts in enumerate(window_slices)
            }
            done = 0
            for future in as_completed(futures):
                w_idx, e_local, xs_local, n_wmp, n_endf, t_wmp, t_endf = future.result()
                results[w_idx] = (e_local, xs_local)
                # CHANGE: accumulate
                pp_n_wmp += n_wmp;  pp_n_endf += n_endf
                pp_t_wmp += t_wmp;  pp_t_endf += t_endf
                done += 1
                if done % max(1, n_actual_windows // 10) == 0:
                    print(f"  [{nuclide.name}] {done:4d}/{n_actual_windows} done "
                          f"({100*done/n_actual_windows:.0f}%)")

    # ── Merge ─────────────────────────────────────────────────────────────────
    energy_grid: List[float] = []
    xs_grid:     List[float] = []

    for w_idx in range(n_actual_windows):
        e_local, xs_local = results[w_idx]
        energy_grid.extend(e_local)
        xs_grid.extend(xs_local)

    # Re-append the final point dropped by the last worker
    last_e = window_slices[-1][-1]
    energy_grid.append(last_e)
    # CHANGE: classify the final evaluation
    t0 = time.perf_counter()
    xs_last = geometry.calculate_nuclide_majorant_xs(energy=last_e, nuclide_name=nuclide_name)
    dt = time.perf_counter() - t0
    xs_grid.append(xs_last)
    wmp = geom.nuclide_objects[nuclide_name]['wmp']
    if wmp is not None and wmp.E_min <= last_e <= wmp.E_max:
        pp_n_wmp += 1; pp_t_wmp += dt
    else:
        pp_n_endf += 1; pp_t_endf += dt

    print(f"[reconr_parallel | {nuclide.name}] merged: {len(energy_grid)} points")
    # CHANGE: print preprocessing split summary
    print(f"[reconr_parallel | {nuclide.name}] preprocessing evaluations — "
          f"WMP: {pp_n_wmp:,} ({pp_t_wmp:.2f}s)  "
          f"ENDF: {pp_n_endf:,} ({pp_t_endf:.2f}s)")

    # ── Safety margin + dedup ─────────────────────────────────────────────────
    e_arr  = np.array(energy_grid)
    xs_arr = np.array(xs_grid) * (1.0 + err_max)

    diffs = np.diff(e_arr)
    if np.any(diffs <= 0):
        print(f"[reconr_parallel | {nuclide.name}] WARNING: "
              f"{(diffs <= 0).sum()} inversions before dedup")
    mask   = np.concatenate(([True], diffs > 0))
    e_arr  = e_arr[mask]
    xs_arr = xs_arr[mask]
    print(f"[reconr_parallel | {nuclide.name}] "
          f"dedup removed {(~mask).sum()} points, final: {len(e_arr)}")

    e_grid_list  = e_arr.tolist()
    xs_grid_list = xs_arr.tolist()

    # ── Window pointer table ──────────────────────────────────────────────────
    window_pointers = [0]
    current_window  = 0
    for idx, e in enumerate(e_grid_list):
        w = int((math.sqrt(e) - math.sqrt(E_min)) / E_spacing)
        while w > current_window:
            window_pointers.append(idx)
            current_window += 1
    window_pointers.append(len(e_grid_list))

    print(f"[reconr_parallel | {nuclide.name}] "
          f"E_first={e_grid_list[0]:.4e}, E_last={e_grid_list[-1]:.4e}")

    # CHANGE: bundle preprocessing counts for the caller
    pp_counts = {
        "n_wmp"    : pp_n_wmp,
        "n_endf"   : pp_n_endf,
        "time_wmp" : pp_t_wmp,
        "time_endf": pp_t_endf,
    }

    return e_grid_list, xs_grid_list, math.sqrt(E_min), E_spacing, window_pointers, pp_counts

# ─────────────────────────────────────────────────────────────────────────────
# Per-material worker globals
# ─────────────────────────────────────────────────────────────────────────────
 
_MATERIAL = None  # set by _material_worker_init in each subprocess
 
 
def _material_worker_init(geometry, material):
    global _GEOMETRY, _MATERIAL
    _GEOMETRY = geometry
    _MATERIAL = material
 
 
def _eval_xs_material(e: float) -> float:
    """Worker-side per-material XS evaluation — uses process-local globals."""
    total = 0.0
    for nuclide_name, nuclide_density in _MATERIAL.nuclides:
        total += nuclide_density * _GEOMETRY.calculate_nuclide_majorant_xs(
            e, nuclide_name,
        )
    return total
 
 
def _process_window_material(
    window_idx:    int,
    window_points: List[float],
    err_lim:       float,
    err_max:       float,
    err_int:       float,
) -> Tuple[int, List[float], List[float]]:
    """
    Identical to _process_window but calls _eval_xs_material instead of
    _eval_xs, so it uses the per-material evaluator set by
    _material_worker_init. Returns a 3-tuple (no pp_counts needed).
    """
    _cache: dict[float, float] = {}
 
    def eval_xs_cached(e: float) -> float:
        if e not in _cache:
            _cache[e] = _eval_xs_material(e)
        return _cache[e]
 
    # ── Pass 1 : err_max coarse refinement ───────────────────────────────────
    point_grid = list(window_points)
    energy_grid: List[float] = []
    xs_grid:     List[float] = []
 
    i = 0
    last_e = point_grid[-1]
    while i < len(point_grid) - 1:
        e_last = point_grid[i]
        e_next = point_grid[i + 1]
        if e_last >= last_e: break
 
        s_last = eval_xs_cached(e_last)
        s_next = eval_xs_cached(e_next)
        e_half, _, _, converged = _truncate_midpoint(e_last, e_next)
        s_half   = eval_xs_cached(e_half)
        s_interp = (s_last + s_next) / 2.0
        err = abs(s_half - s_interp) / s_half if s_half != 0 else 0.0
 
        if err > err_max and not converged:
            point_grid.insert(i + 1, e_half)
        else:
            energy_grid.append(e_last)
            xs_grid.append(s_last)
            i += 1
 
    energy_grid.append(point_grid[-1])
    xs_grid.append(eval_xs_cached(point_grid[-1]))
 
    # ── Pass 2 : err_lim fine refinement ─────────────────────────────────────
    i = 0
    last_e = energy_grid[-1]
    while i < len(energy_grid) - 1:
        e      = energy_grid[i]
        e_next = energy_grid[i + 1]
        if e >= last_e: break
 
        s      = xs_grid[i]
        s_next = xs_grid[i + 1]
        e_half, _, _, converged = _truncate_midpoint(e, e_next)
        s_half   = eval_xs_cached(e_half)
        s_interp = (s + s_next) / 2.0
        err  = abs(s_half - s_interp) / s_half if s_half != 0 else 0.0
        area = 0.5 * abs(s_half - s_interp) * (e_next - e)
 
        if err > err_lim and area > err_int and not converged:
            energy_grid.insert(i + 1, e_half)
            xs_grid.insert(i + 1, s_half)
        else:
            i += 1
 
    return window_idx, energy_grid[:-1], xs_grid[:-1]
 
 
def build_majorant_xs_material(
    geometry,
    material,
    err_lim      : float        = 0.001,
    err_max      : float        = 0.01,
    err_int      : float | None = None,
    last_energy  : float | None = None,
    last_window  : int   | None = None,
    n_workers    : int   | None = None,
) -> Tuple[List[float], List[float], float, float, List[int]]:
    """
    Build a RECONR-style majorant XS grid for a single material.
    Same algorithm as build_majorant_xs_grid() but scoped to one material.
 
    Returns
    -------
    5-tuple: (e_grid_list, xs_grid_list, sqrt_E_min, e_spacing, window_pointers)
    """
    if err_int is None:
        err_int = err_lim / 20_000
 
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)
 
    # ── Collect nuclides from the material ────────────────────────────────────
    nuclides: dict = {}
    for pair in material.nuclides:
        nuclide_name = pair[0]
        nuclide_density = pair[1]
        wmp = geom.nuclide_objects[nuclide_name]['wmp']
        if wmp is not None:
            nuclides[nuclide_name] = wmp
 
    if not nuclides:
        raise ValueError(f"No WMP nuclides found in material '{material.name}'.")
 
    # ── Energy bounds and window structure ────────────────────────────────────
    E_min     = -np.inf
    E_max_nuc =  np.inf
    E_spacing =  np.inf
    minimum_spacing_nuclide = None
 
    for name, nuc in nuclides.items():
        if nuc.E_min > E_min:       E_min = nuc.E_min
        if nuc.E_max < E_max_nuc:   E_max_nuc = nuc.E_max
        if nuc.spacing <= E_spacing:
            E_spacing = nuc.spacing
            minimum_spacing_nuclide = name
 
    E_max = last_energy if last_energy is not None else E_max_nuc
 
    if E_max == E_max_nuc:
        n_windows = nuclides[minimum_spacing_nuclide].n_windows
    else:
        n_windows = int(math.ceil(
            (math.sqrt(E_max) - math.sqrt(E_min)) / E_spacing
        )) + 1
 
    print(f"[build_majorant_xs_materials | {material.name}] {n_windows} WMP windows, "
          f"E ∈ [{E_min:.3e}, {E_max:.3e}] eV, spacing = {E_spacing:.6f} √eV")
    print(f"[build_majorant_xs_materials | {material.name}] "
          f"err_max={err_max}, err_lim={err_lim}, err_int={err_int:.3e}, "
          f"workers={n_workers}")
 
    # ── Initial point grid ────────────────────────────────────────────────────
    point_grid: List[float] = []
    for i in range(n_windows):
        e = (math.sqrt(E_min) + i * E_spacing) ** 2
        if e > E_max:
            break
        point_grid.append(e)
 
    if point_grid[-1] < E_max:
        point_grid.append(E_max)
 
    # ── Split into per-window slices ──────────────────────────────────────────
    window_slices    = _split_into_windows(point_grid, E_min, E_spacing)
    n_actual_windows = len(window_slices)
    print(f"[build_majorant_xs_materials | {material.name}] "
          f"{n_actual_windows} non-empty windows")
 
    # ── Dispatch workers ──────────────────────────────────────────────────────
    # CHANGE: now fully parallel — XS evaluation moved to module-level
    # _eval_xs_material / _process_window_material so workers can pickle it.
    # Previously this was a serial loop over a closure (not picklable).
    results: dict[int, Tuple[List[float], List[float]]] = {}
 
    if n_workers == 1:
        _material_worker_init(geometry, material)
        for w_idx, pts in enumerate(window_slices):
            _, e_local, xs_local = _process_window_material(
                w_idx, pts, err_lim, err_max, err_int
            )
            results[w_idx] = (e_local, xs_local)
            if w_idx % max(1, n_actual_windows // 10) == 0:
                print(f"  [{material.name}] window {w_idx:4d}/{n_actual_windows} "
                      f"({100*w_idx/n_actual_windows:.0f}%)")
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_material_worker_init,
            initargs=(geometry, material),
        ) as pool:
            futures = {
                pool.submit(
                    _process_window_material, w_idx, pts, err_lim, err_max, err_int
                ): w_idx
                for w_idx, pts in enumerate(window_slices)
            }
            done = 0
            for future in as_completed(futures):
                w_idx = futures[future]
                _, e_local, xs_local = future.result()
                results[w_idx] = (e_local, xs_local)
                done += 1
                if done % max(1, n_actual_windows // 10) == 0:
                    print(f"  [{material.name}] {done:4d}/{n_actual_windows} done "
                          f"({100*done/n_actual_windows:.0f}%)")
 
    # ── Merge ─────────────────────────────────────────────────────────────────
    energy_grid_full: List[float] = []
    xs_grid_full:     List[float] = []
    for w_idx in range(n_actual_windows):
        e_local, xs_local = results[w_idx]
        energy_grid_full.extend(e_local)
        xs_grid_full.extend(xs_local)
 
    last_e = window_slices[-1][-1]
    energy_grid_full.append(last_e)
    # CHANGE: use _eval_xs_material (module-level) instead of the old
    # eval_cached closure which no longer exists after the parallel refactor
    _material_worker_init(geometry, material)
    xs_grid_full.append(_eval_xs_material(last_e))
 
    print(f"[build_majorant_xs_materials | {material.name}] "
          f"merged: {len(energy_grid_full)} points")
 
    # ── Safety margin + dedup ─────────────────────────────────────────────────
    e_arr  = np.array(energy_grid_full)
    xs_arr = np.array(xs_grid_full) * (1.0 + err_max)
 
    diffs = np.diff(e_arr)
    if np.any(diffs <= 0):
        print(f"[build_majorant_xs_materials | {material.name}] WARNING: "
              f"{(diffs <= 0).sum()} inversions before dedup")
    mask   = np.concatenate(([True], diffs > 0))
    e_arr  = e_arr[mask]
    xs_arr = xs_arr[mask]
    print(f"[build_majorant_xs_materials | {material.name}] "
          f"dedup removed {(~mask).sum()} points, final: {len(e_arr)}")
 
    e_grid_list  = e_arr.tolist()
    xs_grid_list = xs_arr.tolist()
 
    # ── Window pointer table ──────────────────────────────────────────────────
    window_pointers = [0]
    current_window  = 0
    for idx, e in enumerate(e_grid_list):
        w = int((math.sqrt(e) - math.sqrt(E_min)) / E_spacing)
        while w > current_window:
            window_pointers.append(idx)
            current_window += 1
    window_pointers.append(len(e_grid_list))
 
    print(f"[build_majorant_xs_materials | {material.name}] "
          f"E_first={e_grid_list[0]:.4e}, E_last={e_grid_list[-1]:.4e}")
 
    return e_grid_list, xs_grid_list, math.sqrt(E_min), E_spacing, window_pointers