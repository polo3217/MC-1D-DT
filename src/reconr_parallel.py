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
the GIL is not a bottleneck (caculate_mat_majorant_xs is CPU-bound).

Because worker processes cannot share the geometry object directly (it is not
picklable in general), we extract the XS evaluation callable into a
top-level function that receives only serialisable arguments.  The geometry
is passed as a picklable proxy if it supports __getstate__, otherwise we fall
back to multiprocessing with initializer-based sharing via a global.
"""

from __future__ import annotations

import math
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Module-level geometry holder (used by worker processes after initialisation)
# ─────────────────────────────────────────────────────────────────────────────

_GEOMETRY = None   # set by _worker_init in each subprocess


def _worker_init(geometry):
    global _GEOMETRY
    _GEOMETRY = geometry


def _eval_xs(e: float) -> float:
    """Worker-side XS evaluation — uses the process-local geometry."""
    return _GEOMETRY.caculate_mat_majorant_xs(e)


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
# Per-window worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_window(
    window_idx:  int,
    window_points: List[float],   # initial point_grid slice for this window
    err_lim:     float,
    err_max:     float,
    err_int:     float,
) -> Tuple[int, List[float], List[float]]:
    """
    Run both RECONR passes for a single WMP window.

    Parameters
    ----------
    window_idx    : window index (returned unchanged for ordered collection)
    window_points : energy points belonging to this window  (≥ 2 points)
    err_lim, err_max, err_int : tolerance parameters

    Returns
    -------
    (window_idx, energy_grid, xs_grid)  — local grids for this window,
    NOT including the last point (to avoid duplication at window boundaries).
    """
    # ── XS cache ─────────────────────────────────────────────────────────────
    # When a midpoint is rejected (err > threshold) it becomes a new grid
    # point, and its XS value sigma_half is reused in the very next iteration
    # as sigma_last.  Without a cache that value would be recomputed.
    # The cache also bridges pass 1 → pass 2: every point accepted in pass 1
    # is already cached, so pass 2 only pays for genuinely new midpoints.
    _cache: dict[float, float] = {}

    def eval_xs_cached(e: float) -> float:
        if e not in _cache:
            _cache[e] = _eval_xs(e)
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
            point_grid.insert(i + 1, e_half)   # refine in-place
        else:
            energy_grid.append(e_last)
            xs_grid.append(sigma_last)
            i += 1

    # Add the last point of this window
    energy_grid.append(point_grid[-1])
    xs_grid.append(eval_xs_cached(point_grid[-1]))

    # ── Pass 2 : err_lim (fine refinement) ───────────────────────────────────
    # All points in energy_grid are already cached from pass 1.
    # Only genuinely new midpoints trigger an actual XS evaluation.
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

    # Drop the last point — it will be the first point of the next window
    # (avoids duplicates at window boundaries after concatenation).
    # The caller re-appends the final point of the last window.
    return window_idx, energy_grid[:-1], xs_grid[:-1]


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
    # Assign each point to a window
    window_of = [
        int((math.sqrt(e) - math.sqrt(E_min)) / E_spacing)
        for e in point_grid
    ]
    max_w = window_of[-1]

    # Group indices by window
    windows: dict[int, List[int]] = {}
    for idx, w in enumerate(window_of):
        windows.setdefault(w, []).append(idx)

    slices: List[List[float]] = []
    for w in range(max_w + 1):
        if w not in windows:
            continue
        idxs = windows[w]
        pts  = [point_grid[i] for i in idxs]

        # Append the first point of the next non-empty window as right boundary
        for w_next in range(w + 1, max_w + 2):
            if w_next in windows:
                pts.append(point_grid[windows[w_next][0]])
                break
            elif w_next > max_w:
                # last window — right boundary is already in pts
                break

        if len(pts) >= 2:
            slices.append(pts)

    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_majorant_xs_grid(
    geometry,
    err_lim:     float = 0.001,
    err_max:     float = 0.01,
    err_int:     float | None = None,
    last_window: int   | None = None,
    last_energy: float | None = None,
    n_workers:   int   | None = None,
) -> Tuple[List[float], List[float], float, float, List[int]]:
    """
    Parallelised drop-in replacement for the serial build_majorant_xs_grid.

    New parameter
    -------------
    n_workers : number of parallel processes (default: os.cpu_count()).
                Set to 1 to run serially (useful for debugging).

    All other parameters and the return value are identical to the original.
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
        for nuclide_obj, density in mat.nuclides:
            nuclides[nuclide_obj.name] = nuclide_obj

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

    E_max = min(E_max_nuc, last_energy) if last_energy is not None else E_max_nuc
    n_windows = nuclides[minimum_spacing_nuclide].n_windows

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
    window_slices = _split_into_windows(point_grid, E_min, E_spacing)
    n_actual_windows = len(window_slices)
    print(f"[reconr_parallel] {n_actual_windows} non-empty windows to process")

    # ── Dispatch workers ──────────────────────────────────────────────────────
    results: dict[int, Tuple[List[float], List[float]]] = {}

    if n_workers == 1:
        # Serial fallback — useful for debugging / profiling
        _worker_init(geometry)
        for w_idx, pts in enumerate(window_slices):
            _, e_local, xs_local = _process_window(
                w_idx, pts, err_lim, err_max, err_int
            )
            results[w_idx] = (e_local, xs_local)
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
                w_idx, e_local, xs_local = future.result()
                results[w_idx] = (e_local, xs_local)
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
    energy_grid.append(window_slices[-1][-1])
    xs_grid.append(geometry.caculate_mat_majorant_xs(window_slices[-1][-1]))

    print(f"[reconr_parallel] merged grid: {len(energy_grid)} points")

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

    return e_grid_list, xs_grid_list, math.sqrt(E_min), E_spacing, window_pointers