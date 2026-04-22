import numpy as np
import pandas as pd
import openmc
import os
import sys
import math
import bisect
sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')


def build_majorant_xs_grid(
        geometry,
        err_lim = 0.001,
        err_max = 0.01,
        err_int = None,
        last_window = None,
        last_energy = None,
        ):
    
    
    # get all the nuclides in the materials or maj_mat
    if geometry.maj_mat_method == "maj_mat":
        materials = [geometry.maj_mat]
    elif geometry.maj_mat_method == "simple":
        materials = geometry.materials

    
    nuclides = {}



    for mat in materials:
        for nuclide_obj, density in mat.nuclides:
            nuclides[nuclide_obj.name] = nuclide_obj  # unique by name

    nuclide_list = list(nuclides.values())

    
    # find E_min, E_max, n_windows and E_spacing
    E_min = -np.inf
    E_max = np.inf
    n_windows = 0
    E_spacing = np.inf
    
    for nuclide in nuclide_list:
        if nuclide.E_min > E_min:
            E_min = nuclide.E_min
        if nuclide.E_max < E_max:
            E_max = nuclide.E_max
            nuclide_max_E = nuclide.name

        if nuclide.spacing <= E_spacing:
            E_spacing = nuclide.spacing
            minimum_spacing_nuclide = nuclide.name

    
    E_max = min(E_max, last_energy) if last_energy is not None else E_max
    n_windows = nuclides[minimum_spacing_nuclide].n_windows


    # create an array of the size of n_windows
    point_grid = []
    # [CHANGED] Debug prints formatted professionally and wrapped in verbose check
    if geometry.verbose:
        print(f"  [RECONR Debug] Target n_windows: {n_windows}")
        print(f"  [RECONR Debug] Maximum Energy (E_max): {E_max:.2e} eV")
    for i in range(n_windows):
        E_grid = (np.sqrt(E_min) + (i)* (E_spacing))**2
        if E_grid > E_max:
            break
        point_grid.append(E_grid)

    # now implement the stacking algorithm from RECONR
    # for now just for the total cross section of u238
    energy_grid = []
    cross_section_grid = []

    e_last = point_grid[0]
    i_grid = 0
    if err_int is None:
        err_int = err_lim / 20000
    last_window = len(point_grid)
    last_energy_to_add = point_grid[-1]

    # [CHANGED] Clean professional print for RECONR start
    print("[RECONR] Evaluating majorant cross section (Stacking Algorithm)")
    if geometry.verbose:
        print(f"  -> err_lim: {err_lim}, err_max: {err_max}, err_int: {err_int}")
        print(f"  -> Last energy to add: {last_energy_to_add:.2e} eV")
        print(f"  -> Number of windows:  {n_windows}")
    e_next = point_grid[i_grid]

    ## err_max (first rough calculation)
    while e_next < last_energy_to_add:
        convergence_flag = False
        #evaluate the cross section at i_grid, i_grid + 1 and i_grid + 1/2 spacing
        e_last = point_grid[i_grid]
        e_next = point_grid[i_grid + 1]
        sigma_total = geometry.caculate_mat_majorant_xs(e_last)
        sigma_total_next = geometry.caculate_mat_majorant_xs(e_next)

        ## additional check from RECONR
        # roudinng error check

        #1. truncate e_half to 7 digits
        e_half = (e_last + e_next) / 2
        #print(f"e_half before truncation = {e_half} eV")
        e_half_truncated = float(f"{e_half:.7e}")
        e_last_truncated = float(f"{e_last:.7e}")
        e_next_truncated = float(f"{e_next:.7e}")
        #print(f"e_half after truncation to 7 digits = {e_half_truncated} eV")

        #2. if e_half_truncated is equal to e_last or e_next, then truncate e_half to 9 digits
        if e_half_truncated == e_last_truncated or e_half_truncated == e_next_truncated:
            e_half_truncated = float(f"{e_half:.9e}")
            e_last_truncated = float(f"{e_last:.9e}")
            e_next_truncated = float(f"{e_next:.9e}")
            #3. if e_half_truncated is still equal to e_last or e_next, then set the convergence flag to True
            if e_half_truncated == e_last_truncated or e_half_truncated == e_next_truncated:
                convergence_flag = True
            


        sigma_total_half = geometry.caculate_mat_majorant_xs(e_half_truncated)



        sigma_total_interp = (sigma_total + sigma_total_next) / 2

        err = abs(sigma_total_half - sigma_total_interp) / sigma_total_half

        if err > err_max and not convergence_flag:
            point_grid = np.insert(point_grid, i_grid + 1, e_half_truncated)
        else:
            energy_grid.append(e_last)
            cross_section_grid.append(sigma_total)
            i_grid += 1
        
        
        

        

    # add the last point
    energy_grid.append(last_energy_to_add)
    cross_section_grid.append(geometry.caculate_mat_majorant_xs(last_energy_to_add))
    # [CHANGED] Professional progress update, restricted to verbose
    if geometry.verbose:
        print("  [RECONR] First pass completed.")

    # RECONR second additional check with errmax
    # avoid to have too many points in high energy thin cross section
    # contribution error should less than 0.5x dsigma x dE
    # refine if both conditions are not met :
    #  1. err > err_lim
    #  2. 0.5x dsigma x dE > err_int


    i_grid = 0
    last_energy_to_add = energy_grid[-1]
    convergence_flag = False
    e_next = 0
    while e_next < last_energy_to_add:
        e = energy_grid[i_grid]
        sigma = cross_section_grid[i_grid]
        if e > last_energy_to_add:
            convergence_flag = True
        e_next = energy_grid[i_grid + 1]
        e_half = (e + e_next) / 2
        e_half_truncated = float(f"{e_half:.7e}")
        e_last_truncated = float(f"{e:.7e}")
        e_next_truncated = float(f"{e_next:.7e}")

        if e_half_truncated == e_last_truncated or e_half_truncated == e_next_truncated:
            e_half_truncated = float(f"{e_half:.9e}")
            e_last_truncated = float(f"{e:.9e}")
            e_next_truncated = float(f"{e_next:.9e}")
            if e_half_truncated == e_last_truncated or e_half_truncated == e_next_truncated:
                convergence_flag = True
                
        sigma_next = cross_section_grid[i_grid + 1]
        sigma_half = geometry.caculate_mat_majorant_xs(e_half_truncated)
        sigma_interp = (sigma + sigma_next) / 2
        if sigma_half == 0:
            err = 0
        else:
            err = abs(sigma_half - sigma_interp) / sigma_half

        
        if err > err_lim and 0.5 * abs(sigma_half - sigma_interp) * (e_next - e) > err_int and not convergence_flag:
            #print("ok")
            energy_grid.insert(i_grid + 1, e_half)
            cross_section_grid.insert(i_grid + 1, sigma_half)

        else:
            i_grid += 1



    # [CHANGED] Professional progress update, restricted to verbose
    if geometry.verbose:
        print(f"  [RECONR] Second pass completed — {len(energy_grid)} points generated.")

    # ── STEP 3: Deduplicate + window pointers ─────────────────────────────────
    e_arr  = np.array(energy_grid)
    xs_arr = np.array(cross_section_grid)

    diffs = np.diff(e_arr)
    
    # [CHANGED] Reformatted warnings and status updates to be cleaner and verbose-dependent
    if geometry.verbose:
        if np.any(diffs <= 0):
            print(f"  [WARNING] {(diffs <= 0).sum()} inversions found in energy grid before dedup")
            print(f"  [WARNING] Worst inversion: {diffs[diffs <= 0].min():.6e} eV")
        else:
            print("  [RECONR] No inversions found in energy grid before deduplication.")

    mask   = np.concatenate(([True], np.diff(e_arr) > 0))
    e_arr  = e_arr[mask]
    xs_arr = xs_arr[mask]
    # [CHANGED] Formatted output
    if geometry.verbose:
        print(f"  [RECONR] Deduplication removed {(~mask).sum()} points.")

    e_grid_list  = e_arr.tolist()
    xs_grid_list = xs_arr.tolist()

    # Build O(1) window pointers
    # [CHANGED] Standardized log format
    print("[RECONR] Building O(1) Window Pointers...")
    window_pointers  = []
    current_window   = 0

    #the window pointer is the index of the first point in the next window
    window_pointers.append(0)
    for idx, e in enumerate(e_grid_list):
        w = int((math.sqrt(e) - np.sqrt(E_min)) / E_spacing)
        while w > current_window:
            window_pointers.append(idx)
            current_window += 1


    

    window_pointers.append(len(e_grid_list))
    # [CHANGED] Cleaned up final output dump to be readable and verbose-only
    if geometry.verbose:
        print(f"  [RECONR] Window pointer table: {len(window_pointers) - 1} windows")
        print(f"  [RECONR] Final grid size:      {len(e_grid_list)} points")
        print(f"  [RECONR] Energy at first window pointer: {e_grid_list[window_pointers[0]]:.6e} eV")
        print(f"  [RECONR] First Energy Grid:              {e_grid_list[0]:.6e} eV")
        print(f"  [RECONR] Energy at last window pointer:  {e_grid_list[window_pointers[-1]-1]:.6e} eV")
        print(f"  [RECONR] Last Energy Grid:               {e_grid_list[-1]:.6e} eV")

    return e_grid_list, xs_grid_list, np.sqrt(E_min), E_spacing, window_pointers