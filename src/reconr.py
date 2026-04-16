import numpy as np
import pandas as pd
import openmc
import os
import sys

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
    print(n_windows)
    print(E_max)
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

    print(f" Evaluating the majorant cross section with RECONR stacking algorithm")
    print(f"err_lim = {err_lim}, err_max = {err_max}, err_int = {err_int}")
    print(f"last energy to add = {last_energy_to_add} eV")
    print(f"number of windows = {n_windows}")
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
            point_grid = np.insert(point_grid, i_grid + 1, e_half)
        else:
            energy_grid.append(e_last)
            cross_section_grid.append(sigma_total)
            i_grid += 1
        
        
        

        

    # add the last point
    energy_grid.append(last_energy_to_add)
    cross_section_grid.append(geometry.caculate_mat_majorant_xs(last_energy_to_add))
    print("done")

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



    table = np.column_stack((energy_grid, cross_section_grid))

    return energy_grid, cross_section_grid
    

    
    
