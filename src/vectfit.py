import sys
import os

sys.path.append('/home/paule/open_mc_projects/windowed_multipole/02_working_notebook_vectfit')

import numpy as np
import pandas as pd
import majorant_multipole as maj

from math import exp, erf, pi, sqrt
from scipy.special import wofz
from openmc.data.data import K_BOLTZMANN


##### LOAD THE MAJORANT DATA FOR THE VECTFIT METHOD #####

def xs_majorant_tables(file_dir: str, verbose: bool = False) -> dict:
    tables = {}
    
    #loop through all the files contained in the file_dir
    #with the title of each file being the name of the nuclide.
    #the output table is a dictionary with the nuclide name as the key and the table as the value.

    for dir in os.listdir(file_dir):
        nuclide_name = dir
        if verbose:
            print(f"Loading table for nuclide: {nuclide_name}")
        path = os.path.join(file_dir, dir, "optimized_poles_residues.csv")
        
        table = pd.read_csv(path)
        table["Optimized Pole"]    = table["Optimized Pole"].str.strip("()").astype(complex)
        table["Optimized Residue"] = table["Optimized Residue"].str.strip("()").astype(complex)
        tables[nuclide_name] = table
    
    return tables 



#### HELPER FUNCTION FOR THE EVALUATION OF THE MAJORANT CROSS SECTION #####

# 1 Evaluation of the faadeva function
def _faddeeva_vect(z):
    """
    Vectorized evaluation of the Faddeeva function:
    
        w(z) = i/pi * ∫_{-∞}^{∞} exp(-t^2) / (z - t) dt
    
    Equivalent to scipy.special.wofz with proper branch handling.
    
    Parameters
    ----------
    z : complex or array_like of complex
        Input values (can be scalar or array)
    
    Returns
    -------
    complex or ndarray of complex
        Faddeeva function evaluated at each z
    """
    z = np.asarray(z, dtype=np.complex128)  # ensure array

    # Create an output array
    w = np.empty_like(z, dtype=np.complex128)

    # Branch depending on Im(z) > 0
    mask = np.angle(z) > 0
    w[mask] = wofz(z[mask])
    w[~mask] = -np.conj(wofz(np.conj(z[~mask])))

    # If input was scalar, return scalar
    if np.isscalar(z):
        return w[0]
    return w

# 2 Doppler broadening of the majorant poles and residues 
def doppler_broadening_vect(E, T, p, r, sqrtAWR):
    """
    Vectorized Doppler broadening over energy array E.
    Returns real part of r * Faddeeva argument.
    """

    E = np.atleast_1d(E)  # Ensure E is array
    sqrtE = np.sqrt(E)

    # Boltzmann factor
    sqrtkT = np.sqrt(T * K_BOLTZMANN)

    # Handle T=0 separately to avoid division by zero
    if sqrtkT == 0:
        psi_chi = -1j / (p - sqrtE)
        c_temp = psi_chi / E
        return (r * c_temp).real

    # Doppler width
    dopp = sqrtAWR / sqrtkT  # scalar

    # Compute z array
    z = dopp * (sqrtE - p)  # array

    # Faddeeva function for all z
    w_val = _faddeeva_vect(z) * dopp / E * np.sqrt(pi)  # array

    # Multiply residue and take real part
    result = (r * w_val).real

    return result


# 3 Doppler broadening of the curvefit polynomials
def _broaden_wmp_polynomials(nuclide, E, dopp, n):
    r"""Evaluate Doppler-broadened windowed multipole curvefit.

    The curvefit is a polynomial of the form :math:`\frac{a}{E}
    + \frac{b}{\sqrt{E}} + c + d \sqrt{E} + \ldots`

    Parameters
    ----------
    E : float
        Energy to evaluate at.
    dopp : float
        sqrt(atomic weight ratio / kT) in units of eV.
    n : int
        Number of components to the polynomial.

    Returns
    -------
    np.ndarray
        The value of each Doppler-broadened curvefit polynomial term.

    """
    sqrtE = sqrt(E)
    beta = sqrtE * dopp
    half_inv_dopp2 = 0.5 / dopp**2
    quarter_inv_dopp4 = half_inv_dopp2**2

    if beta > 6.0:
        # Save time, ERF(6) is 1 to machine precision.
        # beta/sqrtpi*exp(-beta**2) is also approximately 1 machine epsilon.
        erf_beta = 1.0
        exp_m_beta2 = 0.0
    else:
        erf_beta = erf(beta)
        exp_m_beta2 = exp(-beta**2)

    # Assume that, for sure, we'll use a second order (1/E, 1/V, const)
    # fit, and no less.

    factors = np.zeros(n)

    factors[0] = erf_beta / E
    factors[1] = 1.0 / sqrtE
    factors[2] = (factors[0] * (half_inv_dopp2 + E)
                + exp_m_beta2 / (beta * sqrt(pi)))

    # Perform recursive broadening of high order components. range(1, n-2)
    # replaces a do i = 1, n-3.  All indices are reduced by one due to the
    # 1-based vs. 0-based indexing.
    for i in range(1, n-2):
        if i != 1:
            factors[i+2] = (-factors[i-2] * (i - 1.0) * i * quarter_inv_dopp4
                + factors[i] * (E + (1.0 + 2.0 * i) * half_inv_dopp2))
        else:
            factors[i+2] = factors[i]*(E + (1.0 + 2.0 * i) * half_inv_dopp2)
    #print("Broadened polynomial factors:", factors)
    return factors


# 4 Evaluation of the curvefit contribution to the majorant cross section at a given energy E and temperature T
def evaluate_curvefit_contribution(nuclide, E, T):
        """
        Evaluate the contribution of the curvefit to the majorant cross section at a given energy E.

        Parameters
        ----------
        E : float
            Energy at which to evaluate the curvefit contribution.
        T : float
            Temperature at which to evaluate the curvefit contribution.
            if T == max --> get max T from self.T_max_curvefit



        Returns
        -------
        curvefit_contribution : float
            Contribution of the curvefit to the majorant cross section at energy E.
        """
        # find the window corresponding to the energy E and the poles associatied to this window
        i_window = min(nuclide.n_windows - 1,
                       int(np.floor((np.sqrt(E) - np.sqrt(nuclide.E_min)) /  nuclide.spacing)))
        if T== "max" :
            T_1 = 293
            
            T_2 = 3000
            range_T = [T_1, T_2]
        else :
            range_T = [T]
            
            
        
        # add the curvefit contribution of the window 
        # the maximum contribution of the curvefit is at maximal temperature so 3000K
        curvefit_contribution = 0.0 
        
        for T_to_eval in range_T:
            sqrtE = sqrt(E)
            invE = 1.0 / E      
            sqrtkT = sqrt(K_BOLTZMANN*T_to_eval)
       
            curvefit_contribution_s = 0.0
            curvefit_contribution_a = 0.0
            broaden = True
            
            if sqrtkT != 0 and nuclide.broaden_poly[i_window] and broaden == True:
                #print("Broadened curvefit contribution evaluation for E =", E, "eV and T =", T_to_eval, "K")
                dopp = nuclide.sqrtAWR/sqrtkT
                #print("Doppler width (dopp) for T =", T_to_eval, "K:", dopp)
                
                
                broadened_polynomials = _broaden_wmp_polynomials(nuclide, E, dopp, nuclide.fit_order + 1)
                
                for i_poly in range(nuclide.fit_order + 1):
                    curvefit_contribution_s += float(nuclide.curvefit[i_window, i_poly, 0] * broadened_polynomials[i_poly])
                    
                    #print(curvefit_contribution_s, flush=True)
                    curvefit_contribution_a += float(nuclide.curvefit[i_window, i_poly, 1] * broadened_polynomials[i_poly])
                    #print(curvefit_contribution_a, flush=True)

            else :
                temp =  invE
                
                #print("Unbroadened curvefit contribution evaluation")
                for i_poly in range(nuclide.fit_order + 1):
                    #print("temp :", temp)
                    curvefit_contribution_s += float(nuclide.curvefit[i_window, i_poly, 0] * temp)
                    #print(curvefit_contribution_s, flush=True)
                    
                    curvefit_contribution_a += float(nuclide.curvefit[i_window, i_poly, 1] * temp)
                    #print(curvefit_contribution_a, flush=True)
                    
                    temp *= sqrtE
            curvefit_contribution_T = curvefit_contribution_s + curvefit_contribution_a
            
            if curvefit_contribution_T > curvefit_contribution:
                #print(f"New max curvefit contribution at T={T_to_eval}K: {curvefit_contribution_T} > {curvefit_contribution}")
                curvefit_contribution = curvefit_contribution_T
                T_max = T_to_eval
                #print(f"New max curvefit contribution at T={T_to_eval}K: {curvefit_contribution_T} > {curvefit_contribution}")

       

        return curvefit_contribution

#### Evaluation of the majorant cross section with the vectfit method ############
def evaluate_sig_maj(nuclide, Energy, maj_table) :
    
    

 

    i_window = min(nuclide.n_windows - 1,
                       int(np.floor((np.sqrt(Energy) - np.sqrt(nuclide.E_min)) / nuclide.spacing)))
    startw = nuclide.windows[i_window, 0] - 1
    endw = nuclide.windows[i_window, 1]

    total_contribution = 0.0

    for pole in range(startw, endw):
        p = maj_table.loc[pole, 'Optimized Pole']
        r = maj_table.loc[pole, 'Optimized Residue']
        t = maj_table.loc[pole, 'Temperature']
        
        total_contribution += doppler_broadening_vect(Energy, t, p, r, nuclide.sqrtAWR)

    curvefit = evaluate_curvefit_contribution(nuclide, Energy, "max")
    
    # print the type of the total_contribution variable
    
    total_contribution += curvefit 
    
    return float(total_contribution)




    
