#### HELPER FUNCTIONS FOR THE SQRT(T)-E ####
from math import sqrt, pi, erf, exp
import numpy as np
import re
import ast
import csv
import pandas as pd
import os

from openmc.data import K_BOLTZMANN
from scipy.special import wofz

from typing import List, Optional

"""
Function library :

LOGISTICS 
    assign_pole_to_window(multipole_data, pole_index) -> list of windows containing the pole
        - Assigns a given pole to its corresponding energy window based on the provided energy grid and pole energies.

    assign_energy_bounds_to_window(multipole_data, windows_list) -> start_e, end_e
        - Retrieves the energy bounds for a given list of windows from the multipole data.

    assign_energy_to_window(multipole_data, E) -> window_index
        - Determines which energy window a given energy E falls into based on the multipole data's
        
TEMPERATURE BOUND IDENTIFICATION
    calculate_majorant_temperature(E, coefficients, T_min, T_max) -> T_maj
        - Calculates the majorant temperature for a given energy E using the provided coefficients and temperature bounds


XS CALCULATION

    evaluate_one_pole_contribution(multipole_data, E, T, pole_index) -> xs total 
        - Evaluates the contribution of a single pole to the cross section at a given energy and temperature using the multipole data.
        - Exactly the same function as in mulipole file of openMC

    evaluate_curve_fit_contribution(multipole_data,E, T) -> xs total
        - Evaluates the contribution of the curve fit to the cross section at a given energy and 

    evaluate_majorant_cross_section(multipole_data, E, T) -> maj_xs
        - Evaluates the majorant cross section at a given energy and temperature by summing the

MATHEMATICS

    faddeeva(Z) -> w_val
        - Computes the Faddeeva function value for a given complex argument Z, which is used

    broaden_wmp_polynomials(multipole_data, E, dopp, n) -> broadened_polynomials
        - Computes the broadened polynomial contributions for a given energy E, Doppler broadening parameter  


TABLES LOADING

    parse_coeffs(coeffs_str)
        support function to load the coefficients from the csv file (stored as string)
    
    load_table_from_csv(file_path) -> data_table
        - Loads the coefficients for the majorant temperature calculation from a CSV file into a pandas Data
        """


## LOGISTICS FUNCTIONS
def assign_pole_windows(multipole_data, pole_index) :
    '''
    Docstring for assign_pole_windows
    
    :param multipole_data: multipole data object
    :param pole_index: indx of the pole to find windows for
    :return: list of windows that contain the pole
    '''
    pole_window = []
    for i_window in range(multipole_data.n_windows):
        # From openMC doc :
        # windows[i, 0] - 1 is the index of the first pole in window i. 
        # windows[i, 1] - 1 is the index of the last pole in window i.
        startw = multipole_data.windows[i_window, 0] - 1
        endw = multipole_data.windows[i_window, 1] - 1
        for i_pole in range(startw, endw + 1):
            if i_pole == pole_index:
                pole_window.append(i_window)
    return pole_window

def assign_energy_bounds_to_window(multipole_data, windows_list) :
    starte = (sqrt(multipole_data.E_min) + (np.min(windows_list)-1)* (multipole_data.spacing))**2
    ende = (sqrt(multipole_data.E_min) + (np.max(windows_list))* (multipole_data.spacing))**2
    return starte, ende

def assign_energy_to_window(multipole_data, E) :
    if E < multipole_data.E_min or E > multipole_data.E_max:
        raise ValueError(f"Energy {E} is out of bounds for the multipole data (E_min: {multipole_data.E_min}, E_max: {multipole_data.E_max})")
    window_index = min(multipole_data.n_windows - 1,
                       int(np.floor((np.sqrt(E) - np.sqrt(multipole_data.E_min)) / multipole_data.spacing)))
    return window_index

### MAJORANT TEMPERATURE IDENTIFICATION FUNCTION

def calculate_majorant_temperature(E, coefficients, T_min, T_max):
    sqrt_T_min = sqrt(T_min)
    sqrt_T_max = sqrt(T_max)

    valid_T = []

    for coeff in coefficients:
        if coeff is None or len(coeff) != 2:
            continue
        
        a, b = coeff
        sqrt_T = a * E + b
        
        if sqrt_T_min < sqrt_T < sqrt_T_max:
            valid_T.append(sqrt_T**2)

    if valid_T:
        return valid_T

    else:
        # no valid root → fallback
        return [T_min, T_max]


### XS CALCULATION FUNCTIONS
def evaluate_one_pole(multipole_data, E, T, pole_index):
        """Compute scattering, absorption, and fission cross sections contribution from an individual pole.

        Parameters
        ----------
        E : Real
            Energy of the incident neutron in eV.
        T : Real
            Temperature of the target in K.
        
        pole_index : int
            Index of the pole to evaluate.

        Returns
        -------
        3-tuple of Real
            Scattering, absorption, and fission microscopic cross sections contribution from the specified pole
            at the given energy and temperature.

        """
        
       
        if E < multipole_data.E_min: return (0, 0, 0)
        if E > multipole_data.E_max: return (0, 0, 0)

        # ======================================================================
        # Bookkeeping

        # Constants that determine which value to access
        _MP_EA = 0       # Pole

        # Residue indices
        _MP_RS = 1       #Residue scattering
        _MP_RA = 2       # Residue absorption
        _MP_RF = 3       # Residue fission

        # Polynomial fit indices
        _FIT_S = 0       # Scattering
        _FIT_A = 1       # Absorption
        _FIT_F = 2       # Fission

        # Define some frequently used variables.
        sqrtkT = sqrt(K_BOLTZMANN * T)
        sqrtE = sqrt(E)
        invE = 1.0 / E

        # Initialize the ouptut cross sections.
        sig_s = 0.0
        sig_a = 0.0
        sig_f = 0.0

        i_pole = pole_index
   
        
        if sqrtkT == 0.0:
            # If at 0K, use asymptotic form.
            
        
            psi_chi = -1j / (multipole_data.data[i_pole, _MP_EA] - sqrtE)
            c_temp = psi_chi / E
            sig_s += (multipole_data.data[i_pole, _MP_RS] * c_temp).real
            sig_a += (multipole_data.data[i_pole, _MP_RA] * c_temp).real

        else:
            # At temperature, use Faddeeva function-based form.
            dopp = multipole_data.sqrtAWR / sqrtkT
            
            
            Z = (sqrtE - multipole_data.data[i_pole, _MP_EA]) * dopp
            w_val = faddeeva(Z) * dopp * invE * sqrt(pi)
            sig_s += (multipole_data.data[i_pole, _MP_RS] * w_val).real
            sig_a += (multipole_data.data[i_pole, _MP_RA] * w_val).real
            
        return sig_s + sig_a

def evaluate_curve_fit_contribution(multipole_data,E, T) :
        """
        Evaluate the contribution of the curvefit to the majorant cross section at a given energy E.

        Parameters
        ----------
        multipole_data : openmc.data.WindowedMultipole
            The multipole data object containing the curvefit information.
        E : float
            Energy at which to evaluate the curvefit contribution.
        T : float
            Temperature at which to evaluate the curvefit contribution.
            

        Returns
        -------
        curvefit_contribution : float
            Contribution of the curvefit to the majorant cross section at energy E.
        """
        # find the window corresponding to the energy E and the poles associatied to this window
        i_window = min(multipole_data.n_windows - 1,
                       int(np.floor((np.sqrt(E) - np.sqrt(multipole_data.E_min)) / multipole_data.spacing)))
      
        # add the curvefit contribution of the window 
        # the maximum contribution of the curvefit is at maximal temperature so 3000K
        curvefit_contribution = 0.0 
        sqrtE = sqrt(E)
        invE = 1.0 / E      
        sqrtkT = sqrt(K_BOLTZMANN * T)
        curvefit_contribution_s = 0.0
        curvefit_contribution_a = 0.0
        if sqrtkT != 0 and multipole_data.broaden_poly[i_window]:
            #print("Broadened curvefit contribution evaluation")
            dopp = multipole_data.sqrtAWR/sqrtkT
            
            
            broadened_polynomials = broaden_wmp_polynomials(multipole_data, E, dopp,multipole_data.fit_order + 1)
            
            for i_poly in range(multipole_data.fit_order + 1):
                curvefit_contribution_s += float(multipole_data.curvefit[i_window, i_poly, 0] * broadened_polynomials[i_poly])
                
                #print(curvefit_contribution_s, flush=True)
                curvefit_contribution_a += float(multipole_data.curvefit[i_window, i_poly, 1] * broadened_polynomials[i_poly])
                #print(curvefit_contribution_a, flush=True)

        else :
            temp =  invE
            #print("Unbroadened curvefit contribution evaluation")
            for i_poly in range(multipole_data.fit_order + 1):
                #print("temp :", temp)
                curvefit_contribution_s += float(multipole_data.curvefit[i_window, i_poly, 0] * temp)
                #print(curvefit_contribution_s, flush=True)
                
                curvefit_contribution_a += float(multipole_data.curvefit[i_window, i_poly, 1] * temp)
                #print(curvefit_contribution_a, flush=True)
                
                temp *= sqrtE

        return (curvefit_contribution_s + curvefit_contribution_a)


def calculate_majorant_pole_contribution(E,coefficients, T_range, multipole_data, pole_index) :
    T_min, T_max = T_range
    Temperatures = calculate_majorant_temperature(E, coefficients, T_min, T_max)

    
    maj_contribution = - np.inf
    if isinstance(Temperatures, list):
        for T in Temperatures:
            maj_tot = evaluate_one_pole(multipole_data,E, T, pole_index)
            
            if maj_tot > maj_contribution:
                maj_contribution = maj_tot
               
            # check that max_contribution is a float
            if not isinstance(maj_contribution, (int, float)):
                raise ValueError(f"Expected maj_contribution to be a float, but got {type(maj_contribution)}")
        
    else:
        maj_contribution = evaluate_one_pole(multipole_data, E, Temperatures, pole_index)
        

    return maj_contribution


def evaluate_majorant_cross_section(multipole_data, E, data_table, T_range :Optional[tuple] = None) :
        """
        Evaluate the majorant cross section at a given energy E and temperature T.

        Parameters
        ----------
        multipole_data : openmc.data.WindowedMultipole
            The multipole data object containing the pole and curvefit information.
        E : float
            Energy at which to evaluate the majorant cross section.
        data_table : pandas.DataFrame
            The data table containing the pole and the coefficients in the format
            pole_index | coefficients |
        T_range : tuple, optional
            A tuple of (T_min, T_max) specifying the temperature range to consider for the

        Returns
        -------
        maj_xs : float
            The majorant cross section at energy E and temperature T.
        """
        maj_xs = 0.0

        if T_range is None:
            T_range = (293, 3000) # default range of temperature in K

        window_index = assign_energy_to_window(multipole_data, E)
        windows = multipole_data.windows
        startw = windows[window_index, 0] - 1
        endw = windows[window_index, 1] - 1
        #print(f"Energy {E} eV is in window {window_index}")
        
        # Add contributions from all poles
        for i_pole in range(startw, endw + 1):
            i_pole = int(i_pole)
            #check if the pole is in the data table
            if i_pole not in data_table['Pole Index'].values:
                continue
            coefficients = data_table.loc[data_table['Pole Index'] == i_pole, 'Coefficients'].values[0]

            maj_xs += calculate_majorant_pole_contribution(multipole_data = multipole_data, 
                                                           E = E, 
                                                           coefficients = coefficients, 
                                                           T_range = T_range, 
                                                           pole_index = i_pole)
        
        # Add contribution from the curvefit
        maj_xs += max(
                evaluate_curve_fit_contribution(multipole_data=multipole_data, E=E, T=T_range[1]),
                evaluate_curve_fit_contribution(multipole_data=multipole_data, E=E, T=T_range[0])
            )
        
        return maj_xs*1.002



### MATHEMATICS FUNCTIONS
def faddeeva(z):
    if np.angle(z) > 0:
        return wofz(z)
    else:
        return -np.conj(wofz(z.conjugate()))

def broaden_wmp_polynomials(multipole_data, E, dopp, n):
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


### LOAD TABLES

def parse_coeffs(s):

    if pd.isna(s) or s == "[]":
        return []
    try:
        # Replace array(...) with ...
        cleaned = re.sub(r'array\((.*?)\)', r'\1', s)
        parsed = ast.literal_eval(cleaned)
        return [x for x in parsed]
    except Exception:
        print("Parse error:", s)
        return None

def xs_majorant_tables(file_dir):
    tables ={}

    for dir in os.listdir(file_dir):
        nuclide_name = dir
        path = os.path.join(file_dir, dir, dir + '.csv')
        try:
            data_table = pd.read_csv(path)
            
            data_table['Coefficients'] = data_table['Coefficients'].apply(parse_coeffs)
            
        except Exception as e:
            print(f"Error loading table from {path}: {e}")
        tables[nuclide_name] = data_table
    return tables
        