import numpy as np
import pandas as pd
import openmc

def discrete_majorant(E : float, T_eval,  T_array : np.ndarray = None, nuclide=None):

        """
        Evaluate the majorant cross section at E over a temperature range and return maximum value
        
        Parameters
        ----------
        E : float
            Energy at which to evaluate the majorant cross section (in eV).
        T_array : array_like
            Array of temperatures at which to evaluate the majorant cross section (in K).
        nuclide : WMPNuclide
            WMPNuclide object containing the data for the nuclide of interest.

        Returns
        -------
        float
            Maximum value of the majorant cross section over the temperature range.
        """

        if T_array is None:
             raise ValueError("T_array must be provided to evaluate the majorant cross section.")
        if T_eval is None:
             raise ValueError("T_eval must be provided to evaluate the majorant cross section.")
        T_array = np.array(T_array)
        
        #find the two temperatures bracketing T_eval

        # check if T_eval is within T_array
        if T_eval in T_array:
            idx = np.where(T_array == T_eval)[0][0]
            xs = nuclide(E, T_eval)
            return xs[0] + xs[1]  # total cross section at T_eval
        
        # check if T_eval is outside the bounds of T_array and extrapolate if necessary
        if T_eval < T_array[0] or T_eval > T_array[-1]:
            print(f"Warning: T_eval={T_eval} K is outside the bounds of T_array. Extrapolating majorant cross section.")
            if T_eval < T_array[0]:
                T1, T2 = T_array[0], T_array[1]
            else:
                T1, T2 = T_array[-2], T_array[-1]
            
            xs1 = nuclide(E, T1)
            sigma1 = xs1[0] + xs1[1]
            xs2 = nuclide(E, T2)
            sigma2 = xs2[0] + xs2[1]
            max_majorant = np.interp(T_eval, [T1, T2], [sigma1, sigma2])
            return max_majorant
        
        idx = np.searchsorted(T_array, T_eval)
        

        T1 = T_array[idx - 1]
        T2 = T_array[idx]

        xs1 = nuclide(E, T1)
        sigma1 = xs1[0] + xs1[1]
        xs2 = nuclide(E, T2)
        sigma2 = xs2[0] + xs2[1]

        max_majorant = np.interp(T_eval, [T1, T2], [sigma1, sigma2])

        

        return max_majorant