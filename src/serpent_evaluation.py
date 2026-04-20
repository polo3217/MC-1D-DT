import numpy as np
import pandas as pd
import openmc

def serpent_majorant(E : float, T_eval,  T_array : np.ndarray = None, nuclide=None):

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
        idx = np.searchsorted(T_array, T_eval)
        T1 = T_array[idx - 1]
        T2 = T_array[idx]

        xs1 = nuclide(E, T1)
        sigma1 = xs1[0] + xs1[1]
        xs2 = nuclide(E, T2)
        sigma2 = xs2[0] + xs2[1]

        max_majorant = np.interp(T_eval, [T1, T2], [sigma1, sigma2])

        

        return max_majorant