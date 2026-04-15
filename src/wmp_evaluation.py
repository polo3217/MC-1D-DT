import numpy as np
import pandas as pd
import openmc

def wmp_majorant(E : float, T_array : np.ndarray = None, nuclide=None):

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
        
        max_majorant = 0.0
        for T in T_array:
            majorant = nuclide(E,T)[0]+ nuclide(E,T)[1] 
            if majorant > max_majorant:
                max_majorant = majorant

        return max_majorant