import numpy as np
import em_law as el 
import scipy.constants as con

def color_correction(em_law_type, nus_in_GHz, transmission = None, *args, **kwargs):
    """
    Returns the color correction the em_law sub-module.
    
    Parameters
    ----------
    em_law_type: function
        Emission law from the em_law sub-module
    nus_in_GHz : float or numpy ndarray
        Frequency in GHz at which we want to integrate the emission law em_law_type over a passband.
    transmission : numpy ndarray, default=None
        Must be the same size as nus_in_GHz. Need not be normalized to one. Assume HFI/LFI definition of transmission, assuming \(\\lambda^2\) factor is multipled and the transmission is in units of MJy/sr. If your bandpass is in \(K_b\) unit then the \(\\lambda^2\) factor is missing.
        
    *args: Positional arguments to pass to the em_law_type function.
    **kwargs: Keyword arguments to pass to the em_law_type function.

    Returns
    -------
    numpy ndarray
        A numpy ndarray for the value of the color correction for an emission law.

    """
    nus_in_Hz = nus_in_GHz * con.giga
    if not isinstance(nus_in_GHz, (list, np.ndarray)):
        return None
    if isinstance(transmission, (list, np.ndarray)):
        if len(transmission) != len(nus_in_GHz):
            raise Exception("ERROR: transmission and frequency arrays are not of same size.")
    else:
        transmission = np.ones(nus_in_GHz.shape)
    weights = transmission / np.trapz(transmission, x = nus_in_Hz)
    num_cc = np.trapz(weights * em_law_type(*args, **kwargs), x = nus_in_Hz)
    deno_cc = em_law_type(*args, **kwargs) * np.trapz(weights, x = nus_in_Hz)
    return num_cc/deno_cc