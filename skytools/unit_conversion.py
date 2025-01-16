#######################################################################
# This file is a part of SkyTools
#
# Sky Tools
# Copyright (C) 2023  Shamik Ghosh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information about SkyTools please visit 
# <https://github.com/1cosmologist/skytools> or contact Shamik Ghosh 
# at shamik@lbl.gov
#
#########################################################################

########################################################
#
#   UNIT CONVERSION CODE
#
#   Cross checked for HFI with few percent error.
#
#   Reference: Planck 2013 IX: HFI Spectral Response
#   
#   version 5: Shamik Ghosh, 2023-03-29 
#
########################################################## 
"""
This module computes unit conversion and color correction factors for CMB observations
based on Planck 2013 IX: HFI Spectral Response. This is meant to be a python equivalent
to the Planck UC_CC IDL codes.
"""
import numpy as np
import scipy.constants as con

from . import em_law as el 

__pdoc__ = {}

T_CMB = 2.7255 # K

def MJysr_to_Kb(nuc_in_GHz):
    """
    Gives conversion factor from MJy/sr to brightness temperature Kb (K_RJ).

    Parameters
    ----------
    nuc_in_GHz: float
                A float denoting the central frequency of the band. No bandpass information needed.
    
    Notes
    -----
    No bandpass is required only central frequency is assumed.

    Returns
    -------
    float
        A float value to convert MJy/sr to Kb.
    """

    return con.c**2. / 2. / (nuc_in_GHz * con.giga)**2. / con.k / 1.e20    # 1e20 is conversion factor from SI unit of emissivity to MJy 1e-6 x 1e26 = 1e20


def KCMB_to_MJysr(nus_in_GHz, nuc_in_GHz=None, transmission=None):
    """
    Gives conversion factor from K_CMB to MJy/sr.
    MJy/sr assumes nu^-1 (IRAS) reference spectrum following Planck. 

    Parameters
    ----------
    nus_in_GHz: float or np.ndarray
                If single float value is provided, assumed to be delta transmission.
                If np.ndarray is provided without transmission, assume tophat transmission.

    nuc_in_GHz: float, default=None
                If nus_in_GHz is a single number, then nuc_in_GHz is ignored, and is assumed to be the same.
                If nus_in_GHz is a np.ndarray, and nuc_in_GHz is not provided, we assume transmission weighted 
                average of nus_in_GHz.
    transmission: np.ndarray, default=None
                Must be the same size as nus_in_GHz. Need not be normalized to one. Assume HFI/LFI definition
                of transmission, assuming $$\\lambda^2$$ factor is multipled and the transmission is in units of
                MJy/sr. If your bandpass is in $$K_b$$ unit then the $$\\lambda^2$$ factor is missing.

    Returns
    -------
    float
        A float value to convert K_CMB to MJy/sr.
    """
    if not isinstance(nus_in_GHz, (list, np.ndarray)):
        return el.B_prime_nu_T(nus_in_GHz) * 1.e20      # 1.e20 factor converts W/m2/Hz to MJy.

    if isinstance(transmission, (list,np.ndarray)):
        if len(transmission) != len(nus_in_GHz):
            raise Exception("ERROR: transmission and frequency arrays are not of same size.")

    else:

        transmission = np.ones(nus_in_GHz.shape) 

    weights = transmission / np.trapz(transmission, x=nus_in_GHz * con.giga)

    if nuc_in_GHz == None:
        nuc_in_GHz = np.trapz(nus_in_GHz*weights, x=nus_in_GHz * con.giga)

    band_integrated_CMB = np.trapz(weights*el.B_prime_nu_T(nus_in_GHz), x=nus_in_GHz * con.giga)
    band_integrated_nucbynu = np.trapz(weights*(nuc_in_GHz / nus_in_GHz), x=nus_in_GHz * con.giga)

    return (band_integrated_CMB / band_integrated_nucbynu) * 1.e20  # 1.e20 factor converts W/m2/Hz to MJy.


def KCMB_to_ySZ(nus_in_GHz, transmission=None):
    """
    Computes conversion factor from K_CMB to y SZ (Compton parameter). 

    Parameters
    ----------
    nus_in_GHz: float or np.ndarray
                If single float value is provided, assumed to be delta transmission.
                If np.ndarray is provided without transmission, assume tophat transmission.

    transmission: np.ndarray, default=None
                Must be the same size as nus_in_GHz. Need not be normalized to one. Assume HFI/LFI definition
                of transmission, assuming \lambda^2 factor is multipled and the transmission is in units of
                MJy/sr. If your bandpass is in K_b unit then the \lambda^2 factor is missing.

    Returns
    -------
    float
        A float value that is the conversion factor from K_CMB to y SZ.
    """

    if not isinstance(nus_in_GHz, (list, np.ndarray)):
        return el.B_prime_nu_T(nus_in_GHz) / el.ysz_spectral_law(nus_in_GHz)     # 1.e20 factor converts W/m2/Hz to MJy.

    if isinstance(transmission, (list,np.ndarray)):
        if len(transmission) != len(nus_in_GHz):
            raise Exception("ERROR: transmission and frequency arrays are not of same size.")

    else:

        transmission = np.ones(nus_in_GHz.shape) 

    weights = transmission / np.trapz(transmission, x=nus_in_GHz * con.giga)

    band_integrated_CMB = np.trapz(weights*el.B_prime_nu_T(nus_in_GHz), x=nus_in_GHz * con.giga)
    band_integrated_ysz = np.trapz(weights*el.ysz_spectral_law(nus_in_GHz), x=nus_in_GHz * con.giga)

# Referenece: Eq 33 from Planck 2013 XI HFI spectral response  
    return band_integrated_CMB/band_integrated_ysz  # returns in K_CMB^(-1) 