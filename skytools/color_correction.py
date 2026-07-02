#######################################################################
# This file is a part of SkyTools
#
# Sky Tools
# Copyright (C) 2026  Shamik Ghosh
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

import numpy as np
import scipy.constants as con 

from . import em_law as el

def mbb_color_correction(beta, T, frequency_in_GHz, transmission, central_frequency_GHz, iras_norm=False):
    """Compute the color-correction factor for a modified blackbody spectrum.

    Parameters
    ----------
    beta : float
        Spectral emissivity index.
    T : float
        Blackbody temperature in kelvin.
    frequency_in_GHz : array-like
        Frequency grid in GHz across the instrument bandpass.
    transmission : array-like
        Bandpass transmission sampled on ``frequency_in_GHz``.
    central_frequency_GHz : float
        Reference (central) frequency in GHz.
    iras_norm : bool, optional
        If ``True``, apply IRAS-style normalization.

    Returns
    -------
    float
        Multiplicative color-correction factor converting band-averaged
        response to the monochromatic response at ``central_frequency_GHz``.
    """
    norm_transmission = transmission / np.trapezoid(transmission, frequency_in_GHz * con.giga)
     
    sed_in_band = el.modified_blackbody(frequency_in_GHz, beta, T,)
    band_sed = np.trapezoid(sed_in_band * norm_transmission, frequency_in_GHz * con.giga)
    
    central_sed = el.modified_blackbody(central_frequency_GHz, beta, T)
    
    iras_norm_factor = 1.
    if iras_norm:
        iras_norm_factor = np.trapezoid((central_frequency_GHz/frequency_in_GHz)*norm_transmission, frequency_in_GHz*con.giga)
    
    return iras_norm_factor * central_sed / band_sed


def powerlaw_color_correction(beta, frequency_in_GHz, transmission, central_frequency_GHz, iras_norm=False):
    """Compute the color-correction factor for a power-law spectrum.

    Parameters
    ----------
    beta : float
        Spectral index of the power law.
    frequency_in_GHz : array-like
        Frequency grid in GHz across the instrument bandpass.
    transmission : array-like
        Bandpass transmission sampled on ``frequency_in_GHz``.
    central_frequency_GHz : float
        Reference (central) frequency in GHz.
    iras_norm : bool, optional
        If ``True``, apply IRAS-style normalization.

    Returns
    -------
    float
        Multiplicative color-correction factor converting band-averaged
        response to the monochromatic response at ``central_frequency_GHz``.
    """
    norm_transmission = transmission / np.trapezoid(transmission, frequency_in_GHz * con.giga)
     
    sed_in_band = (frequency_in_GHz * con.giga)**beta
    band_sed = np.trapezoid(sed_in_band * norm_transmission, frequency_in_GHz * con.giga)
    
    central_sed = (central_frequency_GHz * con.giga)**beta
    
    iras_norm_factor = 1.
    if iras_norm:
        iras_norm_factor = np.trapezoid((central_frequency_GHz/frequency_in_GHz)*norm_transmission, frequency_in_GHz*con.giga)
    
    return iras_norm_factor * central_sed / band_sed