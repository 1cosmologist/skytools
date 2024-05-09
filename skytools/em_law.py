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
"""
Module containing emission law functions. These produce the frequency scaling
of different emission components relevant to the CMB observations.
"""
import numpy as np
import scipy.constants as con

T_CMB = 2.72548 # K

def B_nu_T(nu_in_GHz, T_planck=T_CMB):
    """
    B_nu_T is the Planck distribution function, defined as: 
    .. math::
        B(\\nu, T) = \\frac{2 h \\nu^3}{c^2} \\frac{1}{e^{\\frac{h \\nu}{k_B T}} - 1}.
    
    It returns the Planck function values for nu (can be vectorized) 
    for a given blackbody temperature.

    Parameters
    ----------
    nu_in_GHz : float or numpy 1D array
        Frequency in GHz at which we want the value of the Planck function.
    T_planck : float, optional
        Temperature of the Planck distribution in Kelvin.
        Default value set to the CMB monopole temperature of 2.7255 K.

    Returns
    -------
    float or numpy 1D array
        A float value (or 1D array) for the value(s) of the Planck distribution.

    """

    nu_in_Hz = nu_in_GHz * con.giga
    x = con.h * nu_in_Hz / con.k / T_planck
    prefactor = 2. * con.h * nu_in_Hz**3. / con.c**2.

    return prefactor / (np.exp(x) - 1.)

def B_prime_nu_T(nu_in_GHz, T_planck=T_CMB):
    """
    B_prime_nu_T is the derivative Planck distribution function defined as:
    .. math::
        \\frac{dB(\\nu, T)}{dT} = \\frac{h \\nu}{k_B T^2} \\frac{e^{\\frac{h \\nu}{k_B T}}}{e^{\\frac{h \\nu}{k_B T}} - 1} B(\\nu, T).
    
    It returns the first derivative of the Planck function with respect to T, 
    for nu (can be vectorized) at a given blackbody temperature.

    Parameters
    ----------
    nu_in_GHz : float or numpy 1D array
        Frequency in GHz at which we want the value of the Planck function 
        derivative. 
    T_planck : float, optional
        Temperature of the Planck distribution in Kelvin.
        Default value set to the CMB monopole temperature of 2.7255 K.
    
    Returns
    -------
    float or numpy 1D array
        A float value (or 1D array) for the value(s) of the differential Planck distribution.

    """
    
    nu_in_Hz = nu_in_GHz * con.giga
    x = con.h * nu_in_Hz / con.k / T_planck
    prefactor = con.h * nu_in_Hz / con.k / T_planck**2.
    return prefactor * np.exp(x) / (np.exp(x) - 1.) * B_nu_T(nu_in_GHz)

def ysz_spectral_law(nu_in_GHz):
    """
    ysz_spectral_law is the SED function for Compton y parameter, defined as:
    .. math:: 
        y_{SZ} = \\frac{dB(\\nu, T)}{dT} \\left(\\frac{\\frac{h \\nu}{k T}}{\\tanh(\\frac{\\frac{h \\nu}{k T}}{2})} - 4\\right) T.
    It returns the frequency scaling of y_SZ for nu (can be vectorized).

    Parameters
    ----------
    nu_in_GHz : float or numpy 1D array
        Frequency in GHz at which we want the value of the y_SZ SED. 
    
    Notes
    ----- 
    Temperature is set to the CMB monopole temperature of 2.7255 K.

    Returns
    -------
    float or numpy 1D array
        A float value (or 1D array) for the value(s) of the y_SZ SED.

    """
    
    nu_in_Hz = nu_in_GHz * con.giga
    x = con.h * nu_in_Hz / con.k / T_CMB

    g_nu_part = x / np.tanh(x/2.) - 4.

    return B_prime_nu_T(nu_in_GHz) * g_nu_part * T_CMB

def greybody(nu_in_GHz, nu_ref_in_GHz, spec_ind, T_grey, flux_ref=1.):
    """
    greybody is the SED function for a greybody distribution, defined as:
    .. math:: 
        I_\\nu = A \\left(\\frac{\\nu}{\\nu_0}\\right)^{\\beta} \\frac{B(\\nu, T_{grey})}{B(\\nu_0, T_{grey})}.
    
    This function allows to set flux at reference frequency. If not used
    the output is just the frequency scaling of the greybody.
    
    Parameters
    ----------
    nu_in_GHz : float or numpy 1D array
        Frequency in GHz at which we want the value of the greybody function. 
    nu_ref_in_GHz : float
        Greybody reference frequency in GHz.
    spec_ind : float
        Spectral index of the greybody.
    T_grey : float
        Greybody temperature in Kelvin.
    flux_ref : float, optional
        Amplitude A of greybody emissions at reference frequency in arbitrary units. 
        Default value is 1 (returns frequency scaling).

    Returns
    -------
    float or numpy 1D array
        A float value (or 1D array) for the value(s) of the greybody SED.

    """
        
    # flux_ref sets the flux at reference frequency. If not set then takes default value of 1.
    # This would give just the frequency scaling of the greybody.

    powlaw = (nu_in_GHz / nu_ref_in_GHz)**(spec_ind)

    return flux_ref * powlaw * B_nu_T(nu_in_GHz, T_planck=T_grey) / B_nu_T(nu_ref_in_GHz, T_planck=T_grey)

def powerlaw(nu_in_GHz, nu_ref_in_GHz, spec_ind=1.):
    """
    powerlaw is the SED function for a powerlaw distribution, defined as:
    .. math:: 
        \\left(\\frac{\\nu}{\\nu_0}\\right)^\\beta.
    
    This function outputs the frequency scaling of a powerlaw distribution.

    Parameters
    ----------
    nu_in_GHz : float or numpy 1D array
        Frequency in GHz at which we want the value of the powerlaw function.
    nu_ref_in_GHz : float
        Powerlaw reference frequency in GHz.
    spec_ind : float, optional
        Spectral index of the powerlaw. Default value set to 1.
    
    Returns
    -------
    float or numpy 1D array
        A float value (or 1D array) for the value(s) of the powerlaw freqency 
        scaling.

    """
        
    return (nu_in_GHz / nu_ref_in_GHz)**spec_ind

def modified_blackbody(nu_in_GHz, spec_ind, T_bb):
    """
    modified_blackbody is the frequency scaling function for a modified blackbody 
    (MBB) distribution, defined as:
    .. math:: 
        \\nu^\\beta B(\\nu, T_{bb}).
    
    Comparing with greybody function, this provides modified blackbody frequency scaling 
    without a reference frequency. In principle it return only the numerator part of 
    the greybody function.

    Parameters
    ----------
    nu_in_GHz : float or numpy 1D array
        Frequency in GHz at which we want the value of the MBB function.
    spec_ind : float
        Spectral index of the MBB.
    T_bb : float
        MBB temperature in Kelvin.
    
    Returns
    -------
    float or numpy 1D array
        A float value (or 1D array) for the value(s) of the MBB frequency scaling.

    """
        
    return (nu_in_GHz * con.giga)**spec_ind * B_nu_T(nu_in_GHz, T_planck=T_bb)
