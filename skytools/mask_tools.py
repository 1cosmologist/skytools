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
SkyTools mask tools library provides functions and tools to process masks.
"""

import numpy as np
import healpy as hp
import multiprocessing as mp 
import joblib as jl 
import os 

from . import hpx_utils as hu
from . import border_finder as bf

datapath = os.getenv('SKYTOOLS_DATA')

def maskdist(mask_in, dist_from_edge_in_rad):
    """
    Calculate the distance map from the edges of masked regions in a HEALPix map.

    This function computes the distance of each valid pixel from the nearest edge of a masked region
    in the input HEALPix map. The distance is calculated in radians and is limited to the specified
    maximum distance from the edge.

    Parameters
    ----------
    mask_in : numpy.ndarray
        A 1D numpy array representing the input HEALPix mask, where masked regions are indicated by 0s.
    dist_from_edge_in_rad : float
        The maximum distance from the edge, in radians, up to which the distances are calculated.

    Returns
    -------
    numpy.ndarray
        A 1D numpy array with the same shape as ``mask_in``, containing the distance from the nearest edge
        for each valid pixel. Pixels beyond the specified distance are assigned the value ``hp.UNSEEN``.
    """

    n_cores = mp.cpu_count()
    nside = hp.get_nside(mask_in)
    
    border0_pix, border1_pix, np0, np1 = bf.get_mask_border(mask_in, need_nos=True)
    del border1_pix

    if np0 == 0:
        raise Exception("ERROR: No masked pixels in the map! Abort!", np0)
              
    border_slice = 2500
    nparts = np0 // border_slice

    if nparts == 0:
        nparts = 1
        npix_per_part = np.ones((nparts,)) * np0
    else:
        if np.mod(np0, nparts) > 0.:
            nparts += 1
            
        npix_per_part = np.zeros((nparts,))
        npix_per_part[:nparts-1] = border_slice

        npix_per_part[nparts-1] = np0 - np.sum(npix_per_part)

    if np.sum(npix_per_part) != np0:
        raise Exception("ERROR: npix_per_part sum does not match npix_0", np.sum(npix_per_part), np0)

    start = int(0)
    stop = int(start+npix_per_part[0])

    dist_map = np.copy(mask_in) * 6*np.pi
    for part in range(nparts):

        # bordermap = np.zeros_like(mask_in)
        # bordermap[border0_pix[start:stop]] = 1.

        # bordermap = hp.smoothing(bordermap, fwhm=(np.sqrt(2.) * dist_from_edge_in_rad),use_weights=True, datapath=datapath)
        # valid_pix = np.where((mask_in == 1) & (bordermap > 5e-4))[0]

        # del bordermap

        vec_bdr = np.array(hp.pix2vec(nside, border0_pix[start:stop]))
        neighbour_list = jl.Parallel(n_jobs=n_cores)(jl.delayed(hp.query_disc)(nside, vec_bdr[:, ivec], 1.05* dist_from_edge_in_rad, inclusive=True) for ivec in np.arange(stop-start, dtype=int))
        valid_pix = np.unique(np.hstack(np.array(neighbour_list, dtype=object)).astype(np.int64))
        valid_pix = valid_pix[valid_pix > 0.]
        valid_pix = valid_pix[mask_in[valid_pix] > 0.] 
        # print(len(valid_pix), npix_per_part[part])

        dist4valid = np.amin(hu.angdist(nside, valid_pix, nside, border0_pix[start:stop]), axis=1)   
        pix2save = np.where(dist4valid < dist_map[valid_pix])[0]
        dist_map[valid_pix[pix2save]] = dist4valid[pix2save]

        if part < nparts-1:
            start = int(stop)
            stop = int(start+npix_per_part[part+1])

    dist_map[dist_map > 2*np.pi] = hp.UNSEEN

    return dist_map     

def apodize_mask(mask_in, aposize_in_deg, apotype="c2", tune=None):
    
    angdist = maskdist(mask_in, np.deg2rad(aposize_in_deg))
    x = np.ones_like(angdist)
    x[angdist == hp.UNSEEN] = 1.
    if apotype.lower() in ["c1", "sin","c2","cos"]:
        # following definition from NaMaster C1/C2 window definition, differing from Grain et al. 2009 definition
        x[angdist != hp.UNSEEN] = np.sqrt((1 - np.cos(angdist[angdist != hp.UNSEEN]))/(1 - np.cos(np.deg2rad(aposize_in_deg)))) 
    else:
        x[angdist != hp.UNSEEN] = angdist[angdist != hp.UNSEEN]/ np.deg2rad(aposize_in_deg)

    apo_mask = np.zeros(mask_in.shape)

    if apotype.lower() in ["c1", "sin"]:    
        # following definition from NaMaster C1 window definition, differing from Grain et al. 2009 definition
        apo_mask[x >= 1.] = 1.
        apo_mask[x <  1.] = x[x <  1.] - (np.sin(2 * np.pi * x[x <  1.]) / 2. / np.pi)

    if apotype.lower() in ["c2","cos"]: 
        # following definition from NaMaster C2 window definition, differing from Grain et al. 2009 definition
        apo_mask[x >= 1.] = 1.
        apo_mask[x <  1.] = 0.5 * (1. - np.cos(np.pi * x[x <  1.]))

    if apotype.lower() in ["mbh"]:
        # following Modified Barlett-Hanning window defined in Gautam, Kumar and Saxena 1996 (IEEE)
        # When tune is 0. matches the Grain et al. C2 window definition.
        # As tune goes closer to 1. the transition becomes steeper like a stright line. The tune parameter sharpens the transition.

        if tune == None:
            tune = 0.15
        if (tune < 0.) or (tune > 1.):
            raise Exception("ERROR: Unphysical tune for MBH apodization. Choose 0<= tune <= 1. Aborting!")

        apo_mask[x > 1.] = 1.
        apo_mask[x <=  1.] = 1. - tune*(1. - x[x <= 1.]) - (1. - tune)*(0.5 + 0.5*np.cos(np.pi*x[x <= 1.]))

    if apotype.lower() in ["cn", "cosn"]:
        # largely based on Nuttall 1981.
        if tune == None:
            tune = 4.

        if tune < 2:
            raise Exception("ERROR: Unphysical tune for cosine^n apodization. Choose tune>=2. Aborting!")

        apo_mask[x > 1.] = 1.
        apo_mask[x <=  1.] = 1. - np.cos(0.5*np.pi*x[x <= 1.])**tune

    if apotype.lower() in ["nut", "nuttall"]:
        # Based on Nuttall 1981. This the minimum 4-term window.
    
        a = np.array([0.35875, 0.48829, 0.14128, 0.01168])

        apo_mask[x >=  1.] = 1.
        apo_mask[x <  1.] = 1. - a[0] - a[1] * np.cos(np.pi*x[x < 1.]) - a[2] * np.cos(2*np.pi*x[x < 1.]) - a[3] * np.cos(3*np.pi*x[x < 1.])
        apo_mask *= mask_in

    return apo_mask

def fsky(mask_in):
    """
    The ``fsky`` function calculates the fraction of the sky that is unmasked.

    Parameters
    ----------
    mask_in : numpy ndarray (npix,)
        Input mask.

    Returns
    -------
    numpy float
        Returns the fraction of the sky that is unmasked.
    """
    npix = mask_in.shape[0]
    
    return np.sum(mask_in**2.) / npix

def fsky_signal(mask_in):
    """
    The ``fsky_signal`` function calculates the sky fraction of a signal-like field that does not scale as inverse squareroot of hitcounts.
    Based on the definition from CMB-S4 colloboration 2020 (https://arxiv.org/pdf/2008.12619) eq 10.

    Parameters
    ----------
    mask_in : numpy ndarray (npix,)
        Input mask.

    Returns
    -------
    numpy float
        Returns the effective sky fraction for signal-mode, which accounts for the masking effect
        on the signal.
    """

    npix = mask_in.shape[0]
    
    return (np.sum(mask_in**2.)**2.) / (np.sum(mask_in**4.) * npix)

def fsky_noise(mask_in):
    """
    The ``fsky_noise`` function calculates the effective sky fraction for the noise, 
    which assumes noise scales as inverse squareroot of hitcounts.
    Based on the definition from CMB-S4 colloboration 2020 (https://arxiv.org/pdf/2008.12619) eq 9.

    Parameters
    ----------
    mask_in : numpy ndarray (npix,)
        Input mask. Which is assumed to be proportional to the hitcount (inverse noise variance weights).

    Returns
    -------
    numpy float
        Returns the effective sky fraction for noise-mode, which accounts for the masking effect
        assuming noise scaling properties.
    """

    npix = mask_in.shape[0]
    
    return (np.sum(mask_in)**2.) / (np.sum(mask_in**2.) * npix)

def fsky_cross(mask_in):
    """
    The ``fsky_cross`` function calculates the effective sky fraction for cross-spectra of noise and signal, 
    which assumes noise scales as inverse squareroot of hitcounts.
    Based on the definition from CMB-S4 colloboration 2020 (https://arxiv.org/pdf/2008.12619) eq 11.

    Parameters
    ----------
    mask_in : numpy ndarray (npix,)
        Input mask. Which is assumed to be proportional to the hitcount (inverse noise variance weights).

    Returns
    -------
    numpy float
        Returns the effective sky fraction for cross of noise and signal, which accounts for the masking effect
        assuming noise scaling properties.
    """
    npix = mask_in.shape[0]
    
    return (np.sum(mask_in**2.) * np.sum(mask_in)) / (np.sum(mask_in**3.) * npix)
    


