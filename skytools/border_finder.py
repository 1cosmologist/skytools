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
This module contains functions for finding the border of a binary mask.
"""
import numpy as np 
import healpy as hp
import os

datapath = os.getenv('SKYTOOLS_DATA')

def nwhere(array, cond, val, ma=False):
    if (cond == 'gt') or (cond == 'GT'):
        if ma:
            where_arr = np.array(np.ma.where(array > val))[0]
        else:
            where_arr = np.array(np.where(array > val))[0]
    elif (cond == 'lt') or (cond == 'LT'):
        if ma:
            where_arr = np.array(np.ma.where(array < val))[0]
        else:
            where_arr = np.array(np.where(array < val))[0]
    elif (cond == 'eq') or (cond == 'EQ'):
        if ma:
            where_arr = np.array(np.ma.where(array == val))[0]
        else:
            where_arr = np.array(np.where(array == val))[0]

    nwh = len(where_arr)

    return where_arr, nwh
    

def get_mask_border(mask_in, need_nos=False):
    """
    Get the border pixels between masked and unmasked regions in a HEALPix map.

    Parameters
    ----------
    mask_in : numpy.ndarray
        A 1D numpy array with the input binary mask, where masked regions are indicated by 0s.
    need_nos : bool, optional
        If True, return the number of pixels in the border region. Default is False.

    Returns
    -------
    p0, p1 : numpy.ndarray
        1D numpy arrays containing the pixel numbers of the border region, with p0 and p1
        referring to the masked and unmasked region border pixels, respectively.
    np0, np1 : int
        If need_nos is True, the number of pixels in the border region.
    """
    npix = hp.get_map_size(mask_in)
    nside = hp.get_nside(mask_in)

    whmask, nmasked = nwhere(mask_in, 'eq', 1.)

    # All masked case
    if nmasked == 0:
        p0 = np.array([])
        p1 = np.array([])
        np0 = 0
        np1 = 0

    # For low number of holes:

    elif np.sum(1. - mask_in) / len(mask_in) <= 5e-2:

        whmask, nwhmask = nwhere(mask_in, 'EQ', 0.) 

        nei = hp.get_all_neighbours(nside, whmask)

        ip = nei.flatten()
        ip = ip[np.where(ip>=0)]

        ip_mask = np.ones(hp.nside2npix(nside), dtype='bool')
        ip_mask[ip] = False
        
        whp1, nwhp1 = nwhere(np.ma.masked_array(mask_in, mask=ip_mask), 'eq', 1, ma=True)

        if nwhp1 == 0:
            print('Something wrong here! Low number of holes case.')

        p1 = np.unique(whp1)

        whp0 = np.sum(np.isin(nei,p1), axis=0)
        
        p0 = whmask[np.where(whp0 >= 1)[0]]

        np0 = len(p0)
        np1 = len(p1)

    # For low number of valid pixels:

    elif np.sum(mask_in) / len(mask_in) <= 5e-2:

        whnomask, nwhnomask = nwhere(mask_in, 'EQ', 1.) 

        nei = hp.get_all_neighbours(nside, whnomask)

        ip = nei.flatten()
        ip = ip[np.where(ip>=0)]

        ip_mask = np.ones(hp.nside2npix(nside), dtype='bool')
        ip_mask[ip] = False
        
        whp0, nwhp0 = nwhere(np.ma.masked_array(mask_in, mask=ip_mask), 'eq', 0, ma=True)

        if nwhp0 == 0:
            print('Something wrong here! Low number of valid pixels case.')

        p0 = np.unique(whp0)

        whp1 = np.sum(np.isin(nei,p0), axis=0)
        
        p1 = whmask[np.where(whp1 >= 1)[0]]

        np0 = len(p0)
        np1 = len(p1)


    # For intermediate cases

    else:
        pixsize = hp.nside2resol(nside)
        mask_lm = hp.map2alm(mask_in, pol=False, use_weights=True, datapath=datapath)
        mask_lm_sm = hp.smoothalm(mask_lm, fwhm=(np.sqrt(2.) * pixsize), pol=False)
        smooth_mask = hp.alm2map(mask_lm_sm, nside, pol=False)

        del mask_lm, mask_lm_sm

        p0 = np.where((mask_in == 0) & (smooth_mask >= 0.15))[0]
        nwh0 = len(p0)

        p1 = np.where((mask_in == 1) & (smooth_mask <= 0.85))[0]
        nwh1 = len(p1)

        if nwh0 != 0 :
            m = np.zeros_like(mask_in)
            m[p1] = 1

            nei = np.unique(hp.get_all_neighbours(nside, p0).flatten())

            m[nei[nei>=0]] = 1

            m *= mask_in

            p1, np1 = nwhere(m, 'eq', 1)

        if nwh1 != 0 :
            m = np.zeros_like(mask_in)
            m[p0] = 1

            nei = np.unique(hp.get_all_neighbours(nside, p1).flatten())

            m[nei[nei>=0]] = 1

            m *= (1 - mask_in)

            p0, np0 = nwhere(m, 'eq', 1)


    p0 = np.unique(p0)
    p1 = np.unique(p1)

    np0 = min(np0, len(p0))
    np1 = min(np1, len(p1))

    if need_nos:
        return p0, p1, np0, np1
    else:
        return p0, p1 