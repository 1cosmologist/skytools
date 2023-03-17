#######################################################################
# This file is a part of SkyToolsLib
#
# Sky Tools Library
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
# For more information about CMBframe please visit 
# <https://github.com/...> or contact Shamik Ghosh 
# at shamik@lbl.gov
#
#########################################################################

import numpy as np 
import healpy as hp
from . import hpx_utils as hu
from . import mask_tools as mt

def smalldisc_mask(nside, lon, lat, radius_in_deg, aposize=None):
    npix = hp.nside2npix(nside)
    mask = np.zeros((npix,))

    vec_cen = np.array(hp.ang2vec(lon, lat, lonlat=True))
    if aposize == None:
        mask[hp.query_disc(nside, vec_cen, np.deg2rad(radius_in_deg), inclusive=True)] = 1.
    else:
        disc_pix, dist_pix = hu.query_dist(nside, vec_cen, np.deg2rad(radius_in_deg), inclusive=True)
        x = ((np.deg2rad(radius_in_deg) - dist_pix) / np.deg2rad(aposize))

        del dist_pix
        mask[disc_pix[x >= 1.]] = 1.
        mask[disc_pix[x < 1.]] = 0.5 * (1. - np.cos(np.pi * x[x <  1.]))

        del disc_pix, x 

    return mask

def latitude_mask(nside, lat_cut, aposize=None, inverse=False):
    npix = hp.nside2npix(nside)
    mask = np.zeros((npix,))

    ipix = np.arange(npix)
    lon, lat = hp.pix2ang(nside, ipix, lonlat=True)

    if aposize == None:
        if inverse:
            mask[ipix[np.abs(lat) < lat_cut]] = 1.
        else:
            mask[ipix[np.abs(lat) > lat_cut]] = 1.
        return mask
    
    if inverse:
        mask[np.abs(lat) >= lat_cut] = 0.
        valid_pix = ipix[np.abs(lat) < lat_cut]
        
        x = ((lat_cut - np.abs(lat)[valid_pix]) / aposize)
        
    else:
        mask[np.abs(lat) <= lat_cut] = 0.
        valid_pix = ipix[np.abs(lat) > lat_cut]
        x = ((np.abs(lat)[valid_pix] - lat_cut) / aposize)

    mask[valid_pix[x >= 1.]] = 1.
    mask[valid_pix[x < 1.]] = 0.5 * (1. - np.cos(np.pi * x[x <  1.]))

    return mask

def intensity_mask(nside, IorP_map, percent_masked, smooth_in_deg=None, percent_apod=0., saturate=False):
    IorP_map = np.array(IorP_map)

    if (percent_masked < 0.) or (percent_apod < 0.) :
        raise Exception("ERROR: Either percent_masked or percent_apod is set to negative. Aborting!")
            
    if percent_masked + percent_apod > 100.:
        raise Exception("ERROR: Percentage masked and apodized adds to more than 100%, which is unphysical! Aborting.")

    if IorP_map.ndim > 1:
        raise Exception("ERROR: Too many dimensions for intensity/polarized intensity map. Aborting!")
        
    npix = hp.nside2npix(nside)
    if smooth_in_deg is None:
        smooth_map = hp.ud_grade(IorP_map, nside)   #Assuming presmoothed
    elif smooth_in_deg > 0.:
        smooth_map = hu.change_resolution(IorP_map, nside, mode='i', fwhm_out=smooth_in_deg*60.)

    del IorP_map

    ipix_sorted = np.argsort(smooth_map)

    mask = np.zeros((npix,))

    if percent_apod > 0.:
        npix_keep = int(((100. - percent_masked - percent_apod) / 100.) * npix)
        if npix_keep > 0: mask[ipix_sorted[0:npix_keep]] = 1.

        npix_apod = int((percent_apod / 100.) * npix)

        if npix_apod > 0: mask[ipix_sorted[npix_keep:npix_keep+npix_apod]] = np.cos(np.pi / 2. * np.arange(npix_apod) / npix_apod)
    else:
        npix_keep = int(((100. - percent_masked) / 100.) * npix)

        if npix_keep > 0: mask[ipix_sorted[0:npix_keep]] = 1.
        
        if saturate: mask[ipix_sorted[npix_keep:npix_keep+npix]] = smooth_map[ipix_sorted[npix_keep]] / smooth_map[ipix_sorted[npix_keep:npix_keep+npix]]
    
    del smooth_map, ipix_sorted
    
    return mask

