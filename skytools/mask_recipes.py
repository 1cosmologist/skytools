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
The SkyTools mask recipes module provides a set of useful functions to construct sky masks.
"""

import numpy as np 
import healpy as hp
from . import hpx_utils as hu
from . import mask_tools as mt

def smalldisc_mask(nside, lon, lat, radius, aposize=None):
    
    """
    Create a circular mask centered at a specified longitude and latitude.

    Parameters
    ----------
    nside : int
        The ``NSIDE`` parameter of the HEALPix map, defining its resolution.
    lon : float
        Longitude of the center of the disc in degrees.
    lat : float
        Latitude of the center of the disc in degrees.
    radius : float
        Radius of the disc in degrees.
    aposize : float, optional
        Apodization size in degrees. If provided, a cosine apodization is applied over this angular distance.

    Returns
    -------
    numpy.ndarray
        A 1D numpy array representing the HEALPix mask, with pixels within the specified disc set to 1.
    """

    npix = hp.nside2npix(nside)
    mask = np.zeros((npix,))

    vec_cen = np.array(hp.ang2vec(lon, lat, lonlat=True))
    if aposize == None:
        mask[hp.query_disc(nside, vec_cen, np.deg2rad(radius), inclusive=True)] = 1.
    else:
        disc_pix, dist_pix = hu.query_dist(nside, vec_cen, np.deg2rad(radius), inclusive=True)
        x = ((np.deg2rad(radius) - dist_pix) / np.deg2rad(aposize))

        del dist_pix
        mask[disc_pix[x >= 1.]] = 1.
        mask[disc_pix[x < 1.]] = 0.5 * (1. - np.cos(np.pi * x[x <  1.]))

        del disc_pix, x 

    return mask

def latitude_mask(nside, lat_cut, aposize=None, inverse=False):
    """
    Create a mask for a specified latitude cut.

    Parameters
    ----------
    nside : int
        The ``NSIDE`` parameter of the HEALPix map, defining its resolution.
    lat_cut : float
        The latitude in degrees below which to mask out pixels, ie \(|b| < \)``lat_cut``.
    aposize : float, optional
        Apodization size in degrees. If provided, a cosine apodization is applied over this angular distance.
    inverse : bool, optional
        If True, mask out pixels above the specified latitude, ie \(|b| > \)``lat_cut``. Default is False.

    Returns
    -------
    numpy.ndarray
        A 1D numpy array representing the binary HEALPix mask, unmasked pixels set to 1.
    """
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

def galridge_mask(nside, lat_cut, lon_cut, aposize=None):
    """
    Create a mask for a Galactic "ridge" region. The mask is set to 1 for all pixels outside a "box" defined by the specified longitude and latitude cuts. 
    If an apodization size is provided, a cosine apodization is applied over this angular distance.

    Parameters
    ----------
    nside : int
        The ``NSIDE`` parameter of the HEALPix map, defining its resolution.
    lat_cut : float
        The latitude in degrees below which to mask out pixels , ie \(|b| < \)``lat_cut``.
    lon_cut : float
        The longitude in degrees below which to mask out pixels, ie \(|l| < \)``lon_cut``.
    aposize : float, optional
        Apodization size in degrees. If provided, a cosine apodization is applied over this angular distance.

    Returns
    -------
    numpy.ndarray
        A 1D numpy array representing the binary HEALPix mask, unmasked pixels set to 1. 
        This masks out a box region centered on the map with size of ``2 * lat_cut`` in latitude and ``2 * lon_cut`` in longitude.
    """
    npix = hp.nside2npix(nside)
    mask = np.zeros((npix,))

    ipix = np.arange(npix)
    lon, lat = hp.pix2ang(nside, ipix, lonlat=True)

    lon[lon > 180] = 360. - lon[lon > 180]

    if aposize == None:
        
        mask[ipix[(np.abs(lat) > lat_cut) | (np.abs(lon) > lon_cut)]] = 1.
        return mask
    
    
    mask[(np.abs(lat) <= lat_cut) & (np.abs(lon) <= lon_cut)] = 0.
    valid_pix = ipix[(np.abs(lat) > lat_cut) & (np.abs(lon) <= lon_cut)]
    x = ((np.abs(lat)[valid_pix] - lat_cut) / aposize)

    mask[valid_pix[x >= 1.]] = 1.
    mask[valid_pix[x < 1.]] = 0.5 * (1. - np.cos(np.pi * x[x <  1.]))

    valid_pix = ipix[(np.abs(lat) <= lat_cut) & (np.abs(lon) > lon_cut)]
    x = ((np.abs(lon)[valid_pix] - lon_cut) / aposize)

    mask[valid_pix[x >= 1.]] = 1.
    mask[valid_pix[x < 1.]] = 0.5 * (1. - np.cos(np.pi * x[x <  1.]))

    valid_pix = ipix[(np.abs(lat) > lat_cut) & (np.abs(lon) > lon_cut)]
    mask[valid_pix] = 1.

    valid_pix = ipix[(np.abs(lat) > lat_cut) & (np.abs(lat) <= lat_cut+1.05*aposize) & (np.abs(lon) > lon_cut)]
    corner_angs = np.array([[lon_cut, lat_cut],[lon_cut, -lat_cut], [360.-lon_cut, -lat_cut], [360.-lon_cut, lat_cut]])
    corner_vecs = np.array(hp.ang2vec(corner_angs[:,0],corner_angs[:,1], lonlat=True), dtype=np.float32)
    valid_pix_vecs = np.array(hp.pix2vec(nside, valid_pix), dtype=np.float32)

    # print(corner_vecs.shape, valid_pix_vecs.shape)
    dist = np.amin(np.arccos(corner_vecs @ valid_pix_vecs).astype(np.float32), axis=0)

    del corner_angs, valid_pix_vecs
    x = (dist / np.deg2rad(aposize))

    mask[valid_pix[x >= 1.]] = 1.
    mask[valid_pix[x < 1.]] = 0.5 * (1. - np.cos(np.pi * x[x <  1.]))

    return mask


def intensity_mask(nside, IorP_map, percent_masked, smooth_in_deg=None, percent_apod=0., saturate=False):
    """
    Generate a mask based on intensity or polarized intensity map.

    This function processes an input HEALPix map to create a binary mask by masking 
    a specified percentage of the highest intensity pixels. Additionally, it offers 
    optional smoothing and apodization features.

    Parameters
    ----------
    nside : int
        The ``NSIDE`` parameter of the HEALPix map, defining its resolution.
    IorP_map : array-like
        A 1D array representing the intensity or polarized intensity map.
    percent_masked : float
        Percentage of the lowest intensity pixels to be masked.
    smooth_in_deg : float, optional
        The smoothing scale in degrees. If provided, the map is smoothed before 
        creating the mask. Default is None, which indicates no smoothing.
    percent_apod : float, optional
        Percentage of pixels to apodize. A cosine apodization is applied over 
        this percentage. Default is 0.
    saturate : bool, optional
        If True, saturate the mask for pixels beyond the specified percentage.
        Default is False.

    Returns
    -------
    numpy.ndarray
        A 1D array representing the binary HEALPix mask, where unmasked pixels 
        are set to 1.
    """

    IorP_map = np.array(IorP_map)

    if (percent_masked < 0.) or (percent_apod < 0.) :
        raise Exception("ERROR: Either percent_masked or percent_apod is set to negative. Aborting!")

    if percent_masked + percent_apod > 100.:
        raise Exception("ERROR: Percentage masked and apodized adds to more than 100%, which is unphysical! Aborting.")

    if IorP_map.ndim > 1:
        raise Exception("ERROR: Too many dimensions for intensity/polarized intensity map. Aborting!")
        
    npix = hp.nside2npix(nside)
    
    # if isinstance(footprint_mask, np.ndarray):
    #     hu.mask_udgrade(footprint_mask, nside)
    # else:  
    #     footprint_mask = np.ones((npix,)) 
       
    # ipix_sel = np.where(footprint_mask > 0.5)[0]
        
    if smooth_in_deg is None:
        smooth_map = hp.ud_grade(IorP_map, nside) #* footprint_mask  #Assuming presmoothed
    elif smooth_in_deg > 0.:
        smooth_map = hu.change_resolution(IorP_map, nside, mode='i', fwhm_out=smooth_in_deg*60.) #* footprint_mask

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

def saturate_mask(mask_in, clip_val):
    """
    Saturate a mask at a given value and normalize.

    Parameters
    ----------
    mask_in : numpy ndarray (npix,)
        Input mask. 
    clip_val : float
        Value above which the mask is set to 1., and below which the mask is divided by the clip_val.

    Returns
    -------
    numpy ndarray (npix,)
        Returns the saturated mask.
    """
    saturated_mask = np.copy(mask_in)
    saturated_mask[saturated_mask >= clip_val] = 1.
    saturated_mask[saturated_mask < clip_val] = saturated_mask[saturated_mask < clip_val] / clip_val

    return saturated_mask