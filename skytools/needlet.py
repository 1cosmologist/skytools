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
"""
CMBframe needlet transformation module provides functions and utilities
to performs needlet transformation of data on a sphere. The module provides
setup of cosine and gaussian needlets bands and are used in conjunction with 
healpix sherical harmonic transformation of the data on the sphere.
"""

import numpy as np 
import healpy as hp
import math as mt
import astropy.io.fits as fits
import os 

datapath = os.getenv('SKYTOOLS_DATA')

cos_band_center = [0, 30, 60, 100, 150, 300, 700, 1200, 2000]
gauss_band_fwhm = [300., 150., 90., 60., 40., 20., 10.]
__pdoc__ = {}
__pdoc__['index_mapper'] = False

def cosine_bands(band_centers = cos_band_center):
    """
    cosine_bands sets up cosine needlet bands for needlet transformations on a sphere.

    You can visualize these bands using `plot_needlet_bands`.

    Parameters
    ----------
    band_centers : list of int, optional 
        A list of ell values where we want the cosine bands to peak.
        Default is [0, 30, 60, 100, 150, 300, 700, 1200, 2000]

    Notes
    -----
    The first needlet band is starts from a peak at the set value of ``band_centers[0]``.
    All ells before ``band_centers[0]`` is included in the first needlet band with unit weights.
    The last needlet should peak at lmax of your computation. So you would like to set 
    ``band_centers[-1]`` to ``lmax`` in most calculations.

    Returns
    -------
    numpy ndarray 
        A 2D numpy array of shape = ``(np.max(band_centers)+1, len(band_centers))``.
        ``np.max(band_centers)`` is ``lmax`` of the needlets and len(band_centers) is no. of needlet bands.
    """
    
    band_centers = np.array(band_centers)
    lmax_band = np.max(band_centers)
    nbands = len(band_centers)

    cos_bands = np.zeros((lmax_band+1, nbands))

    # C_m -> denotes the band center for which we are computing
    # C_l -> denotes the band center just to the left of the current band
    # C_r -> denotes the band center just to the right of current band.

    # For the first band:
    C_m = band_centers[0]
    C_r = band_centers[1]

    # To the left of C_m for first band
    if C_m > 0 :
        cos_bands[0:C_m+1, 0] = 1. 

    # To the right of C_m for first band
    ells = C_m + np.arange(C_r - C_m + 1, dtype=np.float64)
    cos_bands[C_m:C_r+1,0] = np.cos((ells - C_m) / (C_r - C_m) * np.pi / 2.)

    if nbands > 2:
        # Compute the cosine bands 1 to nbands-2
        for band in range(1, nbands-1) :
            C_l = band_centers[band-1]
            C_m = band_centers[band]
            C_r = band_centers[band+1]

            # Left part of the band:
            ells = C_l + np.arange(C_m - C_l + 1, dtype=np.float64)
            cos_bands[C_l:C_m+1, band] = np.cos((C_m - ells ) / (C_m - C_l) * np.pi / 2.)

            # Right part of the band:
            ells = C_m + np.arange(C_r - C_m + 1, dtype=np.float64)
            cos_bands[C_m:C_r+1, band] = np.cos((ells - C_m) / (C_r - C_m) * np.pi / 2.)
        
        # Compute cosine band for nbands-1 band:
        C_l = band_centers[nbands-2]
        C_m = band_centers[nbands-1]

        # The last band only has the left part of the cosine
        ells = C_l + np.arange(C_m - C_l + 1, dtype=np.float64)
        cos_bands[C_l:C_m+1,nbands-1] = np.cos((C_m - ells) / (C_m - C_l) * np.pi / 2.) 

    return cos_bands

def gaussian_bands(fwhms=gauss_band_fwhm, lmax_band=None):
    """
    gaussian_bands sets up gaussian needlet bands for needlet transformations on a sphere.

    You can visualize these bands using `plot_needlet_bands`.

    Parameters
    ----------
    fwhms : list of floats, optional
        A list of fwhm values in arcmin of the different gaussian needlet map resolutions. 
        Defaults is [300., 150., 90., 60., 40., 20., 10.]
    lmax_band : int, optional 
        It is recommended to set this to the lmax using in the computation.
        Default is ``int(360.* 60./ np.min(fwhms))``

    Notes
    -----
    The number of gaussian needlet bands produced is 1 more than the number of fwhms
    set. The final band is there to pick up the residual weight after needlet weights for 
    all the gaussian bands.

    Returns
    -------
    numpy ndarray
        A 2D numpy array of shape = ``(lmax_band+1, len(fwhms) + 1)``. 
        ``nbands=len(fwhms)+1`` is no. of needlet bands.
    """    
    
    fwhms = np.array(fwhms)
    sorted_fwhms = np.sort(fwhms)[::-1]
    nbands = len(fwhms) + 1

    if lmax_band == None:
        lmax_band = int(360.* 60./ np.min(fwhms))

    gauss_bands = np.zeros((lmax_band+1, nbands))

    # First band:
    gauss_bands[:,0] = hp.gauss_beam(np.deg2rad(sorted_fwhms[0]/60.), lmax=lmax_band)**2.

    # Intermediate bands:
    if nbands > 2:
        for band in range(1,nbands-1):
            gauss_bands[:,band] = hp.gauss_beam(np.deg2rad(sorted_fwhms[band]/60.), lmax=lmax_band)**2. -\
                hp.gauss_beam(np.deg2rad(sorted_fwhms[band-1]/60.), lmax=lmax_band)**2.
    
    # Last band:
    gauss_bands[:,nbands-1] = 1. - hp.gauss_beam(np.deg2rad(sorted_fwhms[nbands-2]/60.), lmax=lmax_band)**2.

    gauss_bands = np.sqrt(gauss_bands)

    return gauss_bands

def get_lmax_band(band):
    """
    Returns the highest ell value where a needlet band is 5% of its peak.

    Parameters
    ----------
    band : numpy array
        A 1D numpy array containing the needlet band. Typically ``needlets[:,i]`` 
        slice for the ith needlet band.

    Notes
    -----
    This function is meant to find the maximum lmax contributing to a needlet
    band. This is set to 5% to deal with the long tail of the gaussian needlets.
    
    Returns
    -------
    int
        An integer value for the nonzero lmax of the needlet band.
    """
    
    lmax = len(band) - 1
    lmax_band = np.max(np.where(band > 0)[0])

    if lmax_band != lmax or band[lmax_band] < 5.e-2:
        needlet_peak = np.min(np.where(band == np.max(band))[0])
        lmax_band = needlet_peak - 1 + min(np.where(band[needlet_peak:] < 5.e-2)[0])

    return lmax_band

def alm2needlet(alm_in, bands, nodegrade=False, needlet_nside=None, nlt_nside_min=None, nlt_nside_max=None):
    """
    Performs needlet transform from input spherical harmonic coefficient (alm) of a map.

    Parameters
    ----------
    alm_in : numpy array
        A 1D numpy array of alm for a map obtained from healpy.
        Does not support multicomponent maps. For polarization 
        input should be either \(a^E_{\\ell m}\) or \(a^B_{\\ell m}\).
    bands : numpy ndarray
        A 2D numpy array of shape ``(lmax+1, nbands)'' conatining 
        needlet bands
    nodegrade : bool, optional
        Sets if all needlet scale maps have the same nside.
        If True then you can set the common nside for all scales 
        with needlet_nside. If nside not specified, the common 
        nside for all scales is set by: math.ceil(math.log2(lmax)),
        with lmax determined from needlet bands.
    needlet_nside : int, optional
        Nside of needlet maps at all bands. Default is None.
    nlt_nside_min : int, optional
        Minimum nside for any needlet scale maps. Default is None.
    nlt_nside_max : int, optional
        Maximum nside for any needlet scale maps. Default is None.

    Notes
    -----
    When nside is not set by user the nside of each needlet scale map 
    is 2^n such that ``2^n >= lmax_band`` (n is integer). The ``lmax_band`` is
    obtained by calling `get_lmax_band`. 

    Returns
    -------
    list
        A list of healpix maps is returned. There will be nband healpix
        maps as 1D numpy arrays. If all maps are not set to have the same
        nside, do not convert the whole list of maps to a numpy ndarray.
    """
    
    if np.array(alm_in).ndim != 1:
        print("ERROR: Cannot needlet transform multiple maps.")
        exit()

    lmin = 0
    lmax = len(bands[:,0]) - 1

    ALM = hp.Alm()
    lmax_sht = ALM.getlmax(len(alm_in))

    nbands = len(bands[0,:])

    # if lmax > lmax_sht:
    #     needlet_bands = np.copy(bands[:lmax_sht+1,:nbands]) 
    #     print("Note bands have higher lmax than alms. Some needlets may have no information")
    # else :
    needlet_bands = np.copy(bands)

    nside_opts = np.array([1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]) 

    needlet_maps = []
    for band in range(nbands):

        lmax_band = get_lmax_band(needlet_bands[:, band])

        # # lmax_band = np.max(np.where(needlet_bands[:,band] > 0))
        # lmax_band = np.max(np.where(needlet_bands[:,band] > 0)[0])
        # # print(lmax_band)
        # if lmax_band != lmax or needlet_bands[lmax_band,band] < 5.e-2:
        #     needlet_peak = np.min(np.where(needlet_bands[:,band] == np.max(needlet_bands[:,band]))[0])
        #     lmax_band = min(np.where(needlet_bands[needlet_peak:,band] < 5.e-2)[0])
        
        if not nodegrade:
            nside_band = np.min(nside_opts[nside_opts >= lmax_band])
            if nlt_nside_min != None :
                if nside_band < nlt_nside_min : nside_band = nlt_nside_min
            if nlt_nside_max != None :
                if nside_band > nlt_nside_max : nside_band = nlt_nside_max 
        else :
            if needlet_nside != None:
                nside = needlet_nside
            else:
                nside = mt.ceil(mt.log2(lmax))

            nside_band = nside 

        # print(nside_band, lmax_band)

        wavalm = hp.almxfl(np.copy(alm_in),needlet_bands[:,band])
        needlet_maps.append(hp.alm2map(wavalm, nside_band, pol=False, verbose=False))

        del lmax_band, nside_band, wavalm

    del alm_in, bands
    return needlet_maps

def map2needlet(map_in, bands, needlet_nside=None, nodegrade=False, nlt_nside_min=None, nlt_nside_max=None):
    """
    Performs needlet transform from input healpix map. This
    is a map level wrapper for alm2needlet function.

    Parameters
    ----------
    map_in : numpy array
        A 1D numpy array for a healpix map. Does not support 
        multicomponent maps. For polarization input should be 
        either E or B mode scalar map. Does not support IQU maps.
    bands : numpy ndarray
        A 2D numpy array of shape ``(lmax+1, nbands)`` conatining 
        needlet bands
    nodegrade : bool, optional
        Sets if all needlet scale maps have the same nside.
        If True then you can set the common nside for all scales 
        with needlet_nside. If nside not specified, the common 
        nside for all scales is set by: \(\\lceil \\log_2(\ell_{\\rm max}) \\rceil \),
        with lmax determined from needlet bands.
    needlet_nside : int, optional
        Nside of needlet maps at all bands. Default is None.
    nlt_nside_min : int, optional
        Minimum nside for any needlet scale maps. Default is None.
    nlt_nside_max : int, optional
        Maximum nside for any needlet scale maps. Default is None.

    Notes
    -----
    When nside is not set by user the nside of each needlet scale map 
    is 2^n such that ``2^n >= lmax_band`` (n is integer). The ``lmax_band`` is
    obtained by calling get_lmax_band. 

    Returns
    -------
    list
        A list of healpix maps is returned. There will be nband healpix
        maps as 1D numpy arrays. If all maps are not set to have the same
        nside, do not convert the whole list of maps to a numpy ndarray.
    """

    if np.array(map_in).ndim != 1:
        print("ERROR: Cannot needlet transform multiple maps.")
        exit()

    lmax_wv = len(bands[:,0])-1
    nside =  hp.npix2nside(len(map_in))
    lmax_map = 3*nside - 1

    if lmax_wv > lmax_map:
        print("WARNING: lmax of needlet bands exceeds map lmax limit.")
        lmax_sht = lmax_map
    else:
        lmax_sht = lmax_wv

    alm_temp = hp.map2alm(map_in, lmax=lmax_sht, pol=False, use_weights=True, datapath=datapath)

    needlet_map = alm2needlet(alm_temp, bands, needlet_nside=needlet_nside, nodegrade=nodegrade, nlt_nside_min=nlt_nside_min, nlt_nside_max=nlt_nside_max)

    del alm_temp

    return needlet_map

def index_mapper(lmax_i, lmax_f, idx_i):
    ALM = hp.Alm()
    l, m = ALM.getlm(lmax_i,idx_i)
    idx_f = ALM.getidx(lmax_f, l, m)
    del ALM, l, m 

    return idx_f

def needlet2alm(nlt_map_in, bands):
    """
    Converts needlet maps back to spherical harmonic coefficients (alm).

    Parameters
    ----------
    nlt_map_in : list
        A list of healpix maps. One map for each needlet band. Each healpix
        map is a numpy 1D array. The nside of the map is determined from the 
        length of the map array. Partial maps not supported.
    bands : numpy ndarray
        A 2D numpy array of shape ``(lmax+1, nbands)`` conatining needlet bands
        used to produce the needlet scale maps. number of maps must match nbands.


    Returns
    -------
    numpy array
        A numpy 1D array of output spherical harmonic coefficients are returned.
        The alm array is indexed in healpix C/python format. The output alm is 
        compatible with healpy SHTs.
    """

    nbands = len(bands[0,:])
    lmax_wv = len(bands[:,0])-1

    ALM = hp.Alm()
    sz = ALM.getsize(lmax_wv)

    alm_out = np.zeros((sz,), dtype=np.complex128)

    map_index = np.vectorize(index_mapper, otypes=[np.int64])

    for band in range(nbands):
        lmax_band = get_lmax_band(bands[:,band])

        # print(nlt_map_in[band].shape, band)
        alm_band = hp.map2alm(np.copy(nlt_map_in[band]),lmax=lmax_band, pol=False, use_weights=True, datapath=datapath)
        
        idx_arr = np.arange(len(alm_band),dtype=np.int64)
        idx_map = map_index(lmax_band, lmax_wv, idx_arr)

        alm_out[idx_map] = alm_out[idx_map] + hp.almxfl(alm_band, bands[:lmax_band+1,band])

        del alm_band, idx_map, idx_arr

    del ALM, nbands, lmax_wv

    return alm_out

def needlet2map(nside, nlt_map_in, bands, map_outfile=None):
    """
    Converts needlet maps back to single healpix map combining all needlet scales.
    This is a wrapper for needlet2alm function.

    Parameters
    ----------
    nside : int
        Healpix nside parameter of the final output map. Sets the pixel density
        of the output map. Valid inputs are integer powers of 2 <= 8192.
    nlt_map_in : list
        A list of healpix maps. One map for each needlet band. Each healpix
        map is a numpy 1D array. The nside of the map is determined from the 
        length of the map array. Partial maps not supported.
    bands : numpy ndarray
        A 2D numpy array of shape ``(lmax+1, nbands)`` conatining needlet bands
        used to produce the needlet scale maps. number of maps must match nbands.
    map_outfile : str, optional
        Output filename, with path, to save the output map as fits file. Only
        valid file extension is fits.

    Returns
    -------
    numpy array
        A numpy 1D array of a healpix map representing the combined information from
        all needlet scales. This map will have the nside set in the input.
    """
    
    alm_temp = needlet2alm(nlt_map_in, bands)

    map_out = hp.alm2map(alm_temp, nside, pol=False, verbose=False)

    if map_outfile != None:
        hp.write_map(map_outfile, map_out, dtype=np.float64, overwrite=True)

    return map_out

def write_needletmap_fits(bands, nlt_map_in, outfile, unit=''):
    """
    Writes needlet maps and corresponding needlet band information to a fits file.

    Paramaters
    ----------
    bands : numpy ndarray
        A 2D numpy array of shape ``(lmax+1, nbands)`` conatining needlet bands
        used to produce the needlet scale maps. Number of maps must match nbands.
    nlt_map_in : list
        A list of healpix maps. One map for each needlet band. Each healpix
        map is a numpy 1D array. 
    outfile : str
        Output filename, with path, to save the input needlet bands and needlet maps 
        as fits file. Only valid file extension is fits. This file is meant to read 
        with read_needletbands_fits and read_needletmap_fits.
    unit : str, optional
        Set the units of the output map.

    Notes
    -----
    Saves the needlet bands in the PrimaryHDU of the fits file. Saves the needlet maps
    as a BinTableHDU with each map as a new column. 
    """

    # hdr_P = fits.Header()
    # hdr_P['nbands'] = 

    hdu_P = fits.PrimaryHDU(bands)
    hdulist = [hdu_P]
    for i in range(len(bands[0,:])):
        tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='Band '+str(i), unit=unit, format='D', array=nlt_map_in[i])])
        hdulist.append(tbhdu)

        del tbhdu

    hdulist = fits.HDUList(hdulist)
    hdulist.writeto(outfile, clobber=True)

def read_needletmap_fits(infile, band_no=None):
    """
    Reads needlet map(s) contained in the BintableHDU of a fits file created with
    write_needletmap_fits. 

    Paramaters
    ----------
    infile : str
        Input filename, with path, to read the needlet scale maps from a fits file. 
        Only valid file extension is fits. Must be created with write_needletmap_fits.
    band_no : int, optional
        Specifies the needlet scale map to fetch by number. Goes from 1 to nbands.
        Default is None.
    Returns
    -------
    list or numpy array
        If band_no is set to None, returns a list of healpix maps. One map for each 
        needlet band. Each healpix map is a numpy 1D array. If band_no is specified, 
        returns the healpix map as a numpy 1D array.
    """
    
    with fits.open(infile) as hdulist:
        needlets = np.array(hdulist[0].data)

        nbands = len(needlets[0,:])

        if band_no == None:
            wav_maps = []
            for i in range(1, nbands+1):
                wav_maps.append(np.array(hdulist[i].data['Band '+str(i-1)]))

            return wav_maps
        else:
            if band_no < nbands:
                return np.array(hdulist[band_no+1].data['Band '+str(band_no)])
            else:
                print("ERROR: band number exceeds number of bands in file.")
                exit()

def read_needletbands_fits(infile):
    """
    Reads needlet band information contained in PrimaryHDU of a fits file created with
    write_needletmap_fits.

    Paramaters
    ----------
    infile : str
        Input filename, with path, to read the needlet bands from a fits file. 
        Only valid file extension is fits. Must be created with write_needletmap_fits.

    Returns
    -------
    numpy ndarray
        A 2D numpy array of shape ``(lmax+1, nbands)`` conatining needlet bands
        used to produce the needlet scale maps contained in the file.
    """
    
    with fits.open(infile) as hdulist:
        return np.array(hdulist[0].data)

