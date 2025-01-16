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
The SkyTools HEALPix utilities module provides useful and frequently used macro funtions 
to augment the function set available with Healpy. Some of these utility functions are 
inspired by HEALPix Fortran Facilities.
"""

import numpy as np
import healpy as hp
import os 

datapath = os.getenv('SKYTOOLS_DATA')

__pdoc__ = {}
__pdoc__['alm_fort2c', 'alm_c2fort'] = False

def apodized_gauss_beam(fwhm, lmax):
    """
    Compute a Gaussian beam with a linear apodization at high ell to transition the beam to zero.

    Parameters
    ----------
    fwhm : float
        The full width at half maximum of the beam in arcmin.
    lmax : int
        The multipole at which the beam goes to zero.

    Returns
    -------
    Bl : array
        The apodized beam up to multipole ``lmax``.
    """
    Bl = hp.gauss_beam(np.deg2rad(fwhm / 60.), lmax=lmax)

    Bl_apo = np.copy(Bl)
    dBl = np.gradient(Bl) 

    ells = np.arange(lmax+1, dtype=np.int16)

    ell_intercept = ells - Bl / dBl
    # lmax_intercept = Bl + (lmax - ells) * dBl

    if np.sum(ell_intercept <= lmax) > 0:
        ell_0 = np.where(ell_intercept <= lmax)[0][-1]
    else:
        print('Warning: lmax is too small for a beam that size')
        # dummy = np.min(lmax_intercept, ell_0)

    tangent = Bl[ell_0] + (ells - ell_0) * dBl[ell_0]
    Bl_apo[ell_0:] = tangent[ell_0:]
    
    return Bl_apo

def compute_beam_ratio(beam_nu, beam_0, thresh=0.):
    """
    Computes beam ratio to change the resolution/beam smoothing of a single map/alm.

    Parameters
    ----------
    beam_nu : numpy ndarray
        A numpy array of shape [lmax+1], containing the original/native beam of the data. 
        If polarized beam contains either the E component or the B component depending on 
        which map/alm is being targeted. This represents $$b^{T/E/B}_{\\ell}$$ for the 
        different maps in the set.
    beam_0 : numpy ndarray
        A numpy array of shape [lmax+1] representing the beam of the common resolution 
        that is being targetted.

    Returns
    -------
    numpy ndarray
        A numpy ndarray of shape [lmax+1] that contains multiplicative factors 
        to convert map alms to the common resolution. 
    """

    lmax_beam = len(beam_nu)

    ratio_nu = np.zeros((lmax_beam))

    lmax_nonzero = np.max(np.where(beam_nu>thresh))+1
    ratio_nu[0:lmax_nonzero] = beam_0[0:lmax_nonzero] / beam_nu[0:lmax_nonzero]

    del lmax_beam, lmax_nonzero, beam_nu, beam_0
    return ratio_nu


def iqu2teb(map_iqu, mask_in=None, nside=None, mode='teb', lmax_sht=None, return_alm=False):
    """
    Returns TEB maps from IQU Healpix maps. 

    Parameters
    ----------
    map_iqu : numpy ndarray
        A numpy array of shape (3, Npix) which contains IQU maps.
    mask_in : numpy ndarray, optional
        A numpy array of shape (Npix,) which contains the mask which will be applied to IQU maps.
    nside : int, optional
        Nside of TEB output maps. Default is None.
    mode : str, optional
        String specifying the output mode map. Possible mode values are all possible variations of "teb" (e.g. "te"). Default is "teb".
        Note, that this keyword has changed from ``teb`` to ``mode``. If you are using version 0.0.1.b5 or earlier, please use ``teb`` instead.
    lmax_sht : int, optional
        Maximum l of the power spectrum. Default is None.
    return_alm : bool, optional
        Returns alm of TEB or the specified mode instead of the map. Default is False. 

    Returns
    -------
    numpy ndarray
        A numpy array of TEB maps or TEB alms.
    """
    if nside == None:
        nside = hp.get_nside(map_iqu[0])

    if not isinstance(mask_in,(list, np.ndarray)):
        mask_in = np.ones_like((hp.nside2npix(nside),))

    mask_arr = [mask_in, mask_in, mask_in]
    alms = hp.map2alm(map_iqu * mask_arr, lmax=lmax_sht, use_weights=True, datapath=datapath)

    mask_bin = np.ones_like(mask_in)
    mask_bin[mask_in == 0.] = 0.

    teb_maps = []
    if ('t' in mode) or ('T' in mode) :
        if return_alm:
            teb_maps.append(alms[0])
        else:
            teb_maps.append(hp.alm2map(alms[0], nside, lmax=lmax_sht, pol=False) * mask_bin)
    if ('e' in mode) or ('E' in mode) :
        if return_alm:
            teb_maps.append(alms[1])
        else:
            teb_maps.append(hp.alm2map(alms[1], nside, lmax=lmax_sht, pol=False) * mask_bin)
    if ('b' in mode) or ('B' in mode) :
        if return_alm:
            teb_maps.append(alms[2])
        else:
            teb_maps.append(hp.alm2map(alms[2], nside, lmax=lmax_sht, pol=False) * mask_bin)

    return np.array(teb_maps)


def roll_bin_Cl(Cl_in, dl_min=10, dlbyl=0.4, dl_max=None, fmt_nmt=False):
    """
    Bins Cl power spectra using box cart averaging.

    Parameters
    ----------
    Cl_in : numpy ndarray
        A numpy array of shape (nmaps, lmax+1) which contains Cl power spectra.
    dl_min : float, optional
        Minimum bin size in terms of multipole number. Default is 10.
    dlbyl : float, optional
        Bin size in terms of fraction of multipole number. Default is 0.4.
    dl_max : float, optional
        Maximum bin size in terms of multipole number. Default is None.
    fmt_nmt : bool, optional
        Format the output for Pymaster (NaMaster) library. Default is False.

    Returns
    -------
    numpy ndarray
        The binned Cl power spectra.
    """
    Cl_in = np.array(Cl_in)

    if Cl_in.ndim > 2:
        raise Exception("ERROR: Upto 2-d Cl arrays supported in form [ndim, lmax+1]")
        
    elif Cl_in.ndim == 2:
        # Assume that Cl_in is [nmaps, lmax+1] in size
        lmax = len(Cl_in[0]) - 1
        nmaps = len(Cl_in)
        Cl_1x2 = np.copy(Cl_in)

        Cl_binned = np.zeros((nmaps, lmax+1))

        for li in range(2, lmax+1) :
            limin = np.maximum((np.floor(np.minimum((1-dlbyl/2)*li, li-(dl_min/2)))), 2)
            limax = np.minimum((np.ceil(np.maximum((1+dlbyl/2)*li, li+(dl_min/2)))), lmax-1)

            if dl_max != None:
                limin = np.maximum(limin, int(np.floor(li - (dl_max/2))))
                limax = np.minimum(limax, int(np.ceil(li + (dl_max/2))))
            # li = li - 2
            # if li < len(leff):
            #     limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 0)
            #     limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), len(leff)-1)

            Cl_binned[:,li] = (np.sum(np.copy(Cl_1x2)[:,limin:limax], axis=1)) / (limax - limin) #)

        del Cl_1x2

        if fmt_nmt:
            Cl_binned = np.reshape(Cl_binned,(nmaps, 1, lmax+1))
    else:
        lmax = len(Cl_in) - 1

        Cl_1x2 = np.copy(np.array(Cl_in))

        # ells = np.arange(lmax+1)
        # mode_factor = 2.*ells + 1. 
        # Cl_1x2 = mode_factor * Cl_1x2

        Cl_binned = np.zeros((lmax+1,))

        for li in range(2, len(Cl_1x2)) :
            limin = np.maximum(int(np.floor(np.minimum((1-dlbyl/2)*li, li-(dl_min/2)))), 2)
            limax = np.minimum(int(np.ceil(np.maximum((1+dlbyl/2)*li, li+(dl_min/2)))), lmax-1)

            if dl_max != None:
                limin = np.maximum(limin, int(np.floor(li - (dl_max/2))))
                limax = np.minimum(limax, int(np.ceil(li + (dl_max/2))))
            # li = li - 2
            # if li < len(leff):
            #     limin = np.maximum(np.int(np.floor(np.minimum(0.8*li, li-5))), 0)
            #     limax = np.minimum(np.int(np.ceil(np.maximum(1.2*li, li+5))), len(leff)-1)
            Cl_binned[li] = (np.sum(Cl_1x2[limin:limax])) / (limax - limin) #) 

        del Cl_1x2 

        if fmt_nmt:
            Cl_binned = np.reshape(Cl_binned,(1,len(Cl_binned)))

    return Cl_binned

def process_alm(alm_in, fwhm_in=None, fwhm_out=None, beam_in=None, beam_out=None, pixwin_in=None, pixwin_out=None, mode='i'):
    """
    This is equivalent to the HEALPix Fortran utility by the same name, used to change the beam and/or pixel window of alms.
    The effective operation is: 
    .. math::
        a^{\\rm out}_{\\ell m} = \\frac{b^{\\rm out}_\\ell p^{\\rm out}_\\ell}{b^{\\rm in}_\\ell p^{\\rm in}_\\ell} a^{\\rm in}_{\\ell m}

    Parameters
    ----------
    alm_in : numpy ndarray
        A 1D or 2D numpy array containing HEALPix maps. The shape of the array should be: ``(nalms, alm_size)`` for
        multiple alms and ``(alm_size,)`` for single alm.
    mode : string, optional
        Determines the choice of beam transfer function. Choices are: ``i`` for intensity-type alms for 
        spin-0/scalar fields (like CMB temperature); ``iqu`` or ``teb`` for ``nalms = 3``; ``e``, ``b`` for
        E- or B-mode alms inputs (accounts for difference in the Gaussian beam definition from intensity);
        ``eb`` or ``teb`` for input of 2 polarized alms but with polarization. Default is ``i``.
    fwhm_in : float, optional
        Full-width at half maximum of the Gaussian beam of the input alm. If ``beam_in`` is also provided, then 
        ``fwhm_in`` is ignored. Default is ``None``.
    fwhm_out : float, optional
        Full-width at half maximum of the Gaussian beam of the output alm. If ``beam_out`` is also provided, then 
        ``fwhm_out`` is ignored. Default is ``None``.
    beam_in : numpy ndarray, optional
        Beam transfer function of the input alm. The shape of the array must be ``(lmax_sht+1,)`` or ``(lmax_sht+1, nalms)``.
        If ``mode`` is ``iqu`` or ``teb``, the shape must be ``(lmax_sht+1, 3)``. Default is ``None``.
    beam_out : numpy ndarray, optional
        Beam transfer function of the output alm. The shape of the array must be ``(lmax_sht+1,)`` or ``(lmax_sht+1, nalms)``.
        If ``mode`` is ``iqu`` or ``teb``, the shape must be ``(lmax_sht+1, 3)``. Default is ``None``.
    pixwin_in : int, optional
        Specifies the ``NSIDE`` of the HEALPix pixel window function to fetch for the input, if applicable. Arbitrary
        pixel window functions are currently not supported. Default is ``None``.
    pixwin_out : int, optional
        Specifies the ``NSIDE`` of the HEALPix pixel window function to fetch for the output, if applicable. Arbitrary
        pixel window functions are currently not supported. Default is ``None``.

    Returns
    -------
    numpy ndarray
        Returns a numpy array for output alms. Shape of output : ``(nalms, alm_size)`` or ``(alm_size,)``.
    """

    alm_in = np.array(alm_in)

    if alm_in.ndim < 2:
        n_alms = 1
        alm_in = np.reshape(alm_in, (1, len(alm_in)))
    elif alm_in.ndim == 2:
        n_alms = alm_in.shape[0]
    else:
        raise Exception("ERROR: The shape alm array is unrecognized. Aborting!")

    if (mode.lower() in ['iqu', 'teb']) and (n_alms != 3):
        raise Exception("ERROR: For IQU/TEB mode 3 alms are to be supplied. Aborting!")


    ALM = hp.Alm()
    lmax = ALM.getlmax(len(alm_in[0]))

    if isinstance(beam_in, (np.ndarray, list, tuple)):
        beam_in = np.array(beam_in)

        if beam_in.ndim == 1:
            nbeams_in = 1.
        else: 
            nbeams_in = beam_in.shape[1]
            if nbeams_in != n_alms:
                raise Exception("ERROR: Either supply same number of beams as alms or supply one to use for all. Aborting!")
                
        if len(beam_in) != lmax+1:
                raise Exception("ERROR: beam_in must have same lmax as alm. Aborting!")
                
        if (mode.lower() in ['iqu', 'teb']) and (nbeams_in != 3):
            raise Exception("ERROR: For IQU/TEB mode 3 input beams are to be supplied. Aborting!")
    else:
        if fwhm_in != None:
            beam_in = hp.gauss_beam(np.deg2rad(fwhm_in / 60.), lmax=lmax, pol=True)[:,:3]
            nbeams_in = 3

            if mode.lower() in ['i', 't']:
                beam_in = beam_in[:,0]
                nbeams_in = 1
            elif mode.lower() in ['e', 'b', 'eb']:
                beam_in = beam_in[:,1]
                nbeams_in = 1
            elif not (mode.lower() in ['iqu','teb']):
                raise Exception("ERROR: Unrecognized mode! Only supported options={t, e, b, eb, i, qu, teb, iqu}. Aborting!")
        else:
            beam_in = np.ones((lmax+1,))
            nbeams_in = 1

            
    if isinstance(beam_out, (np.ndarray, list, tuple)):
        beam_out = np.array(beam_out)
        if beam_out.ndim == 1:
            nbeams_out = 1.
        else: 
            nbeams_out = beam_out.shape[1]
            if nbeams_out != n_alms:
                raise Exception("ERROR: Either supply same number of beams as alms or supply one to use for all. Aborting!")

        if len(beam_out) != lmax+1:
                raise Exception("ERROR: beam_out must have same lmax as alm. Aborting!")

        if (mode.lower() in ['iqu', 'teb']) and (nbeams_out != 3):
            raise Exception("ERROR: For IQU/TEB mode 3 output beams are to be supplied. Aborting!")
    else:
        if fwhm_out != None:
            beam_out = hp.gauss_beam(np.deg2rad(fwhm_out / 60.), lmax=lmax, pol=True)[:,:3]
            nbeams_out = 3

            if mode.lower() in ['i', 't']:
                beam_out = beam_out[:,0]
                nbeams_out = 1
            elif mode.lower() in ['e', 'b', 'eb',]:
                beam_out = beam_out[:,1]
                nbeams_out = 1
            elif not (mode.lower() in ['iqu','teb']):
                raise Exception("ERROR: Unrecognized mode! Only supported options={t, e, b, eb, i, teb, iqu}. Aborting!")
        else:
            beam_out = np.ones((lmax+1,))
            nbeams_out = 1

    # print(n_alms, beam_in.shape, beam_out.shape, alm_in.shape)
    if isinstance(pixwin_in, (int,float)):
        pixwin_in = hp.pixwin(int(pixwin_in), lmax=lmax)
        beam_in *= pixwin_in
    
    if isinstance(pixwin_out, (int,float)):
        pixwin_out = hp.pixwin(int(pixwin_out), lmax=lmax)
        beam_out *= pixwin_out

    if (nbeams_in == 1) and (nbeams_out == 1):
        beam_factor = compute_beam_ratio(beam_in, beam_out)
        alm_out = np.zeros_like(alm_in)

        for i in range(n_alms):
            alm_out[i] = hp.almxfl(alm_in[i], beam_factor)

        del alm_in, beam_factor, beam_in, beam_out

        if n_alms == 1: return alm_out[0]

        return alm_out
    
    if n_alms > 1 and nbeams_in*nbeams_out > 1:
        beam_factor = np.zeros((lmax+1, max(nbeams_in, nbeams_out)))
        if nbeams_in == nbeams_out:
            for ibeam in range(nbeams_in):
                beam_factor[:,ibeam] = compute_beam_ratio(beam_in[:,ibeam], beam_out[:,ibeam])
        elif (nbeams_in > nbeams_out) and (nbeams_out == 1):
            for ibeam in range(nbeams_in):
                beam_factor[:,ibeam] = compute_beam_ratio(beam_in[:,ibeam], beam_out)
        
        # print(beam_factor.shape, beam_factor)
        alm_out = np.zeros_like(alm_in)
        for i in range(n_alms):
            alm_out[i] = hp.almxfl(alm_in[i], beam_factor[:,i])

        return alm_out
    
    else:
        raise Exception("ERROR: Wrong number of beams given. Ensure nbeams_in >= nbeams_out and n_alms >= max(nbeams_in, nbeams_out). Aborting!")
    
    
def change_resolution(map_in, nside_out=None, mode='i', lmax_sht=None, fwhm_in=None, fwhm_out=None, beam_in=None, beam_out=None, pixwin_in=None, pixwin_out=None):
    """
    The ``change_resolution`` function is a map level reconvolution utility. This is a map-level wrapper for 
    the ``process_alm`` function, to change resolution of a HEALPix map via spherical harmonic transforms (SHT). This 
    is particularly useful for changing resolution of polarization maps.

    Parameters
    ----------
    map_in : numpy ndarray
        A 1D or 2D numpy array containing HEALPix maps. The shape of the array should be: ``(nmaps, npix)`` for
        multiple maps and ``(npix,)`` for single map.
    nside_out : int, optional
        Value of HEALPix ``NSIDE`` value for the output map. Default is the same ``NSIDE`` 
        as the input map.
    mode : string, optional
        Determines the type of SHT that is performed on the map. Choices are: ``i`` for intensity-type maps for 
        spin-0/scalar fields (like CMB temperature); ``iqu`` for ``nmap = 3`` and IQU map input; ``e``, ``b`` for
        E- or B-mode scalar map inputs (accounts for difference in the Gaussian beam definition from intensity);
        ``eb`` or ``teb`` for two or three input scalar maps but with polarization. Note: only ``iqu`` option 
        assumes spin-2 fields for SHT. Default is ``i``.
    lmax_sht : int, optional
        ``lmax`` used in the SHT. Default is ``3 * NSIDE - 1`` for ``NSIDE`` of input map.
    fwhm_in : float, optional
        Full-width at half maximum of the Gaussian beam of the input map. If ``beam_in`` is also provided, then 
        ``fwhm_in`` is ignored. Default is ``None``.
    fwhm_out : float, optional
        Full-width at half maximum of the Gaussian beam of the output map. If ``beam_out`` is also provided, then 
        ``fwhm_out`` is ignored. Default is ``None``.
    beam_in : numpy ndarray, optional
        Beam transfer function of the input map. The shape of the array must be ``(lmax_sht+1,)`` or ``(lmax_sht+1, nmaps)``.
        If ``mode`` is ``iqu`` or ``teb``, the shape must be ``(lmax_sht+1, 3)``. Default is ``None``.
    beam_out : numpy ndarray, optional
        Beam transfer function of the output map. The shape of the array must be ``(lmax_sht+1,)`` or ``(lmax_sht+1, nmaps)``.
        If ``mode`` is ``iqu`` or ``teb``, the shape must be ``(lmax_sht+1, 3)``. Default is ``None``.
    pixwin_in : int, optional
        Specifies the ``NSIDE`` of the HEALPix pixel window function to fetch for the input, if applicable. Arbitrary
        pixel window functions are currently not supported. Default is ``None``.
    pixwin_out : int, optional
        Specifies the ``NSIDE`` of the HEALPix pixel window function to fetch for the output, if applicable. Arbitrary
        pixel window functions are currently not supported. Default is ``None``.

    Returns
    -------
    numpy ndarray
        Returns a numpy array for output maps. Shape of output : ``(nmaps, npix_out)`` or ``(npix_out,)``.

    See also
    --------
    ``process_alm``
    """
    map_to_grd = np.array(map_in)

    if map_to_grd.ndim == 1 :
        nside_in = hp.get_nside(map_to_grd)
        nmaps = 1
    else:
        nside_in = hp.get_nside(map_to_grd[0])
        nmaps = len(map_to_grd[:,0])

    if (mode.lower() in ['iqu', 'teb']) and (nmaps != 3):
        raise Exception("ERROR: NMAPS != 3 is wrong for mode TEB/IQU mode.")

    if nside_out == None:
        nside_out = nside_in 

    if lmax_sht == None:
        lmax = 3 * min(nside_in, nside_out) - 1
    else:
        lmax = min(3 * min(nside_in, nside_out) - 1, lmax_sht)

    if (mode.lower() == 'iqu') or (nmaps == 1):
        alms = hp.map2alm(map_to_grd, lmax=lmax, use_weights=True, datapath=datapath)
    if (mode.lower() in ['i', 't', 'e', 'b', 'eb', 'teb']) and (nmaps > 1):
        alms = []
        for i in range(nmaps):
            alms.append(hp.map2alm(map_to_grd[i], lmax=lmax, use_weights=True, datapath=datapath))
    alms = np.array(alms)

    alms_out = process_alm(alms, mode=mode, fwhm_in=fwhm_in, fwhm_out=fwhm_out, beam_in=beam_in, beam_out=beam_out, pixwin_in=pixwin_in, pixwin_out=pixwin_out)
    
    del alms 

    if nmaps == 1:
        maps_out = hp.alm2map(alms_out, nside_out, lmax=lmax, pol=False)
    elif (mode.lower() == 'iqu'):
        maps_out = hp.alm2map(alms_out, nside_out, lmax=lmax, pol=True)
    elif (mode.lower() in ['i', 't', 'e', 'b', 'eb', 'teb']) and (nmaps > 1):
        maps_out = []
        for i in range(nmaps):
            maps_out.append(hp.alm2map(alms_out[i], nside_out, lmax=lmax, pol=True))
    maps_out = np.array(maps_out)

    return maps_out

def mask_udgrade(mask_in, nside_out, cut_val=0.9):
    """
    The ``mask_udgrade`` function does a udgrade operation to a provided mask, with a threshold that specifies the pixels that are part of the mask after the ud_grade operation.

    Parameters
    ----------
    mask_in : numpy ndarray (npix,) or (nmaps,npix)
        Input mask or sequence of input masks which have the same size. 
    nside_out : int
        Value of HEALPix ``NSIDE`` value for the output mask.
    cut_val : float, optional
        Value above which we threshold the udgraded mask to be 1, for values of pixels that are below this value, the output mask is set to 0. 
        The value must be between 0 and 1. Default is ``0.9``.

    Returns
    -------
    numpy ndarray (npix,) or (nmaps,npix)
        Returns the upgraded or degraded mask(s).
    """
    nside_in = hp.get_nside(mask_in)
    if nside_out != nside_in:
        mask_out = hp.ud_grade(mask_in, nside_out)
    else:
        mask_out = np.copy(mask_in)
        
    mask_out[mask_out > cut_val] = 1.
    mask_out[mask_out <= cut_val] = 0.

    return mask_out


def alm_fort2c(alm_in):
    # Assume alm shape to be [lmax, mmax] for nmaps = 1 and [nmaps, lmax, mmax] >= 1

    alm_fort = np.array(alm_in)

    alm_dim = alm_fort.ndim

    if alm_dim == 3:
        nmaps = len(alm_fort[:,0,0])
        lmax = len(alm_fort[0,:,0]) - 1
        mmax = len(alm_fort[0,0,:]) - 1
    elif alm_dim == 2:
        lmax = len(alm_fort[:,0]) - 1
        mmax = len(alm_fort[0,:]) - 1
    else:
        raise Exception("ERROR: Fortran-type alm has wrong dimensions. Only [nmaps, lmax, mmax] or [lmax, mmax] supported")

    ALM = hp.Alm()
    c_alm_size = ALM.getsize(lmax,mmax)
    ls, ms = ALM.getlm(lmax)

    idx_arr = np.arange(c_alm_size)

    if alm_dim == 3:
        alm_c = np.zeros((nmaps, c_alm_size), dtype=np.complex128)
        alm_c[:,idx_arr] = alm_fort[:, ls, ms]
    else:
        alm_c = np.zeros((c_alm_size,), dtype=np.complex128)
        alm_c[idx_arr] = alm_fort[ls, ms]

    return alm_c
    

def alm_c2fort(alm_in):
    # Assume alm shape to be [midx,] for nmaps = 1 and [nmaps, midx] >= 1

    alm_c = np.array(alm_in)

    alm_dim = alm_c.ndim

    if alm_dim == 2:
        nmaps = len(alm_c[:,0]) 
        midx = len(alm_c[0,:])
    elif alm_dim == 1:
        midx = len(alm_c[:])
    else:
        raise Exception("ERROR: C-type alm has wrong dimensions. Only [nmaps, midx] or [midx] supported")
    
    ALM = hp.Alm()
    lmax = ALM.getlmax(midx)
    mmax = lmax 

    idx_arr = np.arange(midx)

    ls, ms = ALM.getlm(lmax, i=idx_arr)

    if alm_dim == 2:
        alm_fort = np.zeros((nmaps, lmax+1, mmax+1), dtype=np.complex128)
        alm_fort[:,ls, ms] = alm_c[:,idx_arr]
    else:
        alm_fort = np.zeros((lmax+1, mmax+1), dtype=np.complex128)
        alm_fort[ls, ms] = alm_c[idx_arr]

    return alm_fort 

def query_dist(nside, vec_center, radius_in_rad, inclusive=True):
    """
    Query the pixels within a given angular distance from a specified direction in a HEALPix map and calculate their distances.

    Parameters
    ----------
    nside : int
        The ``NSIDE`` parameter of the HEALPix map.
    vec_center : ndarray
        A 3-element array representing the Cartesian coordinates of the center vector.
    radius_in_rad : float
        The angular radius in radians within which pixels are queried.
    inclusive : bool, optional
        If True, includes pixels whose centers lie exactly on the radius. Default is True.

    Returns
    -------
    disc_pix : ndarray
        An array of pixel indices within the specified angular distance.
    pix_dist : ndarray
        An array of angular distances (in radians) from the center vector to each pixel in `disc_pix`.
    """

    disc_pix = np.array(hp.query_disc(nside, vec_center, radius_in_rad, inclusive=inclusive))

    vec_center = np.reshape(vec_center, (3,1))
    vec_disc = np.array(hp.pix2vec(nside, disc_pix))
    
    pix_dist = np.arccos(vec_disc.T @ vec_center)[:,0]

    return disc_pix, pix_dist


def angdist(nside1, pixlist1, nside2, pixlist2):
    """
    Calculate the angular distance between a set of HEALPix pixels on a sky map at two different resolutions.

    Parameters
    ----------
    nside1 : int
        The ``NSIDE`` parameter of the first sky map.
    pixlist1 : ndarray of int64
        The list of pixels in the first sky map.
    nside2 : int
        The ``NSIDE`` parameter of the second sky map.
    pixlist2 : ndarray of int64
        The list of pixels in the second sky map.

    Returns
    -------
    ang_dist : ndarray of float32
        The angular distances (in radians) between the pixels in the two sky maps.
    """
    vec_mat1 = np.array(hp.pix2vec(nside1, pixlist1), dtype=np.float32)       
    vec_mat2 = np.array(hp.pix2vec(nside2, pixlist2), dtype=np.float32)

    # print(vec_mat1.T.shape, vec_mat2.shape)

    return np.arccos(vec_mat1.T @ vec_mat2).astype(np.float32)

    
def alm_c_lmaxchanger(lmax_i, lmax_f):
    """
    Adjusts the lmax of healpy spherical harmonic coefficients.
    This function returns the indices corresponding to the adjusted lmax.


    Parameters
    ----------
    lmax_i : int
        The initial maximum multipole order.
    lmax_f : int
        The final maximum multipole order.

    Returns
    -------
    numpy.ndarray
        An array of indices corresponding to perform the size adjustment.
        If ``lmax_i`` is less than ``lmax_f``, the function returns indices for
        the larger-lmax alm array that gets filled by the smaller-lmax alm array. 
        If ``lmax_i`` is greater than ``lmax_f``, the function returns indices selecting
        the smaller-lmax alm elements. 
        If they are equal, it returns indicies that maps the alm array to itself..    
    """

    ALM = hp.Alm()
    if lmax_i < lmax_f:
        cidx_max = ALM.getsize(lmax_i)
        ls, ms = ALM.getlm(lmax_i, np.arange(cidx_max, dtype=np.int64))
        return ALM.getidx(lmax_f, ls, ms)
    elif lmax_i > lmax_f:
        cidx_max = ALM.getsize(lmax_i)
        ls, ms = ALM.getlm(lmax_i, np.arange(cidx_max, dtype=np.int64))

        ms = ms[ls <= lmax_f]
        ls = ls[ls <= lmax_f]
        return ALM.getidx(lmax_i, ls, ms)
    else:
        return np.arange(ALM.getsize(lmax_i), dtype=np.int64)
