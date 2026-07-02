import healpy as hp
import numpy as np

from . import needlet as nlt

def super_filter(map_in, nside_sup, operation):
    """Aggregate a HEALPix map to a super-pixel grid and apply a reduction.

    Parameters
    ----------
    map_in : array-like
        Input HEALPix map in RING ordering.
    nside_sup : int
        Target (coarser) NSIDE for super-pixel aggregation.
    operation : callable
        Reduction function applied along axis 1 of grouped NESTED pixels
        (e.g., ``np.mean``, ``np.sum``, ``np.max``).

    Returns
    -------
    numpy.ndarray
        Filtered map at ``nside_sup`` in RING ordering.
    """
    nside_in = hp.get_nside(map_in)
    order_diff = int(np.log2(nside_in / nside_sup))
    npix_sup = hp.nside2npix(nside_sup)

    nest_slice = 1
    for ord in range(order_diff):
        nest_slice += 4**ord * 3

    nest_slice = np.int32(nest_slice)
    
    map_nest_super = np.reshape(hp.reorder(map_in, r2n=True), (npix_sup, nest_slice))
    operation_super = hp.reorder(operation(map_nest_super, axis=1), n2r=True)

    return operation_super

def harmonic_low_pass_filter(ell_one_cut, ell_zero_cut, lmax):
    """Build a harmonic low-pass transfer function with cosine transition.

    Parameters
    ----------
    ell_one_cut : int or float
        Multipole where the response is fully one.
    ell_zero_cut : int or float
        Multipole where the response reaches zero.
    lmax : int
        Maximum multipole included in the filter.

    Returns
    -------
    numpy.ndarray
        Low-pass filter values for multipoles ``0..lmax``.
    """
    bandcenters = np.array([ell_one_cut, ell_zero_cut, lmax])
    cos_bands = nlt.cosine_bands(bandcenters)

    return cos_bands[:, 0]

def harmonic_high_pass_filter(ell_zero_cut, ell_one_cut, lmax):
    """Build a harmonic high-pass transfer function with cosine transition.

    Parameters
    ----------
    ell_zero_cut : int or float
        Multipole where the response is zero.
    ell_one_cut : int or float
        Multipole where the response reaches one.
    lmax : int
        Maximum multipole included in the filter.

    Returns
    -------
    numpy.ndarray
        High-pass filter values for multipoles ``0..lmax``.
    """
    bandcenters = np.array([ell_zero_cut, ell_one_cut, lmax])
    cos_bands = nlt.cosine_bands(bandcenters)
    cos_bands[:,1] = cos_bands[:,1]**2 + cos_bands[:,2]**2
    cos_bands[:,1][cos_bands[:,1] > 0.] = np.sqrt(cos_bands[:,1][cos_bands[:,1] > 0.])
    
    return cos_bands[:,1]

def harmonic_bandpass_filter(ell_min_zero, ell_min_one, ell_max_one, ell_max_zero, lmax):
    """Build a harmonic band-pass transfer function with cosine transition.

    Parameters
    ----------
    ell_min_zero : int or float
        Lower multipole where the response is zero.
    ell_min_one : int or float
        Lower multipole where the response reaches one.
    ell_max_one : int or float
        Upper multipole where the response starts leaving one.
    ell_max_zero : int or float
        Upper multipole where the response returns to zero.
    lmax : int
        Maximum multipole included in the filter.

    Returns
    -------
    numpy.ndarray
        Band-pass filter values for multipoles ``0..lmax``.
    """
    bandcenters = np.array([ell_min_zero, ell_min_one, ell_max_one, ell_max_zero, lmax])
    cos_bands = nlt.cosine_bands(bandcenters)
    filter = cos_bands[:, 1:4].sum(axis=1)
    filter[filter > 1] = 1. 
    
    return filter