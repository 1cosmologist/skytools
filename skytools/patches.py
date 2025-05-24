import numpy as np
from astropy.io import fits
import healpy as hp

def write_IQU_patchmap_fits(bands, x, y, iqu_patch, outfile, unit=''):
    """
    Writes a single IQU patch map and the needlet band filters to a FITS file.

    Parameters
    ----------
    bands : numpy.ndarray
        2D array of shape (lmax+1, nbands) containing the needlet band filters.
    x : int
        Number of pixels in x direction.
    y : int
        Number of pixels in y direction.
    iqu_patch : numpy.ndarray
        3D array of shape (3, x, y) for I, Q, U components of the patch map.
    outfile : str
        Path to output FITS file.
    unit : str, optional
        Units to store in each map column header.
    """
    assert iqu_patch.shape == (3, x, y), f"Expected shape (3, {x}, {y}), got {iqu_patch.shape}"

    # Primary HDU to store bandpass filters
    hdu_P = fits.PrimaryHDU(bands)
    hdulist = [hdu_P]

    comp_names = ['I', 'Q', 'U']
    cols = []
    for j in range(3):
        flat_array = iqu_patch[j].flatten()
        col = fits.Column(name=comp_names[j], format='D', unit=unit, array=flat_array)
        cols.append(col)

    tbhdu = fits.BinTableHDU.from_columns(cols)
    hdulist.append(tbhdu)

    fits.HDUList(hdulist).writeto(outfile, overwrite=True)

def read_IQU_patchmap_fits(filename, x, y):
    """
    Reads IQU patch map and needlet band filters from a FITS file.

    Parameters
    ----------
    filename : str
        Path to the FITS file written by `write_IQU_patchmap_fits`.

    Returns
    -------
    bands : numpy.ndarray
        Needlet band filters (shape: lmax+1, nbands)
    x : int
        Number of pixels in x direction
    y : int
        Number of pixels in y direction
    iqu_patch : numpy.ndarray
        3D array of shape (3, x, y) for I, Q, U patch map.
    """
    with fits.open(filename) as hdul:
        bands = hdul[0].data
        table_data = hdul[1].data

        # Extract and reshape the IQU components
        iqu_patch = np.zeros((3, x, y))
        for i, comp in enumerate(['I', 'Q', 'U']):
            flat = table_data[comp]
            assert flat.size == x * y, f"Expected {x}x{y} for {comp}, got {flat.size}"
            iqu_patch[i] = flat.reshape((x, y))

    return bands, iqu_patch