# SkyTools

## *This a library of tools for CMB sky data analysis on a sphere. Many of these tools have been ported from the Planck Sky Model.*

#### Installation Instructions

Install inside a conda environment. The current version targets python version **>= 3.9**. You also need `meson` and `pkg-config` installed in the conda environment for the build backend. All three can be installed by doing `conda install <package>`.


Following python packages are required for SkyTools to work: 
1. `astropy` 
2. `healpy (>=1.16.0)`  
3. `numpy` 
4. `scipy`  
5. `joblib`  

To install download the zip of the project or git clone the project. Unzip the folder if downloaded. Then at the root level of the project (where you find the `pyproject.toml` file) do `pip install .` to install to your active conda environment. You can check if the package is installed in the conda environment by doing `conda list | grep skytools`. Try `python -c "import skytools"` in the same conda environment to verify that the package has been installed correctly. 

For SkyTools to work correctly, set an environment variable `SKYTOOLS_DATA` pointing to HEALPix data folder (available with the HEALPix distribution) by adding `export SKYTOOLS_DATA='<path to Healpix data folder>` in your `bashrc` or `bash_profile`.
