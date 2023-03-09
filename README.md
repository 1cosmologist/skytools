# SkyTools

##### _This a library of tools for CMB sky data analysis on a sphere. Many of these tools have been ported from the Planck Sky Model._

#### Installation Instructions

There are currently no stable releases as the package is in alpha phase. Install inside a conda environment. The current version targets python version >= 3.9. Following python packages are required for SkyTools to work: `astropy`, `healpy (>=1.16.0)`, `numpy`, `scipy` and `joblib`. Additionally ensure `pkg-config` needs to installed in the conda environment (`conda install pkg-config`). 

To install download the zip of the project or git clone the project. Unzip the folder if downloaded. Then at the root level of the project (where you find the `pyproject.toml` file) do `pip install .` to install to your active conda environment. You can check if the package is installed in the conda environment by doing `conda list | grep skytools`. Try `python -c "import skytools"` in the same conda environment to verify that the package has been installed correctly. 
