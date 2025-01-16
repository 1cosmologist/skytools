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
import numpy as np
""" 
The SkyTools binning utilities module provides useful and frequently used macro funtions 
to bin power spectrum data.
"""

class bins:
    
    def __init__(self, lmax_o, bsz_0=31, fixed_bins=False, loglims=[[1.3]]):

        """
        Initializes an instance of the bins class to bin power spectrum data.

        Parameters
        ----------
        lmax_o : int
            The maximum multipole order for the output.
        bsz_0 : int, optional
            The initial bin size. Default is 30.
        fixed_bins : bool, optional
            If True, uses fixed bin sizes; otherwise, uses logarithmic scaling. Default is False.
        loglims : list of lists, optional
            Specifies scaling and limits for logarithmic bins. Each list should contain one or two elements:
            [scaling] or [scaling, lmax_scale]. Default is [[1.3]].

        Attributes
        ----------
        lmax_o : int
            The maximum multipole order for the output.
        bin_sz : numpy.ndarray
            An array containing the sizes of bins.
        leff : numpy.ndarray
            An array containing the effective multipole moments for each bin.
            
        Notes
        -----
        Set scaling as [scaling, lmax_scale] for each choice. 
        If same scaling wanted for entire ell range set only scaling.
        Thus loglims is a tuple if single or double element lists: 
        [[sca_1, lmax_sca1],[sca_2, lmax_sca2],...] 
        """
        self.lmax_o = lmax_o
        
        i = 0
        bin_sz = []
        # ell_min = [] 
        leff = []
        while True:
            # print(i)
            if i == 0:
                ell_min_d = 2
                ell_max_d = bsz_0
            elif fixed_bins:
                ell_min_d = ell_max_d + 1
                ell_max_d = ell_min_d + bsz_0
            else :
                ell_min_d = ell_max_d + 1
                
                for i in range(len(loglims)):

                    if len(loglims[i]) == 1:
                        ell_max_d = int(np.maximum(ell_min_d + bsz_0, np.ceil(loglims[i][0] * ell_min_d)))
                        break
                    elif len(loglims[i]) == 2:
                        if ell_min_d < loglims[i][1]:
                            ell_max_d = int(np.maximum(ell_min_d + bsz_0, np.ceil(loglims[i][0] * ell_min_d)))
                            break

            if ell_max_d > lmax_o:
                # print(ell_max_d)
                break
            else:
                # ell_min.append(ell_min_d)
                bin_sz.append(int(ell_max_d - ell_min_d + 1))
                leff.append((ell_min_d + ell_max_d) / 2.)
                # print(ell_min[i], ell_max_d, LL[i])
                i = i + 1

        self.bin_sz = np.array(bin_sz)
        # ell_min = np.array(ell_min)
        self.leff = np.array(leff) 


    def binner(self, Dell_in, is_Cell = False):
        """
        Bins the input Dell power spectrum according to the binning scheme defined by the class instance.

        Parameters
        ----------
        Dell_in : numpy ndarray
            The input power spectrum. Can be either D_ell or C_ell.
            For C_ell, set is_Cell = True.
        is_Cell : bool, optional
            If the input power spectrum is C_ell, set this to True.
            Default is False.

        Returns
        -------
        numpy ndarray
            The binned power spectrum.
        """
        ell = np.arange(self.lmax_o+1)
        Dell_factor = ell * (ell + 1.) / 2. / np.pi

        Dell = np.copy(Dell_in[0:self.lmax_o+1])

        if is_Cell :
            Dell = Dell_factor[0:self.lmax_o+1] * Dell    

        Dell_binned = []

        # print(bsz)
        lmin_i = 2
        lmax_i = lmin_i + self.bin_sz[0]

        for i in range(0,len(self.leff)):
            # print(lmin_i, lmax_i)                                               
            dummy = np.sum(Dell[lmin_i:lmax_i])  / (lmax_i - lmin_i)
            
            Dell_binned.append(dummy)  

            if i < len(self.leff)-1:
                lmin_i = lmax_i
                lmax_i = lmin_i + self.bin_sz[i+1]    
            
        del dummy, Dell
        return np.array(Dell_binned)
    
    def ell_eff(self):
        """
        Returns the effective multipole number for each bin.

        Returns
        -------
        numpy ndarray
            The effective multipole number for each bin.
        """
        return self.leff