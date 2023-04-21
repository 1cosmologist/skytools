import numpy as np

class bins:
    
    def __init__(self, lmax_o, bsz_0=30, fixed_bins=False, loglims=[[1.3]]):
# 
# Set scaling as [scaling, lmax_scale] for each choice. 
# If same scaling wanted for entire ell range set only scaling.
# Thus loglims is a tuple if signgle or double element lists: 
# [[sca_1, lmax_sca1],[sca_2, lmax_sca2],...] 
#

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
                        ell_max_d = np.int(np.maximum(ell_min_d + bsz_0, np.ceil(loglims[i][0] * ell_min_d)))
                        break
                    elif len(loglims[i]) == 2:
                        if ell_min_d < loglims[i][1]:
                            ell_max_d = np.int(np.maximum(ell_min_d + bsz_0, np.ceil(loglims[i][0] * ell_min_d)))
                            break

            if ell_max_d > lmax_o:
                # print(ell_max_d)
                break
            else:
                # ell_min.append(ell_min_d)
                bin_sz.append(np.int(ell_max_d - ell_min_d + 1))
                leff.append((ell_min_d + ell_max_d) / 2.)
                # print(ell_min[i], ell_max_d, LL[i])
                i = i + 1

        self.bin_sz = np.array(bin_sz)
        # ell_min = np.array(ell_min)
        self.leff = np.array(leff) 


    def binner(self, Dell_in, is_Cell = False):
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
        return self.leff