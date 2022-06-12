# auxiliary function for find_split_peaks

def f_par_mcv(self, kini, fst, peaks, par_mcv):
    # kini = peaks[0]# kini - initial position of the peak counter
    # fst - number of forward steps intended to make
    peaks = gmm.peaks
    par_mcv = gmm.par_mcv
    fst_new = 0
    if kini + fst > np.size(peaks[0]):
        fst_new = 0
    else:
        fst_new = fst
        while True:
            if peaks[kini+fst_new, 1] - peaks[kini, 1] > fst * par_mcv * peaks[kini, 1]:
                break
            else:
                fst_new += 1
                if kini + fst_new > np.size(peaks[0]):
                    fst_new = 0
                    break
        return fst_new
