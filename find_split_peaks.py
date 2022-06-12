#find splitting peaks 


def find_split_peaks(self, peaks, x, y, par_mcv):
    peaks = gmm.peaks
    x = self.x1
    y = baseline_removals.y_base
    par_mcv = gmm.par_mcv
    seg_vec = np.zeros((np.size(peaks[0]), 1))

    for k in range(1, np.size(peaks[0])):
        zakm = np.argwhere(x <= peaks[k + 1, 1] & x >= peaks[k, 1])
        warty = y(zakm)
        miny = min(warty)
        idxm_min_y = warty.index(miny)
        prawzak = zakm(idxm_min_y[1])
        seg_vec[k] = prawzak

    seg_vec.shape = max(x.shape)
    seg_vec_c = np.concatenate((1, seg_vec), axis=0)
    M_B_H = np.zeros(np.size(peaks[0]), 1)
    for kk in range(1, np.size(peaks[0])):
        if y(seg_vec_c[kk]) >= y(seg_vec_c[kk]):
            M_B_H[kk] = y(seg_vec_c[kk]) + 1
        else:
            M_B_H[kk] = y(seg_vec_c[kk+1]) + 1

    max_peaks = np.max(peaks)
    ppe = peaks / M_B_H
    Kini = f_par_mcv(1, self.par_p_l_p, peaks, par_mcv)
    Splitt_v = np.zeros(np.floor(((np.max(x) - x[0]) / x[0] * par_mcv)), 1)
    kspl = 0

    while True:
        if f_par_mcv(Kini, self.par_p_l_p, peaks, par_mcv) == 0:
            break

        Top = Kini + f_par_mcv(Kini, self.par_p_l_p, peaks, par_mcv)
        Zak = np.array(Kini, Top)
        # verify quality condition for the best peak in the range Zak=Kini:Top
        pzak2 = peaks[Zak, 1]

        ppezak = ppe[Zak]
        ixq = np.argwhere(pzak2 > (self.par_p_sens * max_peaks) & ppezak > self.par_q_thr)

        if np.size(ixq) >= 1:
            [mpk, impk] = np.max(ppe(Zak[ixq]))
            kkt = ixq(impk[0])
            kspl += 1
            Splitt_v[kspl] = Zak[kkt]
            Kini = Top + f_par_mcv(Top, self.par_p_j, peaks, par_mcv)

        else:
            Kini = Top + f_par_mcv(Top, 1, peaks, par_mcv)

    Splitt_v = np.rane(1, kspl)
    return np.array(Splitt_v, seg_vec_c)
