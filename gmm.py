#decomposition of the proteomic spectrum into Gaussian mixture model

def gmm(self, x, y_base):
    x = self.x1
    y_base = baseline_removals.y_base
    seg_vec_c = find_split_peaks.seg_vec_c

    splitt_v = find_split_peaks.Splitt_v
    # find peaks
    # peaks at which point, the value of the peak
    peaks, _ = find_peaks(x, height=0)
    results_half = peak_widths(x, peaks, rel_height=0.5)
    res_width = results_half[0]

    # estimate resolution
    par_mcv = self.res_par * np.mean((res_width[:, 2] - res_width[:, 1])/peaks[:, 1])

    # find good quality splitting peaks
    splitt_v, seg_vec_c = find_split_peaks(peaks, x, y_base, par_mcv)
    KSP = np.max(splitt_v.shape)  # number of plitting peaks
    # buffers for splitter parameters
    mu_mx_1 = np.zeros((KSP, self.buf_size_split_par))
    ww_mx_1 = np.zeros((KSP, self.buf_size_split_par))
    sig_mx_1 = np.zeros((KSP, self.buf_size_split_par))

    for kk in range(1, KSP):
        #  disp(['split Progress: ' str[kk] ' of ' str[KSP])
        # Gaussian mixture decomposition of the splitting segment
        ww_pick, mu_pick, sig_pick = gmm_decomp_split_segment(x, y_bas, splitt_v, seg_vec_c, peaks, par_mcv, kk)
        mu_mx_1[kk, :] = mu_pick
        ww_mx_1[kk, :] = ww_pick
        sig_mx_1[kk, :] = sig_pick

     # PHASE 2 - GMM DECOMPOSITIONS OF SEGMENTS

     # buffers for segments decompositions paramteres
    KSP1 = KSP + 1
    mu_mx_2 = np.zeros((KSP1, self.buf_size_seg_Par))
    ww_mx_2 = np.zeros((KSP1, self.buf_size_seg_Par))
    sig_mx_2 = np.zeros((KSP1, self.buf_size_seg_Par))
    for ksp in range(1, KSP1):
     #    disp(['Seg Progress: ' num2str(ksp) ' of ' num2str(KSP1)])
        ww_out, mu_out, sig_out = gmm_decomp_segment(x, y_bas, ww_mx_1, mu_mx_1, sig_mx_1, peaks, splitt_v, par_mcv, ksp - 1)
        mu_mx_2[ksp, :] = mu_out
        ww_mx_2[ksp, :] = ww_out
        sig_mx_2[ksp, :] = sig_out

     # AGGREGATION
     # aggregate components computed in PHASE 1 and in PHASE 2

    NUMPICK = np.size(peaks[0])
    mu_vec = np.zeros((7 * NUMPICK, 1))
    ww_vec = np.zeros((7 * NUMPICK, 1))
    sig_vec = np.zeros((7 * NUMPICK, 1))

    kacum = 0

    for ksp in range(1, KSP + 1):
        mu_a = mu_mx_2[ksp, :]
        ww_a = ww_mx_2[ksp, :]
        sig_a = sig_mx_2[ksp, :]
        ixp = max(np.argwhere(ww_a > 0))
        if len(ixp) == 0:
            # emergency correction in the case of detected ovelap between splitters
            ww_a_l_new, mu_a_l_new, sig_a_l_new, ww_a_p_new, mu_a_p_new, sig_a_p_new = emcor(ksp - 1, mu_mx_1, ww_mx_1, sig_mx_1, x, y_bas, splitt_v, peaks, par_mcv)
            if ksp > 1:
                ww_mx_1[ksp - 1, :] = ww_a_l_new
                mu_mx_1[ksp - 1, :] = mu_a_l_new
                sig_mx_1[ksp - 1, :] = sig_a_l_new

            if ksp < KSP + 1:
                ww_mx_1[ksp, :] = ww_a_p_new
                mu_mx_1[ksp, :] = mu_a_p_new
                sig_mx_1[ksp, :] = sig_a_p_new

        else:
            mu_a = mu_a[1: ixp]
            ww_a = ww_a[1: ixp]
            sig_a = sig_a[1: ixp]
            for kk in range(1, len(ww_a)):
                kacum = + 1
                mu_vec[kacum] = mu_a[kk]
                ww_vec[kacum] = ww_a[kk]
                sig_vec[kacum] = sig_a[kk]

    for ksp in range(1, KSP):
        mu_a = mu_mx_1[ksp, :]
        ww_a = ww_mx_1[ksp, :]
        sig_a = sig_mx_1[ksp, :]
        ixp = max(np.argwhere(ww_a > 0))
        mu_a = mu_a[1: ixp]
        ww_a = ww_a[1: ixp]
        sig_a = sig_a[1: ixp]

        for kk in range(1, len(ww_a)):
            kacum = + 1
            mu_vec[kacum] = mu_a[kk]
            ww_vec[kacum] = ww_a[kk]
            sig_vec[kacum] = sig_a[kk]

    mu_vec = mu_vec[1: kacum]
    ww_vec = ww_vec[1: kacum]
    sig_vec = sig_vec[1: kacum]

    mu_sort, muixs = np.sort(mu_vec)
    mu_gmm = mu_sort
    ww_gmm = ww_vec[muixs]
    sig_gmm = sig_vec[muixs]

    return (ww_gmm, mu_gmm, sig_gmm)
