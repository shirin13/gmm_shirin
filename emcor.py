# emergency correction function launched in the case of (rare) overlap between splitters
def emcor(self, ksp, mu_mx_1, ww_mx_1, sig_mx_1, x, y_bas, splitt_v, peaks, par_mcv):

    x = self.x
    y_base = baseline_removals.y_base
    splitt_v = find_split_peaks.Splitt_v
    peaks = gmm.peaks
    par_mcv = gmm.par_mcv
    KSP = len(splitt_v)

    ww_a_l_new = 0 * ww_mx_1[1, :]
    mu_a_l_new = 0 * ww_mx_1[1, :]
    sig_a_l_new = 0 * ww_mx_1[1, :]

    ww_a_p_new = 0 * ww_mx_1[1, :]
    mu_a_p_new = 0 * ww_mx_1[1, :]
    sig_a_p_new = 0 * ww_mx_1[1, :]

    if ksp > 0:
        mu_l = mu_mx_1[ksp, :]
        ww_l = ww_mx_1[ksp, :]
        sig_l = sig_mx_1[ksp, :]
        ixp = max(np.argwhere(ww_l > 0))

        if len(ixp) == 0:
            KSll = 0
            xll = x[1]
            mu_l = []
            ww_l = []
            sig_l = []
        else:
            mu_l = mu_l[1: ixp]
            ww_l = ww_l[1: ixp]
            sig_l = sig_l[1: ixp]
            KSll = len(ww_l)
            xll, xlp = find_ranges(mu_l, sig_l)
    else:
        KSll = 0
        xll = x[1]
        mu_l = []
        ww_l = []
        sig_l = []

    if ksp < KSP:
        mu_p = mu_mx_1[ksp + 1, :]
        ww_p = ww_mx_1[ksp + 1, :]
        sig_p = sig_mx_1[ksp + 1, :]
        ixp = max(np.argwhere(ww_p > 0))
        mu_p = mu_p[1: ixp]
        ww_p = ww_p[1: ixp]
        sig_p = sig_p[1: ixp]
        KSpp = len(ww_p)
        xlp, xpp = find_ranges(mu_p, sig_p)
    else:
        KSpp = 0
        xpp = x[-1]
        mu_p = []
        ww_p = []
        sig_p = []

    idm = np.argwhere(x >= xll & x <= xpp)
    x_o = x[idm]
    y_o = y_bas[idm]
    pyl = 0 * x_o
    pyp = 0 * x_o

    if ksp > 0:
        for kks in range(1, KSll):
            norm_pdf = 1/(sig_l[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * (x_o - mu_l[kks]) ** 2 / 2 * sig_l[kks] ** 2)
            pyl = pyl + ww_l[kks] * norm_pdf
        idl = np.argwhere(x_o >= xll & x_o <= peaks(splitt_v[ksp], 1))

        pylpds = pyl[idl]
        y_o[idl] = pylpds

    if ksp < KSP:
        for kks in range(1, KSpp):
            norm_pdf = 1/(sig_p[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * (x_o - mu_p[kks]) ** 2 / 2 * sig_p[kks] ** 2)
            pyp = pyp + ww_p[kks] * norm_pdf

        idp = np.argwhere(x_o >= peaks(splitt_v[(ksp + 1)], 1) & x_o <= xpp)
        pyppds = pyp[idp]
        y_o[idp] = pyppds

    KS = KSll + KSpp

    pp_ini = np.array(ww_l, ww_p) / np.sum(np.array(ww_l, ww_p))
    mu_ini = np.array(mu_l, mu_p)
    sig_ini = np.array(sig_l, sig_p)

    par_sig_min = self.res_par_2 * par_mcv * np.mean(x_o)
    pp_est, mu_est, sig_est = my_EM_iter(x_o, y_o, pp_ini, mu_ini, sig_ini, 0, par_sig_min, em_tol)
    KS = len(pp_est)
    KSpp = KS - KSll

    _, scale = qua_scal(x_o, y_o, pp_est, mu_est, sig_est)
    ww_est = pp_est * scale

    if ksp > 0:
        ww_a_l_new[1: KSll] = ww_est[1: KSll]
        mu_a_l_new[1: KSll] = mu_est[1: KSll]
        sig_a_l_new[1: KSll] = sig_est[1: KSll]

    if ksp < KSP:
        ww_a_p_new[1: KSpp] = ww_est[KSll + 1: KS]
        mu_a_p_new[1: KSpp] = mu_est[KSll + 1: KS]
        sig_a_p_new[1: KSpp] = sig_est[KSll + 1: KS]

    # ok = plot_res_new_scale(x_o, y_o, ww_est, mu_est, sig_est)

    return (ww_a_l_new, mu_a_l_new, sig_a_l_new, ww_a_p_new, mu_a_p_new, sig_a_p_new)
