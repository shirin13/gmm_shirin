#       gmm decomposition of a segment based on dynamic programming initialization
def gmm_decomp_segment(self, x, y_bas, ww_mx_1, mu_mx_1, sig_mx_1, peaks, splitt_v, par_mcv, fr_no):
    # buffers
    x = self.x
    y_base = baseline_removals.y_base
    splitt_v = find_split_peaks.Splitt_v
    peaks = gmm.peaks
    par_mcv = gmm.par_mcv
    fr_no = 1
    ww_dec = np.zeros((1, self.buf_size_seg_par))
    mu_dec = np.zeros((1, self.buf_size_seg_par))
    sig_dec = np.zeros((1, self.buf_size_seg_par))

    invec = np.zeros()
    yinwec = np.zeros()

    # assign fragment number to ksp
    KSP = len(splitt_v)
    ksp = fr_no

    # find separated fragments
    if ksp > 0:
        mu_l = mu_mx_1[ksp, :]
        ww_l = ww_mx_1[ksp, :]
        sig_l = sig_mx_1[ksp, :]
        ktr = max((np.where(ww_l > 0)))
        mu_l = mu_l[1: ktr]
        ww_l = ww_l[1: ktr]
        sig_l = sig_l[1: ktr]
    else:
        mu_l = []
        ww_l = []
        sig_l = []

    if ksp < KSP:
        mu_p = mu_mx_1[ksp+1, :]
        ww_p = ww_mx_1[ksp+1, :]
        sig_p = sig_mx_1[ksp+1, :]
        ktr = max((np.argwhere(ww_p > 0)))
        mu_p = mu_p[1:ktr]
        ww_p = ww_p[1:ktr]
        sig_p = sig_p[1:ktr]
    else:
        mu_p = []
        ww_p = []
        sig_p = []

    x_out, y_out = find_segment(ksp, peaks, splitt_v, x, y_bas, mu_l, ww_l, sig_l, mu_p, ww_p, sig_p)

    x_out = x_out[:]
    y_out = y_out[:]
    if len(x_out) > 300:
        dx = (x_out[len(x_out)] - x_out[1]) / 200
        x_out_bb = np.arange(x_out[1], x_out[len(x_out)], dx)
        x_out_b = x_out_bb[1:200] + 0.5 * dx
        y_out_b, yb = bindata(y_out, x_out, x_out_bb)
        ixnn = np.argwhere(~ math.isnan(y_out_b))
        y_out_b = y_out_b[ixnn]
        x_out_b = x_out_b[ixnn]
        y_out_b = y_out_b[:]
        x_out_b = x_out_b[:]
    else:
        y_out_b = y_out
        x_out_b = x_out

    # find appropriate gmm model for the segment

    quamin = np.inf
    N = len(x_out)
    Nb = len(x_out_b)
    par_sig_min = self.res_par_2 * self.par_mcv * np.mean(x_out)

    while True:
        if len(x_out) < 3:
            continue

        else:
            KSmin = min(1, (np.floor((x_out[N] - x_out[1]) / par_sig_min) - 1))
            if KSmin <= 0:
                wwec = y_out / (np.sum(y_out))
                mu_est = np.sum(np.multiply(x_out, wwec))
                pp_est = 1
                sig_est = np.sqrt(np.sum(((x_out - np.multiply(np.power(np.sum(np.multiply(x_out, wwec))), 2), wwec))))
                qua, scale = qua_scal(x_out, y_out, pp_est, mu_est, sig_est)
            else:
                KS = KSmin
                # penetration - how far are we searching for minimum
                par_penet = min(np.array(self.penet_par_2, np.array(np.floor((x_out[N] - x_out[1]) / par_sig_min), np.floor(len(x_out) / 4))))
                kpen = 0

                # name=['dane_nr' num2str(ksp)]
                # save(name, 'mz_out', 'y_out', 'mz_out_b', 'y_out_b', 'QFPAR', 'PAR_sig_min', 'PAR_penet');
                aux_mx = dyn_pr_split_w_aux(x_out_b, y_out_b, self.qfpar, par_sig_min)
                while KS < self.buf_size_seg_par:
                    KS = KS + 1
                    kpen = kpen + 1

                    if KS > KSmin + 1 and KS >= len(x_out) / 2:
                        break

                    Q, opt_part = dyn_pr_split_w(x_out_b, y_out_b, KS-1, aux_mx, self.qfpar, par_sig_min)
                    part_cl = np.array(1, opt_part, Nb+1)

                    # set initial cond
                    pp_ini = np.zeros((1, KS))
                    mu_ini = np.zeros((1, KS))
                    sig_ini = np.zeros((1, KS))
                    for kkps in range(1, KS):
                        invec = x_out_b[part_cl[kkps]:part_cl[kkps+1]-1]
                        yinwec = y_out_b[part_cl[kkps]:part_cl[kkps+1]-1]
                        wwec = yinwec / (np.sum(yinwec))
                        pp_ini[kkps] = np.sum(yinwec) / np.sum(y_out)
                        mu_ini[kkps] = np.sum(np.multiply(invec, wwec))
                        # sig_ini[(kkps)]=sqrt(sum(((invec-sum(invec.*wwec')).^2).*wwec'))
                        sig_ini[kkps] = 0.5 * (max(invec) - min(invec))

                    pp_est, mu_est, sig_est, TIC, l_lik, bic = my_EM_iter(x_out, y_out, pp_ini, mu_ini, sig_ini, 0, par_sig_min, em_tol)

                    # compute quality indices of gmm model of the fragment
                    qua, scale = qua_scal(x_out, y_out, pp_est, mu_est, sig_est)
                    quatest = qua + self.prec_par_2 * KS

                    if (quatest < quamin):
                        quamin = quatest
                        pp_min = pp_est
                        mu_min = mu_est
                        sig_min = sig_est
                        scale_min = scale

                    elif (quatest > quamin) and (kpen > self.par_penet):
                        pp_est = pp_min
                        mu_est = mu_min
                        sig_est = sig_min
                        scale = scale_min
                        break

            if self.draw == 1:
                # figure(2)
                plt.subplot(3, 1, 3)
                ok = plot_res_new_scale(x_out, y_out, pp_est * scale, mu_est, sig_est)
                plt.xlabel(str(fr_no),  str(len(pp_est)))

            ww_o = pp_est * scale
            mu_o = mu_est
            sig_o = sig_est

            for kkpick in range(1, len(ww_o)):
                mu_dec[kkpick] = mu_o[kkpick]
                ww_dec[kkpick] = ww_o[kkpick]
                sig_dec[kkpick] = sig_o[kkpick]

    return (ww_dec, mu_dec, sig_dec)
