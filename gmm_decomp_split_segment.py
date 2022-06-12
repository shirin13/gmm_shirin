# gmm decomposition of splitter segment based on dynamic programming initialization for splitting segments


def gmm_decomp_split_segment(self, x, y_base, splitt_v, seg_vec_c, peaks, par_mcv, sp_no):
    x = self.x
    y_base = baseline_removals.y_base
    splitt_v = find_split_peaks.Splitt_v
    seg_vec_c = find_split_peaks.seg_vec_c
    peaks = gmm.peaks
    par_mcv = gmm.par_mcv
    sp_no = 1

    invec = np.zeros()
    yinwec = np.zeros()

    # input
    # mz,y_bas - spectrum
    # splitt_v, seg_vec_c - list of splitting peaks and segment bounds computed by find_split_peaks
    # peaks ,par_mcv - peaks,
    # sp_no - number of splitting segment

    #  find appropriate gmm model for the splitting segment no Sp_No
    # buffers
    ww_pick = np.zeros((1, self.buf_size_split_par))
    mu_pick = np.zeros((1, self.buf_size_split_par))
    sig_pick = np.zeros((1, self.buf_size_split_par))

    # un-binned data
    x_out, y_out = find_split_segment(splitt_v[sp_no], x, y_base, seg_vec_c, peaks, par_mcv)
    x_out = x_out[:]
    y_out = y_out[:]
    # bin if necessary
    if len(x_out) > 300:
        dx = (x_out(len(x_out)) - x_out[0]) / 200
        x_out_bb = np.arange((x_out(1), x_out(len(x_out)), dx))
        x_out_b = x_out_bb[1:200] + 0.5 * dx
        [y_out_b, yb] = bindata(y_out, x_out, x_out_bb)
        ixnn = np.argwhere(math.isnan(y_out_b))
        y_out_b = y_out_b[ixnn]
        x_out_b = x_out_b[ixnn]
        y_out_b = y_out_b[:]
        x_out_b = x_out_b[:]
    else:
        y_out_b = y_out
        x_out_b = x_out

    N = len(x_out)
    Nb = len(x_out_b)
    quamin = np.inf
    par_sig_min = self.res_par_2 * par_mcv * np.mean(x_out)
    KSmin = min(2, (np.floor((x_out[N] - x_out[1]) / par_sig_min) - 1))
    if KSmin <= 0:
        wwec = y_out / (np.sum(y_out))
        mu_est = np.sum(np.dot(x_out, wwec))
        pp_est = 1
        sig_est = np.sqrt(np.sum(np.dot(((x_out - mu_est) ** 2), wwec)))
        qua, scale = qua_scal(x_out, y_out, pp_est, mu_est, sig_est)
    else:

        KS = KSmin
        par_penet = min([self.penet_par_1, np.floor((x_out[N] - x_out[1]) / par_sig_min)])
        kpen = 0
        Q = 0

        aux_mx = dyn_pr_split_w_aux(x_out_b, y_out_b, self.qfpar, par_sig_min)
        while KS <= 2 * (KSmin + par_penet):
            KS = KS + 1
            kpen = kpen + 1

            Q, opt_part = dyn_pr_split_w(x_out_b, y_out_b, KS - 1, aux_mx, self.qfpar, par_sig_min)
            part_cl = np.array(1, opt_part, Nb + 1)

            # set initial cond
            pp_ini = np.zeros((1, KS))
            mu_ini = np.zeros((1, KS))
            sig_ini = np.zeros((1, KS))
            for kkps in range(1, KS):
                invec = x_out_b[part_cl[kkps]: part_cl[kkps + 1] - 1]
                yinwec = y_out_b[part_cl[kkps]: part_cl[kkps + 1] - 1]
                wwec = yinwec/(np.sum(yinwec))
                pp_ini[kkps] = np.sum(yinwec)/np.sum(y_out_b)
                mu_ini[kkps] = np.sum(np.multiply(invec, wwec))
                # sig_ini(kkps) = np.sqrt(sum(((invec-sum(invec.*wwec')).^2).*wwec'))
                sig_ini[kkps] = 0.5 * (max(invec) - min(invec))

            pp_est, mu_est, sig_est, TIC, l_lik, bic = my_EM_iter(x_out, y_out, pp_ini, mu_ini, sig_ini, 0, par_sig_min, em_tol)

            # compute quality indices and scale of gmm model of the fragment
            qua, scale = qua_scal(x_out, y_out, pp_est, mu_est, sig_est)
            quatest = qua + self.prec_par_1 * KS

            if (quatest < quamin):
                quamin = quatest
                pp_min = pp_est
                mu_min = mu_est
                sig_min = sig_est
                scale_min = scale

            if (quatest > quamin) and (kpen > par_penet):
                pp_est = pp_min
                mu_est = mu_min
                sig_est = sig_min
                scale = scale_min
                break

     # pick and store results
    dist = np.abs(np.divide((mu_est - peaks[splitt_v[sp_no], 1]), sig_est))
    ixf = np.argwhere(dist <= 3)
    if len(ixf) == 0:
        tmp = min(dist)
        ixf = dist.index(min(dist))
        ixnf = np.argwhere(dist > tmp)
    else:
        ixnf = np.argwhere(dist > 3)

    mu_p = mu_est[ixf]
    ww_p = scale * pp_est[ixf]
    sig_p = sig_est[ixf]

    mu_t = mu_est[ixnf]
    ww_t = scale * pp_est[ixnf]
    sig_t = sig_est[ixnf]

    inn = np.argwhere(mu_t < max(mu_p) & mu_t > min(mu_p))
    mu_tp = mu_t[inn]
    ww_tp = ww_t[inn]
    sig_tp = sig_t[inn]

    mu_pp = np.array(mu_p, mu_tp)
    ww_pp = np.array(ww_p, ww_tp)
    sig_pp = np.array(sig_p, sig_tp)

    for kkpick in range(1, len(ww_pp)):
        mu_pick[kkpick] = mu_pp[kkpick]
        ww_pick[kkpick] = ww_pp[kkpick]
        sig_pick[kkpick] = sig_pp[kkpick]

    # plots
    if self.draw == 1:
        plt.subplot(2, 1, 1)
        plt.plot(x_out, y_out, 'k')
        plt.plot(peaks[splitt_v[sp_no], 1], peaks[splitt_v[sp_no], 1], np.array(0, max(y_out)), 'r')
        plt.ylabel('y (no. of counts)')
        #plt.title('Splitter segment: ' str(sp_no))
        plt.subplot(2, 1, 2)
        ww_est = scale * pp_est
        ok = plot_gmm(x_out, y_out, ww_est, mu_est, sig_est)
        ok = fill_red(ww_pp, mu_pp, sig_pp)
        #plt.title('Splitter: ' str(sp_no))

    return (ww_pick,mu_pick,sig_pick)
