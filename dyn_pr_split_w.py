#   main dynamic programming algorithm for computing initial conditions for EM iterations

def dyn_pr_split_w(self, data, ygreki, k_gr, aux_mx, par, par_sig_min):
    par_sig_min = gmm_decomp_split_segment.par_sig_min
    # initialize
    Q = np.zeros((1, K_gr))
    N = len(data)
    p_opt_idx = np.zeros((1, N))
    p_aux = np.zeros((1, N))
    opt_pals = np.zeros((k_gr, N))
    for kk in range(1, N):
        p_opt_idx[kk] = my_qu_ix_w(data[kk:N], ygreki[kk:N], par, par_sig_min)

    # aux_mx - already computed

    # iterate
    for kster in range(1, k_gr):
        for kk in range(1, N - kster):
            for jj in range(kk + 1, N - kster + 1):
                p_aux[jj] = aux_mx[kk, jj] + p_opt_idx[jj]

        mm, ix = min(p_aux[kk + 1:N - kster + 1])
        p_opt_idx[kk] = mm
        opt_pals[kster, kk] = kk + ix[1]

    Q[kster] = p_opt_idx[1]

    # restore optimal decisions
    opt_part = np.zeros((1, k_gr))
    opt_part[1] = opt_pals[k_gr, 1]
    for kster in range(k_gr-1, 1, -1):
        opt_part[k_gr - kster + 1] = opt_pals[kster, opt_part[k_gr - kster]]

    return (Q, opt_part)
