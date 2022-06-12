# auxiliary function for dynamic programming - compute quality index matrix


def dyn_pr_split_w_aux(self, data, ygreki, par, par_sig_min):
    N = len(data)
    # aux_mx
    aux_mx = np.zeros((N, N))
    for kk in range(1, N-1):
        for jj in range(kk+1, N):
            aux_mx[kk, jj] = my_qu_ix_w(data[kk:jj-1], ygreki[kk:jj-1], par, par_sig_min)

    return aux_mx
