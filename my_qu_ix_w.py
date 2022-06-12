# auxiliary function - computing quality index for dynamic programming

def my_qu_ix_w(self, invec, yinwec, par, par_sig_min):
    par_sig_min = gmm_decomp_split_segment.par_sig_min
    invec = gmm_decomp_split_segment.invec
    yinwec = gmm_decomp_split_segment.yinwec
    #invec = invec[:]
    #yinwec = yinwec[:]
    if (invec[len(invec)] - invec[1]) <= (par_sig_min or np.sum(yinwec) <= 1.0e-3):
        wyn = np.inf
    else:
        wwec = yinwec / (np.sum(yinwec))
        wyn1 = (par + np.sqrt(np.sum(np.dot(((invec - np.sum(np.dot(invec, wwec))) ** 2), wwec)))) / (max(invec) - min(invec))
        wyn = wyn1
    return wyn1
