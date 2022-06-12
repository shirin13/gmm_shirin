# auxiliary function find reasonable ranges for a group of Gaussian components

def find_ranges(self, mu_v, sig_v):
    K = len(mu_v)
    mzlv = np.zeros((K, 1))
    mzpv = np.zeros((K, 1))

    for kk in range(1, K):
        mzlv[kk] = mu_v[kk] - np.multiply(3, sig_v[kk])
        mzpv[kk] = mu_v[kk] + np.multiply(3, sig_v[kk])

    mzlow = min(mzlv)
    mzhigh = max(mzpv)
    return (mzlow, mzhigh)
