# auxiliary function for filling splitter components in red

def fill_red(self, ww_v, mu_v, sig_v):
    KS = len(ww_v)
    xl, xp = find_ranges(mu_v, sig_v)
    dlx = (xp - xl) / 100
    xr = np.arange(xl, xp, dlx)
    py = 0 * xr
    for kks in range(1, KS):
        norm_pdf = 1/(sig_v[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * (xr - mu_v[kks]) ** 2 / 2 * sig_v[kks] ** 2)
        py = py + (ww_v[kks] * norm_pdf)

    plt.figure(xr, py, 'r')
