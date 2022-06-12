# find segment between two neighboring splitters
def find_segment(self, ksp, peaks, splitt_v, x, y_bas, mu_l, ww_l, sig_l, mu_p, ww_p, sig_p):
    
    splitt_v = find_split_peaks.Splitt_v
    peaks = gmm.peaks
    KSP = len(splitt_v)

    if ksp > 0:
        xl = peaks[splitt_v[ksp], 1]
    else:
        xl = x[1]
    if ksp < KSP:
        xp = peaks[splitt_v[ksp + 1], 1]
    else:
        xp = x[len(x)]

    idm = np.argwhere(x >= xl & x <= xp)
    x_o = x[idm]
    y_o = y_bas[idm]

    if self.draw == 1:
        xll = max((xl - round((xp - xl) / 5)), x[1])
        xpp = min((xp + round((xp - xl) / 5)), x(len(x)))
        ixp = np.argwhere((x > xll) & (x < xpp))
        x_o_p = x[ixp]
        y_o_p = y_bas[ixp]
        # figure(2)
        plt.subplot(3, 1, 1)
        plt.plot(x_o_p, y_o_p, 'k')
        plt.title('Splitters')

    pyl = np.zeros(x_o)
    if ksp > 0:
        KS = len(ww_l)
        for kks in range(1, KS):
            norm_pdf = 1/(sig_l[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * (x_o - mu_l[kks]) ** 2 / 2 * sig_l[kks] ** 2)
            pyl = pyl + ww_l[kks] * norm_pdf

        if self.draw == 1:
            ok = fill_red(ww_l, mu_l, sig_l)

    pyp = 0 * x_o
    if ksp < KSP:
        KS = len(ww_p)
        for kks in range(1, KS):
            norm_pdf = 1/(sig_p[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * (x_o - mu_p[kks]) ** 2 / 2 * sig_p[kks] ** 2)
            pyp = pyp + ww_p[kks] * norm_pdf
        if self.draw == 1:
            ok = fill_red(ww_p, mu_p, sig_p)

    y_out = y_o - pyl - pyp

    if ksp > 0:
        iyl = np.argwhere(pyl > 0.05 * max(pyl))
        if len(iyl) > 1:
            x_ol = x_o[iyl]
            y_ol = y_out[iyl]
            mp = min(y_ol)
            imp = y_ol.index(mp)
            x_l = x_ol(imp[1])
        else:
            x_l = x_o[1]

    else:
        x_l = xl

    if ksp < KSP:
        iyp = np.argwhere(pyp > 0.05 * max(pyp))
        if len(iyp) > 1:
            x_op = x_o[iyp]
            y_op = y_out[iyp]
            mp = min(y_op)
            imp = y_op.index(mp)
            x_p = x_op[imp[1]]
        else:
            x_p = x_o(len(x_o))
    else:
        x_p = xp

    ix = np.argwhere((x_o <= x_p) & (x_o >= x_l))
    y_out = y_out[ix]
    x_out = x_o[ix]

    iy = np.argwhere(y_out > 0)
    y_out = y_out[iy]
    x_out = x_out[iy]

    if self.draw == 1:
        plt.subplot(3, 1, 2)
        plt.plot(x_o, 0 * y_o, 'r')
        plt.plot(x_out, y_out, 'k')
        #plt.title('Segment:' str(ksp))
