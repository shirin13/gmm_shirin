# find splitter segment around the split peak no k

def find_split_segment(self, k_pick, x, y_bas, seg_vec_c, peaks, par_mcv):
    x = self.x1
    y_base = baseline_removals.y_base
    splitt_v = find_split_peaks.Splitt_v
    seg_vec_c = find_split_peaks.seg_vec_c
    peaks = gmm.peaks
    par_mcv = gmm.par_mcv

    zakl = seg_vec_c.max(1, (k_pick-4))
    zakp = seg_vec_c.min(np.size(peaks[0]), (k_pick+5))

    mzp = x[zakp]
    mzl = x[zakl]
    mzPP = peaks[k_pick, 1]

    if (mzp-mzPP) < 5 * mzPP * par_mcv:
        zakm = np.argwhere(x >= mzp & x <= mzp + 5 * mzPP * par_mcv)
        warty = y_bas[zakm]
        miny = warty.min()
        idxm = miny.index(min(miny))
        prawzak = zakm[idxm[1]]
    else:
        prawzak = zakp

    if (mzPP-mzl) < 5 * mzPP * par_mcv:
        zakm = np.argwhere(x <= mzl & x >= mzl - 5 * mzPP * par_mcv)
        warty = y_bas[zakm]
        miny = warty.min()
        idxm = miny.index(min(miny))
        lewzak = zakm[idxm[1]]
    else:
        lewzak = zakl

    x_o = x[lewzak:prawzak]
    y_bas_o = y_bas[lewzak:prawzak]

    yp = y_bas[prawzak]
    yl = y_bas[lewzak]

    dxl = x[lewzak+1] - x[lewzak]
    dxp = x[prawzak] - x[prawzak-1]

    j = np.arange(x[lewzak], (x[lewzak]-dxl), dxl)
    o = np.arange(dxp, x[prawzak], dxp)

    xaugl = x[lewzak] - 6 * par_mcv * j
    xaugp = x[prawzak] + o + 6 * par_mcv * x[prawzak]

    mu = 0
    sigma_1 = 2 * par_mcv * x[prawzak]
    sigma_2 = 2 * par_mcv * x[lewzak]
    norm_pdf_1 = 1/(sigma_1 * np.sqrt(2 * math.pi)) * np.exp(-1 * ((xaugp - mzp) - mu) ** 2 / 2 * sigma_1 ** 2)
    norm_pdf_2 = 1/(sigma_2 * np.sqrt(2 * math.pi)) * np.exp(-1 * ((xaugl - mzl) - mu) ** 2 / 2 * sigma_2 ** 2)

    yop = np.sqrt(2 * math.pi) * (2 * par_mcv * x(prawzak)) * yp * norm_pdf_1
    yol = np.sqrt(2 * math.pi) * (2 * par_mcv * x(lewzak)) * yl * norm_pdf_2

    # x_ss = [xaugl'; mz_o; xaugp']
    # y_ss = [yol'; y_bas_o; yop']
    return (xaugl, yol)
