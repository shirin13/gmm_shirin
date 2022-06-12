# plot MS signal and its GMM model

def plot_gmm(self, x, y, ww_gmm, mu_gmm, sig_gmm):
    ploty = 0 * x
    plt.plot(x, y, 'k')

    KS = len(ww_gmm)
    for kks in range(1, KS):
        ixmz = np.argwhere(np.abs((x - mu_gmm[kks]) / sig_gmm[kks]) < 4)
        norm_pdf = 1/(sig_gmm[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * (x[ixmz] - mu_gmm[kks]) ** 2 / 2 * sig_gmm[kks] ** 2)
        ploty[ixmz] = ploty[ixmz] + ww_gmm[kks] * norm_pdf
        plt.plot(x[ixmz], ww_gmm[kks] * norm_pdf, 'g')

    plt.plot(x, ploty, 'r')

    plt.xlabel('M/Z')
    plt.ylabel('Intensity')
