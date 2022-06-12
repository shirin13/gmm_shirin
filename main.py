# demo script for for illustration of the algorithm for Gaussian mixture decomposition of protein MS signals

def main(x, y):

    add_path = 'ms_gmm'

    x = x[:]
    mean_y = np.mean(y, 2)  # create mean spectrum

    # initialization - remove baseline & trim above zero
    YB = baseline_removals(x, mean_y)
    ixg0 = np.argwhere(YB > 0)
    y_b = YB[ixg0]
    x = x[ixg0]

    # main function for GMM decomposition of MS signal
    ww_gmm, mu_gmm, sig_gmm = ms_gmm(x, y_b)

    # show results
    plt.figure(3)
    plt.plot_gmm(x, y_b, ww_gmm, mu_gmm, sig_gmm)
    plt.title('MS signal (black), GMM model (red), components (green)')
