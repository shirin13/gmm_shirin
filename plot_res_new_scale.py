# plot MS signal versus GMM model (used for for segments)
def plot_res_new_scale(self, data, ygreki, wwoc, mivoc, sigvoc):
    xx = data
    y_a = ygreki

    ploty = 0 * xx

    KS = len(wwoc)

    for kks in range(1, KS):
        norm_pdf = 1 / (sigvoc[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * (xx - mivoc[kks]) ** 2 / 2 * sigvoc[kks] ** 2)
        ploty = ploty + (wwoc[kks] * norm_pdf)

        plt.plot(xx, y_a, 'k', xx, ploty, 'r')

    for kks in range(1, KS):
        plt.plot(xx, wwoc[kks] * norm_pdf, 'g')

    plt.xlabel('m/z')
