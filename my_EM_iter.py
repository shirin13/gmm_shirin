# EM_iter function - iterations of the EM algorithm

def my_EM_iter(self, x, y, pp_ini, mu_ini, sig_ini, draw, SIG_MIN, eps_change):
    # VALUES FOR CONSTANTS

    # threshold value for terminating iterations eps_change

    # draw = 1 - show graphically positions of means and standard deviations of components during iterartions
    # draw = 0 - omit show option;

    # SIG_MIN - minimum value of sigma squared
    x = self.x1


    SIG_SQ = SIG_MIN * SIG_MIN

    # data length
    N = len(y)

    # correct sig_in if necessary
    sig_ini = max(sig_ini, SIG_MIN)

    # main buffer for intermediate computational results pssmac=zeros(KS,N)
    # initial values for weights, means and standard deviations of Gaussian components
    ppoc = pp_ini
    mivoc = mu_ini
    sigkwvoc = np.power(sig_ini, 2)

    change = 1.0

    # MAIN LOOP
    while change > eps_change:

        ixu = np.argwhere(ppoc > 1.0e-3)
        ppoc = ppoc[ixu]
        mivoc = mivoc[ixu]
        sigkwvoc = sigkwvoc[ixu]
        # ixu = find(sigkwvoc>0.01)
        # ppoc=ppoc(ixu)
        # mivoc=mivoc(ixu)
        # sigkwvoc=sigkwvoc(ixu)
        KS = len[ixu]

        pssmac = np.zeros((KS, N))

        oldppoc = ppoc
        oldsigkwvoc = sigkwvoc

        ppoc = max(ppoc, 1.0e-6)

        for kskla in range(1, KS):
            norm_pdf = 1/(np.sqrt(sigkwvoc[kskla]) * np.sqrt(2 * math.pi)) * np.exp(-1 * (x - mivoc[kskla]) ** 2 / 2 * np.sqrt(sigkwvoc[kskla]) ** 2)
            pssmac[kskla, :] = norm_pdf

        denpss = ppoc * pssmac
        denpss = max(min(denpss[denpss > 0]), denpss)
        for kk in range(1, KS):
            macwwwpom = np.divide((np.multiply((ppoc[kk] * pssmac[kk, :])), y), denpss)
            denom = np.sum(macwwwpom)
            minum = np.sum(macwwwpom * x)
            mivacoc = minum / denom
            mivoc[kk] = mivacoc
            pomvec = (x - (mivacoc * np.multiply((np.ones((N, 1)))), (x - mivacoc * (np.ones((N, 1))))))
            sigkwnum = np.sum(macwwwpom * pomvec)
            sigkwvoc[kk] = max(sigkwnum / np.array(denom, SIG_SQ))
            ppoc[kk] = np.sum(macwwwpom) / np.sum(y)

        change = np.sum(np.abs(ppoc - oldppoc)) + np.sum(np.divide(((np.abs(sigkwvoc - oldsigkwvoc)), sigkwvoc))) / (len(ppoc))

        if draw == 1:
            plt.plot(mivoc, np.sqrt(sigkwvoc), '*')
            plt.xlabel('means')
            plt.ylabel('standard deviations')
            #plt.title(['Progress of the EM algorithm: change=' str(change)])

    # RETURN RESULTS
    l_lik = np.sum(np.multiply(math.log(denpss), y))
    mu_est = np.sort(mivoc)
    isort = np.argsort(mivoc)
    sig_est = np.sqrt(sigkwvoc[isort])
    pp_est = ppoc[isort]
    TIC = np.sum(y)
    bic = l_lik - ((3 * KS-1) / 2) * math.log(TIC)
    return (pp_est, mu_est, sig_est, TIC, l_lik, bic)
