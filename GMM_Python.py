
import csv
import numpy as np

import scipy.stats as sp

from scipy import sparse

import scipy

import math

from scipy.sparse.linalg import spsolve

from scipy.linalg import cholesky

from scipy.stats import norm

from scipy.signal import find_peaks, peak_widths

import matplotlib.pyplot as plt

from numpy.linalg import norm

from scipy.sparse import linalg


#np.seterr(divide='ignore', invalid='ignore')
class GaussianMixModel():
    def __init__(self, data):

        np.seterr(divide='ignore', invalid='ignore')

        self.data = data
        data1 = np.array(data).astype('int')
        data1 = data1.reshape(72, 2)
        #data1 = data1.reshape(72, 8)
        #data1 = data1.reshape(9026, 2)

        self.x = data1[:, 0]
        #self.y = data1[:, 1:]
        self.y = data1[:, 1]

        #self.y = np.ravel(self.y)

        #self.y = np.reshape(self.y, (72,7))
        #self.y = self.y.reshape(-1,7)
        #self.y1 = self.y.T
        #self.y1 = self.y
        #self.x1 = self.x.reshape(1, -1)
        #self.x1 = self.x.T
        self.x1 = np.asarray(self.x)
        self.y1 = self.y.flatten()

        #self.m, self.n = data1.shape

        self.res_par = 0.5  # res_par - used in the main body of ms_gmm - multiplied by estimated average width of the peak in the spectrum defines resolutioon of the  decomposition

        self.par_p_sens = 0  # par_p_sens parameter for peak detection sensitivity used in find_split_peaks split peaks must be of height >= Par_P_Sens * maximal peak height

        self.par_q_thr = 1.3  # par_q_thr parameter for peak quality threshold used in find_split_peaks split peaks must have quality >= Par_Q_Thr

        self.par_ini_j = 5  # par_ini_j parameter for initial jump used in find_split_peaks

        self.par_p_l_r = 4  # par_p_l_r parameter for range for peak lookup used in find_split_peaks

        self.par_p_j = 4  # par_p_j parameter for jump used in find_split_peaks

        self.qfpar = 0.5  # qfpar - parameter used in the dynamic programming quality funtion

        self.prec_par_1 = 0.002  # prec_par_1 - precision parameter - weight used to pick best gmm decomposition penalty coefficient for number of components in the quality funtio

        # res_par_2 - used in the EM iterations to define lower bounds for standard deviations
        self.res_par_2 = 0.5

        # penet_par_1  - penetration parameter 1 used to continue searching for best number of components in gmm decomposition (bigger for splitting segments)
        self.penet_par_1 = 15

        # prec_par_2 - precision parameter 2 - weight used to pick best gmm decomposition
        self.prec_par_2 = self.penet_par_1

        # penet_par_1  - penetration parameter 2 used to continue searching for best number of components in gmm decomposition (smaller for segments)
        self.prec_par_2 = 15

        # buf_size_split_par - size of the buffer for computing GMM paramters of splitters
        self.buf_size_split_par = 10

        # buf_size_seg_par - size of the buffer for computing GMM paramters of segments
        self.buf_size_seg_par = 30

        # eps_par - parameter for EM iterations - tolerance for mixture parameters change
        self.eps_par = 0.0001

        self.draw = 1  # draw - show plots of decompositions during computations used in many finctions

    # def baseline_removals(self, data, lam=1e4, ratio=0.05, itermax=100):
    #     """
    #     Baseline correction using asymmetrically
    #     reweighted penalized least squares smoothing
    #     Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    #     Analyst, 2015, 140, 250 (2015)
    #     y:
    #             input data (i.e. chromatogram of spectrum)
    #         lam:
    #             parameter that can be adjusted by user. The larger lambda is,
    #             the smoother the resulting background, z
    #         ratio:
    #             wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
    #         itermax:
    #             number of iterations to perform
    #     Output:
    #         the fitted background vector
    #
    #     """
    #     data = self.data
    #     y = self.y1
    #     x = self.x1
    #     #y_base = 0
    #     for i in range(len(y[0])):
    #         N = len(y[0])
    #         D = sparse.eye(N, format='csc')
    #         # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    #         D = D[1:] - D[:-1]
    #         D = D[1:] - D[:-1]
    #         H = lam * D.T * D
    #         w = np.ones(N)
    #         #w = np.squeeze(np.asarray(w))
    #         # offset = np.zeros(7)
    #
    #         for j in range(itermax):
    #             W = sparse.diags(w, 0, shape=(N, N)) # (72,) (72, 2)
    #             WH = sparse.csc_matrix(W + H)
    #             C = sparse.csc_matrix(cholesky(WH.todense()))
    #             tmp = y[i, :] * w
    #             gh = spsolve(C.T, tmp.T)
    #             y_base = spsolve(C, gh)# spsolve(C, spsolve(C.T, w * y))
    #             d = y - y_base
    #             dn = d[d < 0]
    #             m = np.mean(dn)
    #             s = np.std(dn)
    #             wt = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))  #(72,7)
    #             if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
    #                 break
    #             w = wt
    #         return y_base




    # def bindata(self, y, x, xrg):
    #     # function [ym,yb] = bindata(y,x,xrg)
    #     # Computes ym(ii) = mean(y(x>=xrg(ii) & x < xrg(ii+1)) for every ii
    #     # using a fast algorithm which uses no looping
    #     # If a bin is empty it returns nan for that bin
    #     # Also returns yb, the approximation of y using binning (useful for r^2 calculations). Example:
    #     # x = randn(100,1)
    #     # %y = x.^2 + randn(100,1)
    #     # xrg = linspace(-3,3,10)'
    #     # [ym,yb] = bindata(y,x,xrg)
    #     # X = [xrg(1:end-1),xrg(2:end)]'
    #     # Y = [ym,ym]'
    #     # plt.plot(x,y,'.',X(:),Y(:),'r-')
    #     #data = self.data
    #     y = self.y1
    #     x = self.x1
    #     #print(y)
    #     #print(y.shape)
    #     for i in range(y.shape[0]):
    #         try:
    #             bins = np.linspace(y.min(), y.max(), 100, endpoint=False)
    #     # digitized = np.digitize(y, bins) #histc
    #     # returns ind, an array the same size as x indicating the bin number that each entry in x sorts into. Use this syntax with any of the previous input argument combinations.
    #     #bin_means = [y[digitized == i].mean() for i in range(1, len(bins))]
    #             bin_means = (np.histogram(y, bins, weights=y)[0] / np.histogram(y, bins)[0])
    #
    #         except ZeroDivisionError:
    #             return 0
    #         return bin_means

    def bindata(self, y, x, xrg):

        a = np.histogram(x, xrg.T)
        whichedge = np.digitize(x, xrg.T)
        bins = min(max(whichedge, 1), len(xrg) - 1)
        xpos = np.ones(np.size(bins, 1), 1)
        ns = scipy.csr_matrix(bins, xpos, 1)
        ysum = scipy.csr_matrix(bins, xpos, y)
        ym = np.divide(ysum.todense, ns.todense)
        yb = ym[bins]
        return (ym, yb)

    # decomposition of the proteomic spectrum into Gaussian mixture model

    def find_peak(self, x, y):
        x = self.x1
        y = self.y1
        # find peaks
        # peaks at which point, the value of the peak
        #ind_peaks, properties = find_peaks(y)
        ind_peaks = find_peaks(y)
        #height_peaks = ind_peaks[1]['peak_heights']  # list of the heights of the peaks
        #peaks = x[ind_peaks[0]]  # list of the peaks positions
        y_peaks = y[ind_peaks[0]] # list of the heights of the peaks
        peaks = x[ind_peaks[0]] # list of the peaks positions
        #peak_height = properties["prominences"]
        #plt.plot(peaks, peak_height)
        #plt.show()
        results_half, pos_res, left_ips, right_ips = peak_widths(y, ind_peaks[0], rel_height=0.5) #max(y)/2)
        res_width = np.vstack((left_ips, right_ips)).T#results_half[0]

        par_mcv = self.res_par * np.mean((res_width[:, 1] - res_width[:, 0]) / peaks) #peaks[:, 1]
        return (peaks, res_width, par_mcv)


    def gmm(self, x, y):
        x = self.x1
        y = self.y1

        # find peaks
        # peaks at which point, the value of the peak
        peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)
        res_par = self.res_par
        buf_size_split_par = self.buf_size_split_par

        splitt_v, seg_vec_c = GaussianMixModel.find_split_peaks(self, peaks, x, y, par_mcv)

        # estimate resolution
        #par_mcv = res_par * np.mean((res_width[:, 2] - res_width[:, 1]) / peaks[:, 1])

        # find good quality splitting peaks
        #splitt_v, seg_vec_c = find_split_peaks(self, peaks, x, y_base, par_mcv)
        KSP = np.max(splitt_v.shape)  # number of splitting peaks
        # buffers for splitter parameters
        mu_mx_1 = np.zeros((KSP, buf_size_split_par))
        ww_mx_1 = np.zeros((KSP, buf_size_split_par))
        sig_mx_1 = np.zeros((KSP, buf_size_split_par))

        for kk in range(1, KSP):
            #  disp(['split Progress: ' str[kk] ' of ' str[KSP])
            # Gaussian mixture decomposition of the splitting segment
            ww_pick, mu_pick, sig_pick = GaussianMixModel.gmm_decomp_split_segment(self, x, y_bas, splitt_v, seg_vec_c, peaks, par_mcv, kk)
            mu_mx_1[kk, :] = mu_pick
            ww_mx_1[kk, :] = ww_pick
            sig_mx_1[kk, :] = sig_pick

        # PHASE 2 - GMM DECOMPOSITIONS OF SEGMENTS

        # buffers for segments decompositions paramteres
        KSP1 = KSP + 1
        mu_mx_2 = np.zeros((KSP1, buf_size_seg_Par))
        ww_mx_2 = np.zeros((KSP1, buf_size_seg_Par))
        sig_mx_2 = np.zeros((KSP1, buf_size_seg_Par))
        for ksp in range(1, KSP1):
            #    disp(['Seg Progress: ' num2str(ksp) ' of ' num2str(KSP1)])
            ww_out, mu_out, sig_out = GaussianMixModel.gmm_decomp_segment(self.x, y_bas, ww_mx_1, mu_mx_1, sig_mx_1, peaks, splitt_v, par_mcv, ksp - 1)
            mu_mx_2[ksp, :] = mu_out
            ww_mx_2[ksp, :] = ww_out
            sig_mx_2[ksp, :] = sig_out

        # AGGREGATION
        # aggregate components computed in PHASE 1 and in PHASE 2

        NUMPICK = np.size(peaks[0])
        mu_vec = np.zeros((7 * NUMPICK, 1))
        ww_vec = np.zeros((7 * NUMPICK, 1))
        sig_vec = np.zeros((7 * NUMPICK, 1))

        kacum = 0

        for ksp in range(1, KSP + 1):
            mu_a = mu_mx_2[ksp, :]
            ww_a = ww_mx_2[ksp, :]
            sig_a = sig_mx_2[ksp, :]
            ixp = max(np.argwhere(ww_a > 0))
            if len(ixp) == 0:
                # emergency correction in the case of detected ovelap between splitters
                ww_a_l_new, mu_a_l_new, sig_a_l_new, ww_a_p_new, mu_a_p_new, sig_a_p_new = GaussianMixModel.emcor(self, ksp - 1, mu_mx_1,
                                                                                                 ww_mx_1, sig_mx_1, x,
                                                                                                 y_bas, splitt_v, peaks,
                                                                                                 par_mcv)
                if ksp > 1:
                    ww_mx_1[ksp - 1, :] = ww_a_l_new
                    mu_mx_1[ksp - 1, :] = mu_a_l_new
                    sig_mx_1[ksp - 1, :] = sig_a_l_new

                if ksp < KSP + 1:
                    ww_mx_1[ksp, :] = ww_a_p_new
                    mu_mx_1[ksp, :] = mu_a_p_new
                    sig_mx_1[ksp, :] = sig_a_p_new

            else:
                mu_a = mu_a[1: ixp]
                ww_a = ww_a[1: ixp]
                sig_a = sig_a[1: ixp]
                for kk in range(1, len(ww_a)):
                    kacum = + 1
                    mu_vec[kacum] = mu_a[kk]
                    ww_vec[kacum] = ww_a[kk]
                    sig_vec[kacum] = sig_a[kk]

        for ksp in range(1, KSP):
            mu_a = mu_mx_1[ksp, :]
            ww_a = ww_mx_1[ksp, :]
            sig_a = sig_mx_1[ksp, :]
            ixp = max(np.argwhere(ww_a > 0))
            mu_a = mu_a[1: ixp]
            ww_a = ww_a[1: ixp]
            sig_a = sig_a[1: ixp]

            for kk in range(1, len(ww_a)):
                kacum = + 1
                mu_vec[kacum] = mu_a[kk]
                ww_vec[kacum] = ww_a[kk]
                sig_vec[kacum] = sig_a[kk]

        mu_vec = mu_vec[1: kacum]
        ww_vec = ww_vec[1: kacum]
        sig_vec = sig_vec[1: kacum]

        mu_sort, muixs = np.sort(mu_vec)
        mu_gmm = mu_sort
        ww_gmm = ww_vec[muixs]
        sig_gmm = sig_vec[muixs]

        return (ww_gmm, mu_gmm, sig_gmm)

    #   main dynamic programming algorithm for computing initial conditions for EM iterations

    def dyn_pr_split_w(self, data, ygreki, k_gr, aux_mx, par, par_sig_min):

        # data = ?????
        #ygreki = ????
        #par = ????
        #k_gr = ????
        #data, ygreki = GaussianMixModel.find_segment(self, ksp, peaks, splitt_v, x, y_bas, mu_l, ww_l, sig_l, mu_p, ww_p, sig_p)
        #aux_max = GaussianMixModel.dyn_pr_split_w_aux(self, data, ygreki, par, par_sig_min)
        # initialize
        Q = np.zeros((1, k_gr))
        N = len(data)
        p_opt_idx = np.zeros((1, N))
        p_aux = np.zeros((1, N))
        opt_pals = np.zeros((k_gr, N))
        for kk in range(1, N):
            p_opt_idx[kk] = GaussianMixModel.my_qu_ix_w(self, data[kk:N], ygreki[kk:N], par, par_sig_min)

        # aux_mx - already computed

        # iterate
        for kster in range(1, k_gr):
            for kk in range(1, N - kster):
                for jj in range(kk + 1, N - kster + 1):
                    p_aux[jj] = aux_mx[kk, jj] + p_opt_idx[jj]

            mm = min(p_aux[kk + 1:N - kster + 1])
            ix = np.argwhere(min(p_aux[kk + 1:N - kster + 1]))
            p_opt_idx[kk] = mm
            opt_pals[kster, kk] = kk + ix[1]

        Q[kster] = p_opt_idx[1]

        # restore optimal decisions
        opt_part = np.zeros((1, k_gr))
        opt_part[1] = opt_pals[k_gr, 1]
        for kster in range(k_gr - 1, 1, -1):
            opt_part[k_gr - kster + 1] = opt_pals[kster, opt_part[k_gr - kster]]

        return (Q, opt_part)

    # auxiliary function for dynamic programming - compute quality index matrix

    def dyn_pr_split_w_aux(self, data, ygreki, par, par_sig_min):

        #data = ???
        #ygreki = ???
        #par = ????
        #par_sig_min = GaussianMixModel.gmm_decomp_split_segment() #????
        N = len(data)
        # aux_mx
        aux_mx = np.zeros((N, N))
        for kk in range(1, N - 1):
            for jj in range(kk + 1, N):
                aux_mx[kk, jj] = GaussianMixModel.my_qu_ix_w(self, data[kk:jj - 1], ygreki[kk:jj - 1], par, par_sig_min)

        return aux_mx

    # emergency correction function launched in the case of (rare) overlap between splitters
    def emcor(self, ksp, mu_mx_1, ww_mx_1, sig_mx_1, x, y_bas, splitt_v, peaks, par_mcv):

        x = self.x1
        y_base = self.y1
        # ksp = ????
        #mu_mx_1 = ???
        #ww_mx_1 =???
        #sig_mx_1 = ???
        #splitt_v, seg_vec_c = GaussianMixModel.find_split_peaks(self, peaks, x, y, par_mcv)
        #peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)
        KSP = len(splitt_v)
        res_par_2 = self.res_par_2
        em_tol = self.eps_par


        ww_a_l_new = 0 * ww_mx_1[1, :]
        mu_a_l_new = 0 * ww_mx_1[1, :]
        sig_a_l_new = 0 * ww_mx_1[1, :]

        ww_a_p_new = 0 * ww_mx_1[1, :]
        mu_a_p_new = 0 * ww_mx_1[1, :]
        sig_a_p_new = 0 * ww_mx_1[1, :]

        if ksp > 0:
            mu_l = mu_mx_1[ksp, :]
            ww_l = ww_mx_1[ksp, :]
            sig_l = sig_mx_1[ksp, :]
            ixp = max(np.argwhere(ww_l > 0))

            if len(ixp) == 0:
                KSll = 0
                xll = x[1]
                mu_l = []
                ww_l = []
                sig_l = []
            else:
                mu_l = mu_l[1: ixp]
                ww_l = ww_l[1: ixp]
                sig_l = sig_l[1: ixp]
                KSll = len(ww_l)
                xll, xlp = GaussianMixModel.find_ranges(self, mu_l, sig_l)
        else:
            KSll = 0
            xll = x[1]
            mu_l = []
            ww_l = []
            sig_l = []

        if ksp < KSP:
            mu_p = mu_mx_1[ksp + 1, :]
            ww_p = ww_mx_1[ksp + 1, :]
            sig_p = sig_mx_1[ksp + 1, :]
            ixp = max(np.argwhere(ww_p > 0))
            mu_p = mu_p[1: ixp]
            ww_p = ww_p[1: ixp]
            sig_p = sig_p[1: ixp]
            KSpp = len(ww_p)
            xlp, xpp = GaussianMixModel.find_ranges(self, mu_p, sig_p)
        else:
            KSpp = 0
            xpp = x[-1]
            mu_p = []
            ww_p = []
            sig_p = []

        idm = np.argwhere(x >= xll & x <= xpp)
        x_o = x[idm]
        y_o = y_bas[idm]
        pyl = 0 * x_o
        pyp = 0 * x_o

        if ksp > 0:
            for kks in range(1, KSll):
                norm_pdf = 1 / (sig_l[kks] * np.sqrt(2 * math.pi)) * np.exp(
                    -1 * (x_o - mu_l[kks]) ** 2 / 2 * sig_l[kks] ** 2)
                pyl = pyl + ww_l[kks] * norm_pdf
            idl = np.argwhere(x_o >= xll & x_o <= peaks[splitt_v[ksp], 1])

            pylpds = pyl[idl]
            y_o[idl] = pylpds

        if ksp < KSP:
            for kks in range(1, KSpp):
                norm_pdf = 1 / (sig_p[kks] * np.sqrt(2 * math.pi)) * np.exp(
                    -1 * (x_o - mu_p[kks]) ** 2 / 2 * sig_p[kks] ** 2)
                pyp = pyp + ww_p[kks] * norm_pdf

            idp = np.argwhere(x_o >= peaks[splitt_v[(ksp + 1)], 1] & x_o <= xpp)
            pyppds = pyp[idp]
            y_o[idp] = pyppds

        KS = KSll + KSpp

        pp_ini = np.array(ww_l, ww_p) / np.sum(np.array(ww_l, ww_p))
        mu_ini = np.array(mu_l, mu_p)
        sig_ini = np.array(sig_l, sig_p)

        par_sig_min = res_par_2 * par_mcv * np.mean(x_o)
        pp_est, mu_est, sig_est = GaussianMixModel.my_EM_iter(self, x_o, y_o, pp_ini, mu_ini, sig_ini, 0, par_sig_min, em_tol)
        KS = len(pp_est)
        KSpp = KS - KSll

        _, scale = GaussianMixModel.qua_scal(self, x_o, y_o, pp_est, mu_est, sig_est)
        ww_est = pp_est * scale

        if ksp > 0:
            ww_a_l_new[1: KSll] = ww_est[1: KSll]
            mu_a_l_new[1: KSll] = mu_est[1: KSll]
            sig_a_l_new[1: KSll] = sig_est[1: KSll]

        if ksp < KSP:
            ww_a_p_new[1: KSpp] = ww_est[KSll + 1: KS]
            mu_a_p_new[1: KSpp] = mu_est[KSll + 1: KS]
            sig_a_p_new[1: KSpp] = sig_est[KSll + 1: KS]

        # ok = plot_res_new_scale(x_o, y_o, ww_est, mu_est, sig_est)

        return (ww_a_l_new, mu_a_l_new, sig_a_l_new, ww_a_p_new, mu_a_p_new, sig_a_p_new)

    # auxiliary function for find_split_peaks

    def f_par_mcv(self, kini, fst, peaks, par_mcv):
        # kini = peaks[0]# kini - initial position of the peak counter
        # fst =  #number of forward steps intended to make
        #peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)
        fst_new = 0
        if kini + fst > np.size(peaks[0]):
            fst_new = 0
        else:
            fst_new = fst
            while True:
                if peaks[kini + fst_new, 1] - peaks[kini, 1] > fst * par_mcv * peaks[kini, 1]:
                    break
                else:
                    fst_new += 1
                    if kini + fst_new > np.size(peaks[0]):
                        fst_new = 0
                        break
            return fst_new

    # auxiliary function for filling splitter components in red

    def fill_red(self, ww_v, mu_v, sig_v):

        #ww_v = ???
        #mu_v = ????
        #sig_v= ??????
        KS = len(ww_v)
        xl, xp = GaussianMixModel.find_ranges(self, mu_v, sig_v)
        dlx = (xp - xl) / 100
        xr = np.arange(xl, xp, dlx)
        py = 0 * xr
        for kks in range(1, KS):
            norm_pdf = 1 / (sig_v[kks] * np.sqrt(2 * math.pi)) * np.exp(
                -1 * (xr - mu_v[kks]) ** 2 / 2 * sig_v[kks] ** 2)
            py = py + (ww_v[kks] * norm_pdf)

        plt.figure(xr, py, 'r')

    # auxiliary function find reasonable ranges for a group of Gaussian components

    def find_ranges(self, mu_v, sig_v):

        #mu_v = ???
        #sig_v = ??
        K = len(mu_v)
        mzlv = np.zeros((K, 1))
        mzpv = np.zeros((K, 1))

        for kk in range(1, K):
            mzlv[kk] = mu_v[kk] - np.multiply(3, sig_v[kk])
            mzpv[kk] = mu_v[kk] + np.multiply(3, sig_v[kk])

        mzlow = min(mzlv)
        mzhigh = max(mzpv)
        return (mzlow, mzhigh)

    # find segment between two neighboring splitters
    def find_segment(self, ksp, peaks, splitt_v, x, y_bas, mu_l, ww_l, sig_l, mu_p, ww_p, sig_p):

        #ksp = ???
        x = self.x1
        y_bas= self.y1
        draw = self.draw
        #peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)
        #splitt_v, seg_vec_c = GaussianMixModel.find_split_peaks(self, peaks, x, y, par_mcv)
        #mu_l = ???
        # #ww_l = ???
        # #sig_l, = ???
        # #mu_p, = ???
        # #ww_p = ???
        # #sig_p = ????

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

        if draw == 1:
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
                norm_pdf = 1 / (sig_l[kks] * np.sqrt(2 * math.pi)) * np.exp(
                    -1 * (x_o - mu_l[kks]) ** 2 / 2 * sig_l[kks] ** 2)
                pyl = pyl + ww_l[kks] * norm_pdf

            if draw == 1:
                ok = GaussianMixModel.fill_red(self, ww_l, mu_l, sig_l)

        pyp = 0 * x_o
        if ksp < KSP:
            KS = len(ww_p)
            for kks in range(1, KS):
                norm_pdf = 1 / (sig_p[kks] * np.sqrt(2 * math.pi)) * np.exp(
                    -1 * (x_o - mu_p[kks]) ** 2 / 2 * sig_p[kks] ** 2)
                pyp = pyp + ww_p[kks] * norm_pdf
            if draw == 1:
                ok = GaussianMixModel.fill_red(self, ww_p, mu_p, sig_p)

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

        if draw == 1:
            plt.subplot(3, 1, 2)
            plt.plot(x_o, 0 * y_o, 'r')
            plt.plot(x_out, y_out, 'k')
            # plt.title('Segment:' str(ksp))

    # find splitting peaks

    def find_split_peaks(self, peaks, x, y, par_mcv):

        x = self.x1
        y = self.y1
        peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)
        seg_vec = np.zeros((np.size(peaks)))
        par_p_l_r = self.par_p_l_r
        par_p_sens = self.par_p_sens
        par_q_thr = self.par_q_thr
        par_p_j = self.par_p_j
        par_ini_j = self.par_ini_j

        #for k in range(np.size(peaks)):
        for k in range(0, np.size(peaks) -1):
            # zakm1 = np.argwhere((x <= peaks[k+1]))
            #zakm = np.flatnonzero((x <= peaks[k+1]) & (x >= peaks[k]))
            zakm = np.argwhere((x <= peaks[k + 1]) & (x >= peaks[k]))
            #zakm = np.argwhere((x <= peaks[k ]) & (x >= peaks[k - 1]))
            #print((np.size(peaks)))
            #zakm1 = x[(x <= peaks[k+1])]
            #print(peaks[k])
            #print(peaks[k + 1])
            #print(zakm1)
            #zakm2 = x[(x >= peaks[k])]
            #zakm = x[(x <= peaks[k+1]) & (x >= peaks[k])]
            #print(zakm2)
            #zakm = x[zakm1 & zakm2]
            #zakm = np.argwhere(zakm1 & zakm2)
            warty = y[zakm]
            miny = min(warty)
            index_of_minimum = np.where(warty == miny)
            idxm_min_y = warty[index_of_minimum]
            #idxm_min_y = warty.index(miny)
            #prawzak = zakm[idxm_min_y]
            prawzak = idxm_min_y
            seg_vec[k] = prawzak


        seg_vec[-1] = len(x)
        #seg_vec_c = np.concatenate((1, seg_vec), axis=0)
        seg_vec_c = np.insert(seg_vec, 0, 1)
        M_B_H = np.zeros((np.size(peaks)))
        for kk in range(1, np.size(peaks[0])):
            if y(seg_vec_c[kk]) >= y(seg_vec_c[kk]):
                M_B_H[kk] = y(seg_vec_c[kk]) + 1
            else:
                M_B_H[kk] = y(seg_vec_c[kk + 1]) + 1

        max_peaks = np.max(peaks)
        ppe = peaks / M_B_H
        Kini = GaussianMixModel.f_par_mcv(self, 1, par_ini_j, peaks, par_mcv)
        splitt_v = np.zeros(np.floor(((np.max(x) - x[0]) / x[0] * par_mcv)), 1)
        kspl = 0

        while True:
            if GaussianMixModelf_par_mcv(self, Kini, par_p_l_r, peaks, par_mcv) == 0:
                break

            Top = Kini + GaussianMixModel.f_par_mcv(self, Kini, par_p_l_r, peaks, par_mcv)
            Zak = np.array(Kini, Top)
            # verify quality condition for the best peak in the range Zak=Kini:Top
            pzak2 = peaks[Zak, 1]

            ppezak = ppe[Zak]
            ixq = np.argwhere(pzak2 > (par_p_sens * max_peaks) & ppezak > par_q_thr)

            if np.size(ixq) >= 1:
                [mpk, impk] = np.max(ppe(Zak[ixq]))
                kkt = ixq(impk[0])
                kspl += 1
                splitt_v[kspl] = Zak[kkt]
                Kini = Top + GaussianMixModel.f_par_mcv(self, Top, par_p_j, peaks, par_mcv)

            else:
                Kini = Top + GaussianMixModel.f_par_mcv(self, Top, 1, peaks, par_mcv)

        splitt_v = np.rane(1, kspl)
        return (splitt_v, seg_vec_c)

    # find splitter segment around the split peak no k

    def find_split_segment(self, k_pick, x, y_bas, seg_vec_c, peaks, par_mcv):

        #k_pick = ????
        x = self.x1
        y_base = self.y1
        #splitt_v, seg_vec_c = GaussianMixModel.find_split_peaks(self, peaks, x, y, par_mcv)
        #peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)

        zakl = seg_vec_c.max(1, (k_pick - 4))
        zakp = seg_vec_c.min(np.size(peaks[0]), (k_pick + 5))

        mzp = x[zakp]
        mzl = x[zakl]
        mzPP = peaks[k_pick, 1]

        if (mzp - mzPP) < 5 * mzPP * par_mcv:
            zakm = np.argwhere(x >= mzp & x <= mzp + 5 * mzPP * par_mcv)
            warty = y_bas[zakm]
            miny = warty.min()
            idxm = miny.index(min(miny))
            prawzak = zakm[idxm[1]]
        else:
            prawzak = zakp

        if (mzPP - mzl) < 5 * mzPP * par_mcv:
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

        dxl = x[lewzak + 1] - x[lewzak]
        dxp = x[prawzak] - x[prawzak - 1]

        j = np.arange(x[lewzak], (x[lewzak] - dxl), dxl)
        o = np.arange(dxp, x[prawzak], dxp)

        xaugl = x[lewzak] - 6 * par_mcv * j
        xaugp = x[prawzak] + o + 6 * par_mcv * x[prawzak]

        mu = 0
        sigma_1 = 2 * par_mcv * x[prawzak]
        sigma_2 = 2 * par_mcv * x[lewzak]
        norm_pdf_1 = 1 / (sigma_1 * np.sqrt(2 * math.pi)) * np.exp(-1 * ((xaugp - mzp) - mu) ** 2 / 2 * sigma_1 ** 2)
        norm_pdf_2 = 1 / (sigma_2 * np.sqrt(2 * math.pi)) * np.exp(-1 * ((xaugl - mzl) - mu) ** 2 / 2 * sigma_2 ** 2)

        yop = np.sqrt(2 * math.pi) * (2 * par_mcv * x(prawzak)) * yp * norm_pdf_1
        yol = np.sqrt(2 * math.pi) * (2 * par_mcv * x(lewzak)) * yl * norm_pdf_2

        # x_ss = [xaugl'; mz_o; xaugp']
        # y_ss = [yol'; y_bas_o; yop']
        return (xaugl, yol)

    #       gmm decomposition of a segment based on dynamic programming initialization
    def gmm_decomp_segment(self, x, y_bas, ww_mx_1, mu_mx_1, sig_mx_1, peaks, splitt_v, par_mcv, fr_no):
        # buffers
        x = self.x
        y_base = self.y1
        #splitt_v, seg_vec_c = GaussianMixModel.find_split_peaks(self, peaks, x, y, par_mcv)
        #peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)
        fr_no = self.eps_par
        draw = self.draw
        buf_size_seg_par = self.buf_size_seg_par
        res_par_2 = self.res_par_2
        qfpar = self.qfpar
        par_penet = self.par_penet
        prec_par_2 = self.prec_par_2
        penet_par_2 = self.penet_par_2
        em_tol = self.eps_par

        #ww_mx_1 = ???
        #mu_mx_1 = ??
        #sig_mx_1= ????

        ww_dec = np.zeros((1, buf_size_seg_par))
        mu_dec = np.zeros((1, buf_size_seg_par))
        sig_dec = np.zeros((1, buf_size_seg_par))

        invec = np.zeros()
        yinwec = np.zeros()

        # assign fragment number to ksp
        KSP = len(splitt_v)
        ksp = fr_no

        # find separated fragments
        if ksp > 0:
            mu_l = mu_mx_1[ksp, :]
            ww_l = ww_mx_1[ksp, :]
            sig_l = sig_mx_1[ksp, :]
            ktr = max((np.where(ww_l > 0)))
            mu_l = mu_l[1: ktr]
            ww_l = ww_l[1: ktr]
            sig_l = sig_l[1: ktr]
        else:
            mu_l = []
            ww_l = []
            sig_l = []

        if ksp < KSP:
            mu_p = mu_mx_1[ksp + 1, :]
            ww_p = ww_mx_1[ksp + 1, :]
            sig_p = sig_mx_1[ksp + 1, :]
            ktr = max((np.argwhere(ww_p > 0)))
            mu_p = mu_p[1:ktr]
            ww_p = ww_p[1:ktr]
            sig_p = sig_p[1:ktr]
        else:
            mu_p = []
            ww_p = []
            sig_p = []

        x_out, y_out = GaussianMixModel.find_segment(self, ksp, peaks, splitt_v, x, y_bas, mu_l, ww_l, sig_l, mu_p, ww_p, sig_p)

        x_out = x_out[:]
        y_out = y_out[:]
        if len(x_out) > 300:
            dx = (x_out[len(x_out)] - x_out[1]) / 200
            x_out_bb = np.arange(x_out[1], x_out[len(x_out)], dx)
            x_out_b = x_out_bb[1:200] + 0.5 * dx
            y_out_b, yb = GaussianMixModel.bindata(self, y_out, x_out, x_out_bb)
            ixnn = np.argwhere(~ math.isnan(y_out_b))
            y_out_b = y_out_b[ixnn]
            x_out_b = x_out_b[ixnn]
            y_out_b = y_out_b[:]
            x_out_b = x_out_b[:]
        else:
            y_out_b = y_out
            x_out_b = x_out

        # find appropriate gmm model for the segment

        quamin = np.inf
        N = len(x_out)
        Nb = len(x_out_b)
        par_sig_min = res_par_2 * par_mcv * np.mean(x_out)

        while True:
            if len(x_out) < 3:
                continue

            else:
                KSmin = min(1, (np.floor((x_out[N] - x_out[1]) / par_sig_min) - 1))
                if KSmin <= 0:
                    wwec = y_out / (np.sum(y_out))
                    mu_est = np.sum(np.multiply(x_out, wwec))
                    pp_est = 1
                    sig_est = np.sqrt(
                        np.sum(((x_out - np.multiply(np.power(np.sum(np.multiply(x_out, wwec))), 2), wwec))))
                    qua, scale = qua_scal(x_out, y_out, pp_est, mu_est, sig_est)
                else:
                    KS = KSmin
                    # penetration - how far are we searching for minimum
                    par_penet = min(np.array(penet_par_2, np.array(np.floor((x_out[N] - x_out[1]) / par_sig_min),
                                                                        np.floor(len(x_out) / 4))))
                    kpen = 0

                    # name=['dane_nr' num2str(ksp)]
                    # save(name, 'mz_out', 'y_out', 'mz_out_b', 'y_out_b', 'QFPAR', 'PAR_sig_min', 'PAR_penet');
                    aux_mx = GaussianMixModel.dyn_pr_split_w_aux(self, x_out_b, y_out_b, qfpar, par_sig_min)
                    while KS < buf_size_seg_par:
                        KS = KS + 1
                        kpen = kpen + 1

                        if KS > KSmin + 1 and KS >= len(x_out) / 2:
                            break

                        Q, opt_part = GaussianMixModel.dyn_pr_split_w(self, x_out_b, y_out_b, KS - 1, aux_mx, qfpar, par_sig_min)
                        part_cl = np.array(1, opt_part, Nb + 1)

                        # set initial cond
                        pp_ini = np.zeros((1, KS))
                        mu_ini = np.zeros((1, KS))
                        sig_ini = np.zeros((1, KS))
                        for kkps in range(1, KS):
                            invec = x_out_b[part_cl[kkps]:part_cl[kkps + 1] - 1]
                            yinwec = y_out_b[part_cl[kkps]:part_cl[kkps + 1] - 1]
                            wwec = yinwec / (np.sum(yinwec))
                            pp_ini[kkps] = np.sum(yinwec) / np.sum(y_out)
                            mu_ini[kkps] = np.sum(np.multiply(invec, wwec))
                            # sig_ini[(kkps)]=sqrt(sum(((invec-sum(invec.*wwec')).^2).*wwec'))
                            sig_ini[kkps] = 0.5 * (max(invec) - min(invec))

                        pp_est, mu_est, sig_est, TIC, l_lik, bic = GaussianMixModel.my_EM_iter(self, x_out, y_out, pp_ini, mu_ini, sig_ini, 0,
                                                                              par_sig_min, em_tol)

                        # compute quality indices of gmm model of the fragment
                        qua, scale = GaussianMixModel.qua_scal(self, x_out, y_out, pp_est, mu_est, sig_est)
                        quatest = qua + prec_par_2 * KS

                        if (quatest < quamin):
                            quamin = quatest
                            pp_min = pp_est
                            mu_min = mu_est
                            sig_min = sig_est
                            scale_min = scale

                        elif (quatest > quamin) and (kpen > par_penet):
                            pp_est = pp_min
                            mu_est = mu_min
                            sig_est = sig_min
                            scale = scale_min
                            break

                if draw == 1:
                    # figure(2)
                    plt.subplot(3, 1, 3)
                    ok = GaussianMixModel.plot_res_new_scale(self, x_out, y_out, pp_est * scale, mu_est, sig_est)
                    plt.xlabel(str(fr_no), str(len(pp_est)))

                ww_o = pp_est * scale
                mu_o = mu_est
                sig_o = sig_est

                for kkpick in range(1, len(ww_o)):
                    mu_dec[kkpick] = mu_o[kkpick]
                    ww_dec[kkpick] = ww_o[kkpick]
                    sig_dec[kkpick] = sig_o[kkpick]

        return (ww_dec, mu_dec, sig_dec)

    # gmm decomposition of splitter segment based on dynamic programming initialization for splitting segments

    def gmm_decomp_split_segment(self, x, y_base, splitt_v, seg_vec_c, peaks, par_mcv, sp_no):
        x = self.x
        y_base = self.y1
        #splitt_v, seg_vec_c = GaussianMixModel.find_split_peaks(self, peaks, x, y, par_mcv)
        #peaks, res_width, par_mcv = GaussianMixModel.find_peak(self, x, y)
        #sp_no = 1 #????
        buf_size_split_par = self.buf_size_split_par
        draw = self.draw
        qfpar = self.qfpar
        res_par_2 = self.res_par_2
        prec_par_1 = self.prec_par_1
        penet_par_1 = self.penet_par_1
        em_tol = self.eps_par

        #invec = np.zeros()
        #yinwec = np.zeros()

        # input
        # mz,y_bas - spectrum
        # splitt_v, seg_vec_c - list of splitting peaks and segment bounds computed by find_split_peaks
        # peaks ,par_mcv - peaks,
        # sp_no - number of splitting segment

        #  find appropriate gmm model for the splitting segment no Sp_No
        # buffers
        ww_pick = np.zeros((1, buf_size_split_par))
        mu_pick = np.zeros((1, buf_size_split_par))
        sig_pick = np.zeros((1, buf_size_split_par))

        # un-binned data
        x_out, y_out = GaussianMixModel.find_split_segment(self, splitt_v[sp_no], x, y_base, seg_vec_c, peaks, par_mcv)
        x_out = x_out[:]
        y_out = y_out[:]
        # bin if necessary
        if len(x_out) > 300:
            dx = (x_out(len(x_out)) - x_out[0]) / 200
            x_out_bb = np.arange((x_out(1), x_out(len(x_out)), dx))
            x_out_b = x_out_bb[1:200] + 0.5 * dx
            y_out_b, yb = GaussianMixModel.bindata(self, y_out, x_out, x_out_bb)
            ixnn = np.argwhere(math.isnan(y_out_b))
            y_out_b = y_out_b[ixnn]
            x_out_b = x_out_b[ixnn]
            y_out_b = y_out_b[:]
            x_out_b = x_out_b[:]
        else:
            y_out_b = y_out
            x_out_b = x_out

        N = len(x_out)
        Nb = len(x_out_b)
        quamin = np.inf
        par_sig_min = res_par_2 * par_mcv * np.mean(x_out)
        KSmin = min(2, (np.floor((x_out[N] - x_out[1]) / par_sig_min) - 1))
        if KSmin <= 0:
            wwec = y_out / (np.sum(y_out))
            mu_est = np.sum(np.dot(x_out, wwec))
            pp_est = 1
            sig_est = np.sqrt(np.sum(np.dot(((x_out - mu_est) ** 2), wwec)))
            qua, scale = GaussianMixModel.qua_scal(self, x_out, y_out, pp_est, mu_est, sig_est)
        else:

            KS = KSmin
            par_penet = min([penet_par_1, np.floor((x_out[N] - x_out[1]) / par_sig_min)])
            kpen = 0
            Q = 0

            aux_mx = GaussianMixModel.dyn_pr_split_w_aux(self, x_out_b, y_out_b, qfpar, par_sig_min)
            while KS <= 2 * (KSmin + par_penet):
                KS = KS + 1
                kpen = kpen + 1

                Q, opt_part = GaussianMixModel.dyn_pr_split_w(self, x_out_b, y_out_b, KS - 1, aux_mx, qfpar, par_sig_min)
                part_cl = np.array(1, opt_part, Nb + 1)

                # set initial cond
                pp_ini = np.zeros((1, KS))
                mu_ini = np.zeros((1, KS))
                sig_ini = np.zeros((1, KS))
                for kkps in range(1, KS):
                    invec = x_out_b[part_cl[kkps]: part_cl[kkps + 1] - 1]
                    yinwec = y_out_b[part_cl[kkps]: part_cl[kkps + 1] - 1]
                    wwec = yinwec / (np.sum(yinwec))
                    pp_ini[kkps] = np.sum(yinwec) / np.sum(y_out_b)
                    mu_ini[kkps] = np.sum(np.multiply(invec, wwec))
                    # sig_ini(kkps) = np.sqrt(sum(((invec-sum(invec.*wwec')).^2).*wwec'))
                    sig_ini[kkps] = 0.5 * (max(invec) - min(invec))

                pp_est, mu_est, sig_est, TIC, l_lik, bic = GaussianMixModel.my_EM_iter(self, x_out, y_out, pp_ini, mu_ini, sig_ini, 0,
                                                                      par_sig_min, em_tol)

                # compute quality indices and scale of gmm model of the fragment
                qua, scale = GaussianMixModel.qua_scal(self, x_out, y_out, pp_est, mu_est, sig_est)
                quatest = qua + prec_par_1 * KS

                if (quatest < quamin):
                    quamin = quatest
                    pp_min = pp_est
                    mu_min = mu_est
                    sig_min = sig_est
                    scale_min = scale

                if (quatest > quamin) and (kpen > par_penet):
                    pp_est = pp_min
                    mu_est = mu_min
                    sig_est = sig_min
                    scale = scale_min
                    break

        # pick and store results
        dist = np.abs(np.divide((mu_est - peaks[splitt_v[sp_no], 1]), sig_est))
        ixf = np.argwhere(dist <= 3)
        if len(ixf) == 0:
            tmp = min(dist)
            ixf = dist.index(min(dist))
            ixnf = np.argwhere(dist > tmp)
        else:
            ixnf = np.argwhere(dist > 3)

        mu_p = mu_est[ixf]
        ww_p = scale * pp_est[ixf]
        sig_p = sig_est[ixf]

        mu_t = mu_est[ixnf]
        ww_t = scale * pp_est[ixnf]
        sig_t = sig_est[ixnf]

        inn = np.argwhere(mu_t < max(mu_p) & mu_t > min(mu_p))
        mu_tp = mu_t[inn]
        ww_tp = ww_t[inn]
        sig_tp = sig_t[inn]

        mu_pp = np.array(mu_p, mu_tp)
        ww_pp = np.array(ww_p, ww_tp)
        sig_pp = np.array(sig_p, sig_tp)

        for kkpick in range(1, len(ww_pp)):
            mu_pick[kkpick] = mu_pp[kkpick]
            ww_pick[kkpick] = ww_pp[kkpick]
            sig_pick[kkpick] = sig_pp[kkpick]

        # plots
        if draw == 1:
            plt.subplot(2, 1, 1)
            plt.plot(x_out, y_out, 'k')
            plt.plot(peaks[splitt_v[sp_no], 1], peaks[splitt_v[sp_no], 1], np.array(0, max(y_out)), 'r')
            plt.ylabel('y (no. of counts)')
            # plt.title('Splitter segment: ' str(sp_no))
            plt.subplot(2, 1, 2)
            ww_est = scale * pp_est
            ok = GaussianMixModel.plot_gmm(self, x_out, y_out, ww_est, mu_est, sig_est)
            ok = GaussianMixModel.fill_red(self, ww_pp, mu_pp, sig_pp)
            # plt.title('Splitter: ' str(sp_no))

        return (ww_pick, mu_pick, sig_pick)

    # EM_iter function - iterations of the EM algorithm

    def my_EM_iter(self, x, y, pp_ini, mu_ini, sig_ini, draw, SIG_MIN, eps_change):
        # VALUES FOR CONSTANTS

        # threshold value for terminating iterations eps_change

        # draw = 1 - show graphically positions of means and standard deviations of components during iterartions
        # draw = 0 - omit show option;

        # SIG_MIN - minimum value of sigma squared
        x = self.x1
        y = self.y1

        draw = self.draw

        #SIG_MIN = par_sig_min #???

        eps_change = self.eps_par

        SIG_SQ = SIG_MIN * SIG_MIN

        #pp_ini = ???
        #mu_ini = ????
        #sig_ini = ????

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
                norm_pdf = 1 / (np.sqrt(sigkwvoc[kskla]) * np.sqrt(2 * math.pi)) * np.exp(
                    -1 * (x - mivoc[kskla]) ** 2 / 2 * np.sqrt(sigkwvoc[kskla]) ** 2)
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

            change = np.sum(np.abs(ppoc - oldppoc)) + np.sum(
                np.divide(((np.abs(sigkwvoc - oldsigkwvoc)), sigkwvoc))) / (len(ppoc))

            if draw == 1:
                plt.plot(mivoc, np.sqrt(sigkwvoc), '*')
                plt.xlabel('means')
                plt.ylabel('standard deviations')
                # plt.title(['Progress of the EM algorithm: change=' str(change)])

        # RETURN RESULTS
        l_lik = np.sum(np.multiply(math.log(denpss), y))
        mu_est = np.sort(mivoc)
        isort = np.argsort(mivoc)
        sig_est = np.sqrt(sigkwvoc[isort])
        pp_est = ppoc[isort]
        TIC = np.sum(y)
        bic = l_lik - ((3 * KS - 1) / 2) * math.log(TIC)
        return (pp_est, mu_est, sig_est, TIC, l_lik, bic)

    # auxiliary function - computing quality index for dynamic programming

    def my_qu_ix_w(self, invec, yinwec, par, par_sig_min):

        #par = ???
        #par_sig_min = gmm_decomp_split_segment.par_sig_min #????
        #invec = gmm_decomp_split_segment.invec #?????????
        #yinwec = gmm_decomp_split_segment.yinwec#??????????
        # invec = invec[:]
        # yinwec = yinwec[:]
        if (invec[len(invec)] - invec[1]) <= (par_sig_min or np.sum(yinwec) <= 1.0e-3):
            wyn = np.inf
        else:
            wwec = yinwec / (np.sum(yinwec))
            wyn1 = (par + np.sqrt(np.sum(np.dot(((invec - np.sum(np.dot(invec, wwec))) ** 2), wwec)))) / (
                        max(invec) - min(invec))
            wyn = wyn1
        return wyn1

    # plot MS signal and its GMM model

    def plot_gmm(self, x, y, ww_gmm, mu_gmm, sig_gmm):

        x = self.x1
        y = self.y1

        #ww_gmm = ????
        #mu_gmm = ???
        #sig_gmm = ????

        ploty = np.zeros(np.size(x))
        plt.plot(x, y, 'k')

        KS = len(ww_gmm)
        for kks in range(1, KS):
            ixmz = np.argwhere(np.abs((x - mu_gmm[kks]) / sig_gmm[kks]) < 4)
            norm_pdf = 1 / (sig_gmm[kks] * np.sqrt(2 * math.pi)) * np.exp(
                -1 * (x[ixmz] - mu_gmm[kks]) ** 2 / 2 * sig_gmm[kks] ** 2)
            ploty[ixmz] = ploty[ixmz] + ww_gmm[kks] * norm_pdf
            plt.plot(x[ixmz], ww_gmm[kks] * norm_pdf, 'g')

        plt.plot(x, ploty, 'r')

        plt.xlabel('M/Z')
        plt.ylabel('Intensity')

    # plot MS signal versus GMM model (used for for segments)
    def plot_res_new_scale(self, data, ygreki, wwoc, mivoc, sigvoc):

        #data = ???
        #ygreki = ???
        #wwocm = ???
        #mivoc = ??
        #sigvoc = ????
        xx = data
        y_a = ygreki

        ploty = np.zeros(np.size(xx))

        KS = len(wwoc)

        for kks in range(1, KS):
            norm_pdf = 1 / (sigvoc[kks] * np.sqrt(2 * math.pi)) * np.exp(
                -1 * (xx - mivoc[kks]) ** 2 / 2 * sigvoc[kks] ** 2)
            ploty = ploty + (wwoc[kks] * norm_pdf)

            plt.plot(xx, y_a, 'k', xx, ploty, 'r')

        for kks in range(1, KS):
            plt.plot(xx, wwoc[kks] * norm_pdf, 'g')

        plt.xlabel('m/z')


    # compute quality indices and scale of gmm model of the segment

    def qua_scal(self, data, ygreki, ppoc, mivoc, sigvoc):

        #data = ???
        #ygreki = ???
        #ppoc = ???
        #mivoc = ???
        #sigvoc = ???
        xx = data
        y_a = ygreki
        ploty = np.zeros(np.size(xx))

        KS = len(ppoc)

        for kks in range(1, KS):
            norm_pdf = 1 / (sigvoc[kks] * np.sqrt(2 * math.pi)) * np.exp(
                -1 * ((xx) - mivoc[kks]) ** 2 / 2 * sigvoc[kks] ** 2)
            ploty = ploty + ppoc[kks] * norm_pdf

        scale = np.sum(y_a) / np.sum(ploty)
        qua = np.sum(np.abs(y_a / scale - ploty)) / np.sum(y_a / scale)
        return (qua, scale)

    # demo script for for illustration of the algorithm for Gaussian mixture decomposition of protein MS signals

    def fit(self, x, y):

        add_path = 'ms_gmm'

        x = self.x1
        y = self.y1
        mean_y = np.mean(y, axis=0)  # create mean spectrum

        # initialization - remove baseline & trim above zero
        #YB = baseline_removals(x, mean_y)
        ixg0 = np.argwhere(mean_y > 0)
        y_b = np.array(ixg0)
        x = np.array(ixg0)

        # main function for GMM decomposition of MS signal
        ww_gmm, mu_gmm, sig_gmm = GaussianMixModel.gmm(self, x, y_b)

        # show results
        plt.figure(3)
        plt.plot_gmm(x, y_b, ww_gmm, mu_gmm, sig_gmm)
        plt.title('MS signal (black), GMM model (red), components (green)')

path = r'C:/Users/Shirin/ShirinsFolder/Study/GMMProject/GMM_Python/20220512-GMM-Datasets-more-complex-Copy.csv'
#path = r'C:/Users/Shirin/ShirinsFolder/Study/GMMProject/GMM_Python/20220512-GMM-Datasets-more-complex-Copy-Copy.csv'
reader = list(csv.reader(open(path, "r")))
x = list(reader)
result = np.array(x).astype('float')
#result1 = result.reshape(-1)
result1 = np.ravel(result)
#result2 = result.reshape(72,7)
data1 = np.array(result1).astype('int')
data1 = data1.reshape(72, 2)
#data1 = data1.reshape(9026, 2)

c = data1[:, 0]
v = data1[:, 1]
g = GaussianMixModel(result1)
g.fit(c, v)

