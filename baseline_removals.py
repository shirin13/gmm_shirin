
import csv
import numpy as np

import scipy.stats as sp

from scipy import sparse

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
        data1 = data1.reshape(72, 8)

        self.x = data1[:, 0]
        self.y = data1[:, 1:]

        #self.y = np.ravel(self.y)

        #self.y = np.reshape(self.y, (72,7))
        #self.y = self.y.reshape(-1,7)
        self.y1 = self.y.T
        self.x1 = self.x.reshape(1, -1)

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

    def baseline_removals(self, data, lam=1e4, ratio=0.05, itermax=100):
        """
        Baseline correction using asymmetrically
        reweighted penalized least squares smoothing
        Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
        Analyst, 2015, 140, 250 (2015)
        y:
                input data (i.e. chromatogram of spectrum)
            lam:
                parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
            ratio:
                wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
            itermax:
                number of iterations to perform
        Output:
            the fitted background vector

        """
        data = self.data
        y = self.y1
        x = self.x1
        #y_base = 0
        for i in range(len(y[0])):
            N = len(y[0])
            D = sparse.eye(N, format='csc')
            # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
            D = D[1:] - D[:-1]
            D = D[1:] - D[:-1]
            H = lam * D.T * D
            w = np.ones(N)
            #w = np.squeeze(np.asarray(w))
            # offset = np.zeros(7)

            for j in range(itermax):
                W = sparse.diags(w, 0, shape=(N, N)) # (72,) (72, 2)
                WH = sparse.csc_matrix(W + H)
                C = sparse.csc_matrix(cholesky(WH.todense()))
                tmp = y[i, :] * w
                gh = spsolve(C.T, tmp.T)
                y_base = spsolve(C, gh)# spsolve(C, spsolve(C.T, w * y))
                d = y - y_base
                dn = d[d < 0]
                m = np.mean(dn)
                s = np.std(dn)
                wt = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))  #(72,7)
                if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                    break
                w = wt
            return y_base


path = r'C:/Users/Shirin/ShirinsFolder/Study/GMMProject/GMM_Python/20220512-GMM-Datasets-more-complex.csv'


# with open(path, 'r', newline='') as f:
#     reader = csv.reader(f, delimiter='\t')
#     #header = next(reader)
#     #rows = [header] + [[row[0], int(row[1])] for row in reader if row]

reader = list(csv.reader(open(path, "r")))
x = list(reader)
result = np.array(x).astype('int')
#result1 = result.reshape(-1)
result1 = np.ravel(result)
#result2 = result.reshape(72,7)

g = GaussianMixModel(result1)
g.baseline_removals(result1, lam=1e4, ratio=0.05, itermax=100)
