
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

    def bindata(self, data, bins):
        # function [ym,yb] = bindata(y,x,xrg)
        # Computes ym(ii) = mean(y(x>=xrg(ii) & x < xrg(ii+1)) for every ii
        # using a fast algorithm which uses no looping
        # If a bin is empty it returns nan for that bin
        # Also returns yb, the approximation of y using binning (useful for r^2 calculations). Example:
        # x = randn(100,1)
        # %y = x.^2 + randn(100,1)
        # xrg = linspace(-3,3,10)'
        # [ym,yb] = bindata(y,x,xrg)
        # X = [xrg(1:end-1),xrg(2:end)]'
        # Y = [ym,ym]'
        # plt.plot(x,y,'.',X(:),Y(:),'r-')
        data = self.data
        y = baseline_removals.y_base
        x = self.x1
        #print(y)
        #print(y.shape)
        for i in range(y.shape[0]):
            try:
                bins = np.linspace(y.min(), y.max(), 100, endpoint=False)
        # digitized = np.digitize(y, bins) #histc
        # returns ind, an array the same size as x indicating the bin number that each entry in x sorts into. Use this syntax with any of the previous input argument combinations.
        #bin_means = [y[digitized == i].mean() for i in range(1, len(bins))]
                bin_means = (np.histogram(y, bins, weights=y)[0] / np.histogram(y, bins)[0])

            except ZeroDivisionError:
                return 0
            return bin_means


path = r'C:/Users/Shirin/ShirinsFolder/Study/GMMProject/GMM_Python/20220512-GMM-Datasets-more-complex.csv'


# with open(path, 'r', newline='') as f:
#     reader = list(csv.reader(f, delimiter='\t'))
#     #header = next(reader)
#     #rows = [header] + [[row[0], int(row[1])] for row in reader if row]

reader = list(csv.reader(open(path, "r")))
x = list(reader)
result = np.array(x).astype('int')
#result1 = result.reshape(-1)
result1 = np.ravel(result)
#result2 = result.reshape(72,7)

g = GaussianMixModel(result1)
g.bindata(result1, 100)
