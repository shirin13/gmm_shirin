{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "56a6ac9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "import scipy.stats as sp\n",
    "\n",
    "from scipy import sparse\n",
    "\n",
    "import math\n",
    "\n",
    "from scipy.sparse.linalg import spsolve\n",
    "        \n",
    "from scipy.linalg import cholesky\n",
    "\n",
    "from scipy.stats import norm\n",
    "\n",
    "from scipy.signal import find_peaks, peak_widths\n",
    "\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9d8f9a29",
   "metadata": {},
   "outputs": [],
   "source": [
    "#np.seterr(divide='ignore', invalid='ignore')\n",
    "class GaussianMixModel(object):\n",
    "    def __init__(self, data):\n",
    "\n",
    "        self.data = data\n",
    "\n",
    "        self.x, self.y = data[:, 0], data[:, 1:]\n",
    "\n",
    "        self.m, self.n = data.shape\n",
    "        \n",
    "        self.res_par= 0.5 #res_par - used in the main body of ms_gmm - multiplied by estimated average width of the peak in the spectrum defines resolutioon of the  decomposition\n",
    "\n",
    "        self.par_p_sens = 0 # par_p_sens parameter for peak detection sensitivity used in find_split_peaks split peaks must be of height >= Par_P_Sens * maximal peak height\n",
    "\n",
    "        self.par_q_thr= 1.3 # par_q_thr parameter for peak quality threshold used in find_split_peaks split peaks must have quality >= Par_Q_Thr\n",
    "\n",
    "        self.par_ini_j = 5 # par_ini_j parameter for initial jump used in find_split_peaks\n",
    "\n",
    "        self.par_p_l_r = 4 # par_p_l_r parameter for range for peak lookup used in find_split_peaks\n",
    "\n",
    "        self.par_p_j = 4 # par_p_j parameter for jump used in find_split_peaks\n",
    "\n",
    "        self.qfpar = 0.5 #qfpar - parameter used in the dynamic programming quality funtion\n",
    "\n",
    "        self.prec_par_1 = 0.002 # prec_par_1 - precision parameter - weight used to pick best gmm decomposition penalty coefficient for number of components in the quality funtio\n",
    "        \n",
    "        self.res_par_2 = 0.5# res_par_2 - used in the EM iterations to define lower bounds for standard deviations\n",
    "\n",
    "        self.penet_par_1 = 15 # penet_par_1  - penetration parameter 1 used to continue searching for best number of components in gmm decomposition (bigger for splitting segments)\n",
    "        \n",
    "        self.prec_par_2 = self.penet_par_1 # prec_par_2 - precision parameter 2 - weight used to pick best gmm decomposition\n",
    "        \n",
    "        self.prec_par_2 = 15 # penet_par_1  - penetration parameter 2 used to continue searching for best number of components in gmm decomposition (smaller for segments)\n",
    "\n",
    "        self.buf_size_split_par = 10 # buf_size_split_par - size of the buffer for computing GMM paramters of splitters\n",
    "        \n",
    "        self.buf_size_seg_par = 30 # buf_size_seg_par - size of the buffer for computing GMM paramters of segments\n",
    "        \n",
    "        self.eps_par = 0.0001 # eps_par - parameter for EM iterations - tolerance for mixture parameters change\n",
    "\n",
    "        self.draw = 1 # draw - show plots of decompositions during computations used in many finctions\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
