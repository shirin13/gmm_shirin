import numpy as np
import math

# compute quality indices and scale of gmm model of the segment

def qua_scal(data, ygreki, ppoc, mivoc, sigvoc):
    xx = data
    y_a = ygreki
    ploty = np.zeros(np.size(xx)) #0 * xx

    KS = len(ppoc)
    print (KS)

    for kks in range(1, KS):
        norm_pdf = 1/(sigvoc[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * ((xx) - mivoc[kks]) ** 2 / 2 * sigvoc[kks]** 2)
        ploty = ploty + ppoc[kks] * norm_pdf

    scale = np.sum(y_a) / np.sum(ploty)
    qua = np.sum(np.abs(y_a / scale-ploty)) / np.sum(y_a / scale)
    return(qua, scale)

y = np.array([22, 92, 178, 340, 538, 661, 685, 582, 423, 312, 272, 354, 563, 921, 1404, 1870, 2204, 2195, 2019, 1858, 1928, 2344, 3103, 3923, 4692, 5140, 4973, 4627, 4428, 4461, 4856, 5336, 5926, 6423, 6696, 6687, 6410, 5753, 5221, 4609, 4220, 4142, 4152, 4235, 4514, 4764, 5250, 5838, 6835, 7910, 8910, 9780, 10488, 10779,  10339, 9420, 8457, 6964, 5610, 4384, 3356, 2302, 1753, 1001, 746, 248, 136, 54, -364, -257, -543, -494])
x = np.array([450,500,505,510,515,520,525,530,535,540,545,550,555,560,565,570, 575,580,585,590,595,600,605,610,615,620,625,630,635,640,645,650,655,660,665,670,675,680,685,690,695,700,705,710,715,720,725,730,735,740,745,750,755,760,765,770,775,780,785,790,795,800,805,810, 815,820,825,830,835,840,845,850])
N = len(y)
mean = y.sum() / N
std = np.sqrt(np.sum((y - mean)**2) / N)
w = np.array(np.size(N))

q = qua_scal(x, y, w , mean, std)