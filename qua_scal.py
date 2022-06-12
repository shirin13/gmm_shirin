import numpy as np
import math

# compute quality indices and scale of gmm model of the segment

def qua_scal(self, data, ygreki, ppoc, mivoc, sigvoc):
    xx = data
    y_a = ygreki
    ploty = 0 * xx

    KS = len(ppoc)

    for kks in range(1, KS):
        norm_pdf = 1/(sigvoc[kks] * np.sqrt(2 * math.pi)) * np.exp(-1 * ((xx) - mivoc[kks]) ** 2 / 2 * sigvoc[kks]** 2)
        ploty = ploty + ppoc[kks] * norm_pdf

    scale = np.sum(y_a) / np.sum(ploty)
    qua = np.sum(np.abs(y_a / scale-ploty)) / np.sum(y_a / scale)
    return(qua, scale)

