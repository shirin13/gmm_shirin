import csv
import numpy as np


import sklearn as sk

import scipy

from BaselineRemoval import BaselineRemoval

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
data1 = np.array(result1).astype('int')
data1 = data1.reshape(72, 8)

x = data1[:, 0]
y = data1[:, 1:]

#self.y = np.ravel(self.y)

#self.y = np.reshape(self.y, (72,7))
#self.y = self.y.reshape(-1,7)
y1 = y.T
print(y1)
