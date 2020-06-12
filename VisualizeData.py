import movielens20MLocal as dl
import movielens as dl1
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pylab as plt
from python_splitters import python_stratified_split
import sar_singlenode as SAR
from timer import Timer
from mf_keras import Mf_keras
from simple_splitter import split
MOVIELENS_DATA_SIZE = "100k"

data = dl1.load_pandas_df(MOVIELENS_DATA_SIZE)
print(len(data['userID']))
data = data.drop(columns=['timestamp'])
data = data.sort_values(by=['userID'])
N = data['userID'].max()
M = data['itemID'].max()
print(N)
print(M)
row = []
col = []
rating = []
for i in range(1, N):
    dataI = data[data['userID'] == i]
    dataI = dataI.sort_values(by=['itemID'])
    row = np.append(row, dataI['userID'].values - 1)
    col = np.append(col, dataI['itemID'].values - 1)
    rating = np.append(rating, dataI['rating'].values)

rating = np.array(rating)
col = np.array(col)
row = np.array(row)
print(row.shape)
print(col.shape)
print(rating.shape)


compMatrix = csr_matrix((rating, (row, col)), shape=(N, M))
plt.spy(compMatrix, markersize=0.5)
plt.legend()
plt.show()


