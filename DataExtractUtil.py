import pandas as pd#
import numpy as np
import movielens as dl1
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import matplotlib.pylab as plt
from simple_splitter import split
import movielens20MLocal as dl
import movielens as dl1
import os


def visualize(csr_data, markersize=0.5):
    plt.spy(csr_data, markersize=markersize)
    plt.legend()
    plt.show()

def splitTrainTestCsrMatrixFormat(df, dropColumns=['timestamp'], rowName='userID', columnName='itemID',
                                  valueName='rating', ratio=0.8):
    N = df[rowName].max()
    M = df[columnName].max()
    train, test = split(df, ratio=ratio)
    csr_train = csrMatrix(train, N, M, dropColumns=dropColumns, rowName=rowName, columnName=columnName,
                          valueName=valueName)
    csr_test = csrMatrix(test, N, M, dropColumns=dropColumns, rowName=rowName, columnName=columnName,
                         valueName=valueName)
    return csr_train, csr_test

def splitMovielensDataSetCsrFormat(MOVIELENS_DATA_SIZE='100k', dropColumns = ['timestamp'], rowName = 'userID', columnName = 'itemID', valueName = 'rating', ratio=0.8):
    df = dl1.load_pandas_df(MOVIELENS_DATA_SIZE)
    return splitTrainTestCsrMatrixFormat(df, dropColumns = dropColumns, rowName = rowName, columnName = columnName, valueName = valueName, ratio = ratio)

def csrMovielensSplit(MOVIELENS_DATA_SIZE='100k', dropColumns=['timestamp'], rowName='userID', columnName='itemID',
                      valueName='rating', ratio=0.8):
    train_path = 'train.npz'
    test_path = 'test.npz'
    if os.path.exists(train_path) & os.path.exists(train_path):
        train_data = load_npz(train_path)
        test_data = load_npz(test_path)
        return train_data, test_data
    else:
        train_data, test_data = splitMovielensDataSetCsrFormat(MOVIELENS_DATA_SIZE=MOVIELENS_DATA_SIZE,
                                                               dropColumns=dropColumns, rowName=rowName,
                                                               columnName=columnName, valueName=valueName, ratio=ratio)
        save_npz(file=train_path, matrix=train_data)
        save_npz(file=test_path, matrix=test_data)
        return train_data, test_data

def csrMatrix(df, N, M, dropColumns = ['timestamp'], rowName = 'userID', columnName = 'itemID', valueName = 'rating'):
    data = df.drop(columns=dropColumns)
    data = data.sort_values(by=[rowName])
    row = []
    col = []
    rating = []
    for i in range(1, N):
        dataI = data[data[rowName] == i]
        dataI = dataI.sort_values(by=[columnName])
        row = np.append(row, dataI[rowName].values - 1)
        col = np.append(col, dataI[columnName].values - 1)
        rating = np.append(rating, dataI[valueName].values)

    rating = np.array(rating)
    col = np.array(col)
    row = np.array(row)
    print(row.shape)
    print(col.shape)
    print(rating.shape)

    return csr_matrix((rating, (row, col)), shape=(N, M))








