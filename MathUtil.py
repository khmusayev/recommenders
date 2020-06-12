import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def nextValueIndex(rows, count):
    n = rows.shape[0]
    if count == n - 1: return count
    for i in range(count, n):
        if rows[count] != rows[count + 1]:
            return count + 1
        else:
            count = count + 1
            if(count == n - 1):
                return count
