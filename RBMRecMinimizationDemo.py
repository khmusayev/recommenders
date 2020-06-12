from DataExtractUtil import csrMovielensSplit, visualize
from RBMRecommender import RBMRecommender
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

train, test = csrMovielensSplit()
M = 50
D = train.get_shape()[1]
K = 10
rec = RBMRecommender(M, [D, K], train, test)
x0 = rec.generateRandomWbcAsX()


result = minimize(rec.valueAt, x0=x0, method='cg', jac=rec.gradientAt, options={'maxiter':100, 'disp':True})

plt.plot(rec.loss_list)
plt.ylabel('loss function')
plt.xlabel('iteration')
plt.show()

plt.plot(rec.mse_train_list)
plt.ylabel('train mse')
plt.xlabel('iteration')
plt.show()

plt.plot(rec.mse_test_list)
plt.ylabel('test mse')
plt.xlabel('iteration')
plt.show()