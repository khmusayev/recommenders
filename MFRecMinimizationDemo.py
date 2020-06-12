from DataExtractUtil import csrMovielensSplit, visualize
from MFRecommender import MFRecommender
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

train, test = csrMovielensSplit()
K = 10
rec = MFRecommender(train, test, hidden_size=K)
x0 = rec.generateZeroInitialPoint()


result = minimize(rec.valueAt, x0=x0, method='cg', jac=rec.gradientAt, options={'maxiter':20, 'disp':True})

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