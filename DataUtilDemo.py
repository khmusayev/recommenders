from DataExtractUtil import csrMovielensSplit, visualize
from RBMRecommender import RBMRecommender
import matplotlib.pyplot as plt
import numpy as np

train, test = csrMovielensSplit()
M = 50
D = train.get_shape()[1]
K = 10
rec = RBMRecommender(M, [D, K], train, test)
rec.mse_train()
print('train mse completed')
loss = rec.total_loss_function()
a, b, c = rec.gradient()

print('starting to investigate the problem')
loss = np.zeros(shape=(10), dtype=np.float)
train_mse = np.zeros(shape=(10), dtype=np.float)
test_mse = np.zeros(shape=(10), dtype=np.float)
learning_rate = 0.000002
for i in range(10):
    fi = float(i)
    rec.update(-learning_rate*fi, a, b, c)
    loss[i] = rec.total_loss_function()
    train_mse[i] = rec.mse_train()
    test_mse[i] = rec.mse_test()
    print(i)

plt.plot(loss, )
plt.ylabel('loss function')
plt.xlabel('iteration')
plt.show()

plt.plot(train_mse)
plt.ylabel('loss function')
plt.xlabel('iteration')
plt.show()

plt.plot( test_mse)
plt.ylabel('loss function')
plt.xlabel('iteration')
plt.show()


