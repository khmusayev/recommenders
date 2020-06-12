from DataExtractUtil import csrMovielensSplit, visualize
from MFRecommender import MFRecommender
import matplotlib.pyplot as plt
import numpy as np

train, test = csrMovielensSplit()
K = 10
rec = MFRecommender(train, test, hidden_size=K)
rec.mse_train()
print('train mse completed')
loss = rec.loss_function()
dw, du, db, dc = rec.gradient()

print('starting to investigate the problem')
loss = np.zeros(shape=(10), dtype=np.float)
train_mse = np.zeros(shape=(10), dtype=np.float)
test_mse = np.zeros(shape=(10), dtype=np.float)
learning_rate = 0.0000001
for i in range(10):
    fi = float(i)
    rec.update(-learning_rate*fi, dw, du, db, dc)
    loss[i] = rec.loss_function()
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


