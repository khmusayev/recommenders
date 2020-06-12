import numpy as np
from MathUtil import sigmoid
from MathUtil import nextValueIndex
from scipy.special import softmax

class MFRecommender(object):

    def __init__(self, csr_train, csr_test, hidden_size = 10):
        self.csr_train = csr_train
        self.csr_test = csr_test
        self.N = csr_train.shape[0]
        self.M = csr_train.shape[1]
        self.K = hidden_size
        self.build()

    def build(self):
        # params
        self.W = np.zeros((self.N, self.K), dtype=np.float32)
        self.U = np.zeros((self.M, self.K), dtype=np.float32)
        self.b = np.zeros(self.N, dtype=np.float32)
        self.c = np.zeros(self.M, dtype=np.float32)
        self.loss_list = []
        self.mse_train_list = []
        self.mse_test_list = []
        self.mu = self.csr_train.mean()

    def valueAt(self, X):
        self.W, self.U, self.b, self.c = self.unpack2WUbc(X)
        print('Calculating value at....')
        res = self.loss_function()
        self.loss_list.append(res)
        self.mse_train_list.append(self.mse_train())
        self.mse_test_list.append(self.mse_test())
        return res


    def loss_function(self):
        return self.loss_function_(self.csr_train, self.mu)

    def loss_function_(self, csr_mat, mu):
        loss = 0
        rows, cols = csr_mat.nonzero()
        ratings = csr_mat.data
        count = 0
        size = len(ratings)
        print('size is ', size)
        for i in range(size):
            rij_prediction = self.predictRatingIJ(rows[i], cols[i], mu)
            loss += np.square(ratings[i] - rij_prediction)
        return loss

    def predictRatingIJ(self, i, j, mu):
        wtu = np.inner(self.W[i], self.U[j])
        return wtu + self.b[i] + self.c[j] + mu

    def gradientAt(self, X):
        self.W, self.U, self.b, self.c = self.unpack2WUbc(X)
        print('calculating gradient...')
        dlossdw, dlossdu, dlossdb, dlossdc = self.gradient()
        return self.flattenWUbc(dlossdw, dlossdu, dlossdb, dlossdc)

    def gradient(self):
        dlossdw = np.zeros(shape=(self.N, self.K), dtype=np.float)
        dlossdu = np.zeros(shape=(self.M, self.K), dtype=np.float)
        dlossdb = np.zeros(shape=(self.N), dtype=np.float)
        dlossdc = np.zeros(shape=(self.M), dtype=np.float)
        rows, cols = self.csr_train.nonzero()
        ratings = self.csr_train.data
        size = len(ratings)
        print('size is ', size)
        for k in range(size):
            i = rows[k]
            j = cols[k]
            rij_prediction = self.predictRatingIJ(i, j, self.mu)
            factor = -2*(ratings[k] - rij_prediction)
            dlossdw[i] += factor*self.U[j]
            dlossdu[j] += factor*self.W[i]
            dlossdb += factor
            dlossdc += factor
        return dlossdw, dlossdu, dlossdb, dlossdc


    def flattenWUbc(self, wl, ul, bl, cl):
        wlf = wl.flatten()
        ulf = ul.flatten()
        res = np.append(wlf, ulf)
        res = np.append(res, bl)
        res = np.append(res, cl)
        return res

    def unpack2WUbc(self, x):
        wl = x[:self.W.size].reshape(self.W.shape)
        ul = x[self.W.size:self.W.size+self.U.size].reshape(self.U.shape)
        bl = x[self.W.size+self.U.size:self.W.size+self.U.size + self.b.size].reshape(self.b.shape)
        cl = x[self.W.size + self.U.size + self.b.size:].reshape(self.c.shape)
        return wl, ul, bl, cl

    def update(self, rate, dw, du, db, dc):
        self.W = self.W + rate*dw
        self.U = self.U + rate*du
        self.b = self.b + rate * db
        self.c = self.c + rate * dc

    def mse(self, csr_mat):
        loss = self.loss_function_(csr_mat, self.mu)
        size = float(len(csr_mat.data))
        return np.sqrt(loss)/size

    def mse_singleUser(self, ratedMovieInd, movieratings):
        if len(ratedMovieInd) == 0: return 0
        v = self.generateVisibleLayer(ratedMovieInd, movieratings)
        v_prime = self.cd_k(v, ratedMovieInd)
        res = np.zeros(shape=movieratings.shape, dtype=np.float)
        count = 0
        for i in np.nditer(ratedMovieInd):
            indexOfRating = np.argmax(v_prime[i])
            indOfRating = float(indexOfRating)
            rat = (indexOfRating + 1.0) * 0.5
            res[count] = rat
            count = count + 1
        return np.sum(np.square(movieratings - res))

    def mse_train(self):
        return self.mse(self.csr_train)

    def mse_test(self):
        return self.mse(self.csr_test)

    def generateZeroInitialPoint(self):
        # params
        wl = np.zeros((self.N, self.K), dtype=np.float32)
        ul = np.zeros((self.M, self.K), dtype=np.float32)
        bl = np.zeros(self.N, dtype=np.float32)
        cl = np.zeros(self.M, dtype=np.float32)
        return self.flattenWUbc(wl, ul, bl, cl)


#r = RBMRecommender(50, [10000, 10])
#r.generateRandomWbc()
#visible = r.generateRandomVisibleLayer()
#h1_v1 = r.probOf_hj1_givenV(visible)
#hidden = r.generateRandomHiddenLayer()
#v1_h1 = r.probOf_vik1_givenh(hidden)
#print(v1_h1.sum(axis=1))

#r.sample_hidden(h1_v1)
#visible_sample = r.sample_visible(v1_h1)
#print(visible_sample.sum(axis=1))
#print("done")
