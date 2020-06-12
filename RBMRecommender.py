import numpy as np
from MathUtil import sigmoid
from MathUtil import nextValueIndex
from scipy.special import softmax

class RBMRecommender(object):

    def __init__(self, hidden_size, visible_size, csr_train, csr_test):
        self.M = hidden_size
        D, K = visible_size
        self.D = D
        self.K = K
        self.N = csr_train.shape[0]
        self.train = csr_train
        self.test = csr_test
        self.loss_list = []
        self.mse_train_list = []
        self.mse_test_list = []
        self.build()

    def build(self):
        self.W = np.zeros((self.D, self.K, self.M))
        self.b = np.zeros((self.D, self.K))
        self.c = np.zeros(self.M)
        self.generateRandomWbc()
        print('Visible size is ' + str(self.D) + ', ' + str(self.K))
        print("Hidden size is " + str(self.M))
        print('build')

    def total_loss_function(self):
        loss = 0
        rows, cols = self.train.nonzero()
        ratings = self.train.data
        count = 0
        for i in range(self.N - 1):
            if i == self.N - 1:
                print('stop')
            end = nextValueIndex(rows, count)
            ratedMovieInd = cols[count : end]
            movieratings = ratings[count : end] *2 - 1
            movieratings = movieratings.astype(int)
            loss += self.loss_function(self.generateVisibleLayer(ratedMovieInd, movieratings), ratedMovieInd)
            if count == end:
                print('stop')
            count = end
        return loss

    def valueAt(self, X):
        self.b, self.c, self.w = self.unpack2Wbc(X)
        print('Calculating value at....')
        res = self.total_loss_function()
        self.loss_list.append(res)
        self.mse_train_list.append(self.mse_train())
        self.mse_test_list.append(self.mse_test())
        return res

    def gradientAt(self, X):
        self.b, self.c, self.w = self.unpack2Wbc(X)
        print('calculating gradient...')
        dlossdb, dlossdc, dlossdw = self.gradient()
        return self.flattenbcw(dlossdb, dlossdc, dlossdw)

    def gradient(self):
        dlossdb = np.zeros(shape=(self.D, self.K), dtype=np.float)
        dlossdc = np.zeros(shape=(self.M), dtype=np.float)
        dlossdw = np.zeros(shape=(self.D, self.K, self.M), dtype=np.float)
        rows, cols = self.train.nonzero()
        ratings = self.train.data
        count = 0
        for i in range(self.N - 1):
            if i == self.N - 1:
                print('stop')
            end = nextValueIndex(rows, count)
            ratedMovieInd = cols[count: end]
            movieratings = ratings[count: end] * 2 - 1
            movieratings = movieratings.astype(int)
            db, dc, dw = self.dLossdb(self.generateVisibleLayer(ratedMovieInd, movieratings), ratedMovieInd)
            dlossdb += db
            dlossdc += dc
            dlossdw += dw
            if count == end:
                print('stop')
            count = end
        return dlossdb, dlossdc, dlossdw


    def probOf_hj1_givenV(self, V):
        W2dotV = np.tensordot(self.W, V, axes = ([0,1], [0, 1]))
        W2dotVAddC = np.add(W2dotV, self.c)
        return sigmoid(W2dotVAddC)

    def probOf_vik1_givenh(self, h):
        W2dotH = np.tensordot(self.W, h, axes=([2], [0]))
        W2dotHAddB = np.add(W2dotH, self.b)
        return softmax(W2dotHAddB, axis=1)

    def loss_function(self, v, ratedMovieInd):
        v_prime = self.cd_k(v, ratedMovieInd)
        return self.free_energy(v) - self.free_energy(v_prime)

    def dLossdb(self, v, ratedMovieInd):
        v_prime = self.cd_k(v, ratedMovieInd)
        db, dc, dw = self.dFdb(v)
        db_p, dc_p, dw_p = self.dFdb(v_prime)
        return db - db_p, dc - dc_p, dw - dw_p

    def free_energy(self, v):
        first_part = np.tensordot(self.b, v, axes=([0, 1], [0, 1]))
        second_part = np.sum(np.log(1 + np.exp(np.add(np.tensordot(v, self.W, axes=([0, 1], [0, 1])), self.c))))
        return -first_part - second_part

    def dFdb(self, v):
        db = -v
        vdotw = np.tensordot(v, self.W, axes=([0, 1], [0, 1]))
        vdotwplusc = np.add(vdotw, self.c)

        denom = 1.0 + np.exp(-vdotwplusc)
        dc = -1.0/denom
        dw = np.tensordot(v, dc, axes=0)
        return db, dc, dw

    def cd_k(self, v, ratedMovieInd):
        prob_h1_v1 = self.probOf_hj1_givenV(v)
        h1 = self.sample_hidden(prob_h1_v1)
        prob_v2_h1 = self.probOf_vik1_givenh(h1)
        v2 = self.sample_visible(prob_v2_h1, ratedMovieInd)
        return v2

    def sample_hidden(self, prob_h1_v1):
        rand = np.random.uniform(0.0, 1.0, prob_h1_v1.shape[0])
        res = np.zeros(prob_h1_v1.shape[0])
        for i in range(prob_h1_v1.shape[0]):
            if rand[i] < prob_h1_v1[i] : res[i] = 1
        return res

    def sample_visible(self, prob_v2_h1, ratedMovieInd):
        res = np.zeros(shape=(self.D, self.K), dtype=np.int)
        if len(ratedMovieInd)==0:
            return res
        for i in np.nditer(ratedMovieInd):
            res[i, np.argmax(prob_v2_h1[i])] = 1
        return res

    def system_energy(self, v, h):
        vw = np.tensordot(v, self.W, axes=([0, 1], [0, 1]))
        vwh = np.tensordot(vw, h, axes=([0], [0]))
        bv = np.tensordot(self.b, v, axes=([0, 1], [0, 1]))
        ch = np.tensordot(self.c, h, axes=([0], [0]))
        return -(vwh + bv + ch)

    def generateRandomWbc(self):
        a = self.W.size
        self.W = np.random.uniform(low=-0.01, high=0.01, size=(self.W.shape[0], self.W.shape[1], self.W.shape[2]))
        self.b = np.random.uniform(low=-0.01, high=0.01, size=(self.b.shape[0], self.b.shape[1]))
        self.c = np.random.uniform(low=-0.01, high=0.01, size=(self.c.shape[0]))
        return self.b, self.c, self.W

    def flattenbcw(self, bl,cl, wl):
        blf = bl.flatten()
        wlf = wl.flatten()
        res = np.append(blf, cl)
        res = np.append(res, wlf)
        return res

    def flattenWbc(self):
        blf = self.b.flatten()
        wlf = self.W.flatten()
        res = np.append(blf, self.c)
        res = np.append(res, wlf)
        return res

    def unpack2Wbc(self, x):
        bl = x[:self.b.size].reshape(self.b.shape)
        cl = x[self.b.size:self.b.size+self.c.size].reshape(self.c.shape)
        wl = x[self.b.size+self.c.size:].reshape(self.W.shape)
        return bl, cl, wl

    def generateRandomWbcAsX(self):
        self.generateRandomWbc()
        return self.flattenWbc()

    def generateRandomHiddenLayer(self):
        res = np.random.randint(low = 0, high = 2, size = self.M)
        return res

    def generateRandomVisibleLayer(self):
        res = np.random.randint(low = 0, high = 2, size = (self.D, self.K))
        return res

    def generateVisibleLayer(self, movieRatingInd, movieRatings):
        res = np.zeros(shape=(self.D, self.K), dtype=np.int)
        size = movieRatings.shape[0]
        for i in range(size):
            res[movieRatingInd[i]][movieRatings[i]] = 1
        return res

    def update(self, rate, db, dc, dw):
        self.W = self.W + rate*dw
        self.b = self.b + rate * db
        self.c = self.c + rate * dc

    def mse(self, csr_mat):
        numberofratings = 0.0
        mse = 0
        rows, cols = csr_mat.nonzero()
        ratings = csr_mat.data
        count = 0
        for i in range(self.N - 1):
            if i == self.N - 1:
                print('stop')
            end = nextValueIndex(rows, count)
            ratedMovieInd = cols[count : end]
            movieratings = ratings[count : end] *2 - 1
            movieratings = movieratings.astype(int)
            mse += self.mse_singleUser(ratedMovieInd, movieratings)
            if count == end:
                print('stop')
            count = end
            numberofratings += float(len(movieratings))
        return np.sqrt(mse/numberofratings)

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
        return self.mse(self.train)

    def mse_test(self):
        return self.mse(self.test)



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
