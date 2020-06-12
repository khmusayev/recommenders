# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from datetime import datetime


class MFtf(object):
    def __init__(self, csr_train, csr_test, hidden_size = 10):
        self.csr_train = csr_train
        self.csr_test = csr_test
        self.N = csr_train.shape[0]
        self.M = csr_train.shape[1]
        self.K = hidden_size
        self.build()


    def build(self):
        # params
        self.W = []
        self.U = []
        self.b = []
        self.c = []
        self.train_sse = []
        self.test_sse = []
        for i in range(self.N):
            self.W.append(tf.Variable(tf.zeros([self.K])))
            self.b.append(tf.Variable(0.0, dtype=tf.float32))
        for j in range(self.M):
            self.U.append(tf.Variable(tf.zeros([self.K])))
            self.c.append(tf.Variable(0.0, dtype=tf.float32))
        mu = self.csr_train.mean()
        # build the objective
        print('Define loss function...')
        objective = self.loss_function(self.csr_train, mu)
        print('loss function defined.')
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(objective)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(objective)

        # build the cost
        # we won't use this to optimize the model parameters
        # just to observe what happens during training

        print('calculate train sse')
        # for calculating SSE
        self.sse = self.loss_function(self.csr_train, mu)
        print('calculate test sse')
        # test SSE
        self.tsse = self.loss_function(self.csr_test, mu)


        initop = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(initop)

    def loss_function(self, csr_mat, mu):
        loss = 0
        rows, cols = csr_mat.nonzero()
        ratings = csr_mat.data
        count = 0
        size = len(ratings)
        print('size is ', size)
        for i in range(size):
            rij_prediction = self.predictRatingIJ(rows[i], cols[i], mu)
            loss += tf.square(ratings[i] - rij_prediction)
            print('iteration inside loss function definition ', i)
        print('loss function defined')
        return loss

    def predictRatingIJ(self, i, j, mu):
        wtu = tf.tensordot(self.W[i], self.U[j], axes=0)
        return wtu + self.b[i] + self.c[j] + mu

    def fit(self, epochs=10, show_fig=True):
        for i in range(epochs):
            t0 = datetime.now()
            print("epoch:", i)
            _, c = self.session.run(
                (self.train_op, self.cost)
            )


            print("duration:", datetime.now() - t0)

            # calculate the true train and test cost
            t0 = datetime.now()
            sse = 0
            test_sse = 0
            n = 0
            test_n = 0

            # use tensorflow to get SSEs
            sse = self.loss_function(self.csr_train)
            tsse = self.loss_function(self.csr_test)
            self.train_sse.append(sse)
            self.test_sse.append(tsse)
            print("calculate cost duration:", datetime.now() - t0)
        if show_fig:
            plt.plot(self.train_sse, label='train sse')
            plt.plot(self.test_sse, label='test sse')
            plt.legend()
            plt.show()



def main():
    train = load_npz("train.npz")
    test = load_npz("test.npz")
    K = 10
    N, M = train.shape
    mf = MFtf(train, test, hidden_size=K)
    mf.fit(epochs=10, show_fig=True)


if __name__ == '__main__':
    main()
