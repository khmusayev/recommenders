# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

class Mf_keras:
  def __init__(
          self,
          N = None,
          M = None
          ):

    self.N = N
    self.M = M

  def fit(self, df_train, df_test):

    # initialize variables
    K = 2000 # latent dimensionality
    mu = df_train.rating.mean()
    epochs = 10
    reg = 0. # regularization penalty


    # keras model
    u = Input(shape=(1,))
    m = Input(shape=(1,))
    u_embedding = Embedding(self.N, K, embeddings_regularizer=l2(reg))(u) # (N, 1, K)
    m_embedding = Embedding(self.M, K, embeddings_regularizer=l2(reg))(m) # (N, 1, K)
    u_bias = Embedding(self.N, 1, embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
    m_bias = Embedding(self.M, 1, embeddings_regularizer=l2(reg))(m) # (N, 1, 1)
    x = Dot(axes=2)([u_embedding, m_embedding]) # (N, 1, 1)

    x = Add()([x, u_bias, m_bias])
    x = Flatten()(x) # (N, 1)

    model = Model(inputs=[u, m], outputs=x)
    model.compile(
      loss='mse',
      # optimizer='adam',
      # optimizer=Adam(lr=0.01),
      optimizer=SGD(lr=0.08, momentum=0.9),
      metrics=['mse'],
      )

    r = model.fit(
      x=[df_train.userID.values, df_train.itemID.values],
      y=df_train.rating.values - mu,
      epochs=epochs,
      batch_size=128,
      validation_data=(
      [df_test.userID.values, df_test.itemID.values],
      df_test.rating.values - mu
      )
    )


    # plot losses
    plt.plot(r.history['loss'], label="train loss")
    plt.plot(r.history['val_loss'], label="test loss")
    plt.legend()
    plt.show()

    # plot mse
    plt.plot(r.history['mean_squared_error'], label="train mse")
    plt.plot(r.history['val_mean_squared_error'], label="test mse")
    plt.legend()
    plt.show()
