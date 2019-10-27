import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from geopy.distance import great_circle as vc
import math as Math
from os import path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.externals import joblib 
import math, time
import keras

def model():
#Generate Data

#Cat 5
  cat_arr_5 = np.full((3000, 1), 5)

  dist_close_arr_5 = np.random.randint(50, size=1000)
  dist_mid_arr_5 = np.random.randint(51, 125, size=1000)
  dist_far_arr_5 = np.random.randint(126, 270, size=1000)

  dam_close_arr_5 = np.random.randint(600000000, 1400000000, size=1000)
  dam_mid_arr_5 = np.random.randint(125000000, 500000000, size=1000)
  dam_far_arr_5 = np.random.randint(50000000, 100000000, size=1000)

  a = np.concatenate((dist_close_arr_5, dist_mid_arr_5, dist_far_arr_5)).reshape(3000,1)
  b = np.concatenate((dam_close_arr_5, dam_mid_arr_5, dam_far_arr_5)).reshape(3000,1)
  final5 = np.concatenate((cat_arr_5, a, b), axis = 1)



  #Cat 4
  cat_arr_4 = np.full((3000, 1), 4)
  dist_close_arr_4 = np.random.randint(50, size=1000)
  dist_mid_arr_4 = np.random.randint(51, 125, size=1000)
  dist_far_arr_4 = np.random.randint(126, 270, size=1000)

  dam_close_arr_4 = np.random.randint(80000000, 100000000, size=1000)
  dam_mid_arr_4 = np.random.randint(30000000, 50000000, size=1000)
  dam_far_arr_4 = np.random.randint(15000000, 25000000, size=1000)

  a = np.concatenate((dist_close_arr_4, dist_mid_arr_4, dist_far_arr_4)).reshape(3000,1)
  b = np.concatenate((dam_close_arr_4, dam_mid_arr_4, dam_far_arr_4)).reshape(3000,1)
  final4 = np.concatenate((cat_arr_4, a, b), axis = 1)

  #Cat 3
  cat_arr_3 = np.full((3000, 1), 3)

  dist_close_arr_3 = np.random.randint(50, size=1000)
  dist_mid_arr_3 = np.random.randint(51, 125, size=1000)
  dist_far_arr_3 = np.random.randint(126, 270, size=1000)

  dam_close_arr_3 = np.random.randint(50000000, 75000000, size=1000)
  dam_mid_arr_3 = np.random.randint(20000000, 38000000, size=1000)
  dam_far_arr_3 = np.random.randint(10000000, 15000000, size=1000)

  a = np.concatenate((dist_close_arr_3, dist_mid_arr_3, dist_far_arr_3)).reshape(3000,1)
  b = np.concatenate((dam_close_arr_3, dam_mid_arr_3, dam_far_arr_3)).reshape(3000,1)
  final3 = np.concatenate((cat_arr_3, a, b), axis = 1)

  #Cat 2
  cat_arr_2 = np.full((3000, 1), 2)

  dist_close_arr_2 = np.random.randint(50, size=1000)
  dist_mid_arr_2 = np.random.randint(51, 125, size=1000)
  dist_far_arr_2 = np.random.randint(126, 270, size=1000)

  dam_close_arr_2 = np.random.randint(33000000, 55000000, size=1000)
  dam_mid_arr_2 = np.random.randint(10000000, 22000000, size=1000)
  dam_far_arr_2 = np.random.randint(3000000, 9800000, size=1000)

  a = np.concatenate((dist_close_arr_2, dist_mid_arr_2, dist_far_arr_2)).reshape(3000,1)
  b = np.concatenate((dam_close_arr_2, dam_mid_arr_2, dam_far_arr_2)).reshape(3000,1)
  final2 = np.concatenate((cat_arr_2, a, b), axis = 1)

  #Cat 1
  cat_arr_1 = np.full((3000, 1), 1)

  dist_close_arr_1 = np.random.randint(50, size=1000)
  dist_mid_arr_1 = np.random.randint(51, 125, size=1000)
  dist_far_arr_1 = np.random.randint(126, 270, size=1000)

  dam_close_arr_1 = np.random.randint(13000000, 26000000, size=1000)
  dam_mid_arr_1 = np.random.randint(4300000, 6400000, size=1000)
  dam_far_arr_1 = np.random.randint(1000000, 3400000, size=1000)

  a = np.concatenate((dist_close_arr_1, dist_mid_arr_1, dist_far_arr_1)).reshape(3000,1)
  b = np.concatenate((dam_close_arr_1, dam_mid_arr_1, dam_far_arr_1)).reshape(3000,1)
  final1 = np.concatenate((cat_arr_1, a, b), axis = 1)

  dataset = np.concatenate((final5, final4, final3, final2, final1), axis = 0)

  scaler_X = MinMaxScaler()
  scaled_X = scaler_X.fit_transform(dataset[:,:2])
  scaler_Y = MinMaxScaler()
  scaled_Y = scaler_Y.fit_transform(dataset[:,2:])

  #build model
  model = Sequential()
  model.add(Dense(64,input_dim= 2, kernel_initializer='normal', activation='relu'))
  model.add(Dense(32, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_squared_error', optimizer='adam')

  model.fit(scaled_X, scaled_Y, batch_size=128, epochs=100, verbose =1)
  print(scaled_X.shape)
  pred = model.predict(scaled_X[0:1])
  print(pred, scaled_X[0])

  model.save_weights(path.abspath('heatweights.h5'))

