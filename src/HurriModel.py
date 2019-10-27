import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from geopy.distance import great_circle as vc
import math as Math

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.externals import joblib 
import math, time
import keras
from os import path

from keras.models import model_from_json


def get_distance(prev, curr):
    # prev and curr are coordinate pairs of (lat, long)
    return vc(prev, curr).miles

def get_direction(prev, curr):
    dLon = curr[1] - prev[1];  
    temp = float(curr[0]) # p[0] is a str?
    y_x = Math.sin(dLon) * Math.cos(temp)
    
    x_x = Math.cos(curr[1]) * Math.sin(temp) - Math.sin(curr[1]) * Math.cos(temp) * Math.cos(dLon);
    brng = Math.degrees(Math.atan2(y_x, x_x)) 
    if (brng < 0):
        brng+= 360
    
    return brng


def coord2grid(lat, lon):
    lat_min = 7.2
    long_min = -109.3
    lat_interval = round(66 - 7.2)
    long_interval = round(13.5 + 109.3)

    gridID = np.floor(lat - 7.200)* long_interval  + np.floor(lon + 109.3)
    gridID = round(gridID)

    return gridID

def process_data(data):
    processed_data = []
    for i in range(len(data)):
        updated_pt = []
        updated_pt.append(data[i][2]) #wind speed
        updated_pt.append(data[i][3]) #pressure
        if (i==0):
            updated_pt.append(0)
        else:
            updated_pt.append(get_distance([data[i-1][1],data[i-1][2]], [data[i][1],data[i][2]])) #distance
        if (i==0):
            updated_pt.append(0)
        else:
            updated_pt.append(get_direction([data[i-1][1],data[i-1][2]], [data[i][1],data[i][2]])) #angle
        updated_pt.append(coord2grid(data[1], data[2])) #gridID
    scaler = joblib.load(path.abspath("andrewscaler.save"))
    processed_data = np.array(processed_data)
    scaled_data = scaler.fit(processed_data)

def grid2coord(grid):
    y = np.floor(grid/round(13.5 + 109.3)) #y_coord
    x = np.floor(grid-y*round(13.5 + 109.3)) #x_coord
    #x and y represent the bottom left corner of the grid
    return (x+0.5-109.3, y+0.5+7.2)

def build_model(layers):
    model = Sequential()

    for x in range(0,3):
        model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
        model.add(Dropout(0.1))

    model.add(LSTM(layers[2], return_sequences=False)) 
    model.add(Dropout(0.1))

    model.add(Dense(output_dim=layers[2]))
    model.add(Activation("tanh"))

    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    return model

def create_trajectory(initial_pts):
    json_file = open(path.abspath('hurrimodel.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    loaded_model.load_weights(path.abspath("hurrimodel.h5"))
    #model_input = process_data(initial_pts)
    model = loaded_model
    pred = model.predict(initial_pts)
    return pred

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def load_hurricane(stock, seq_len):#load a single hurricane
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # Because index starts at 0
    result = []

    for index in range(len(data) - sequence_length):
        seq = data[index: index + sequence_length]
        result.append(seq)
                
    result = np.array(result)
    result = result[:,:-1]
    return np.reshape(result, (result.shape[0], result.shape[1], amount_of_features))

def prep_hurricane(hurr, name):
    data_pad = [hurr[ hurr.loc[:, 'unique-key'] == name].loc[:, ['WindSpeed', 'Pressure', 'distance', 'direction', 'gridID']]]
    hurr.drop(['Month', 'Day', 'Hour', 'Lat', 'Long', 'unique-key'], axis = 1, inplace = True)
    hurr = hurr[hurr['distance'] > 0]

    hurr['distance'] = np.log(hurr['distance'])

    hurr = hurr[hurr['direction'] > 0]
    hurr['direction'] = np.log(hurr['direction'])
    
#     print (hurr)
#     print ()
#     print (data_pad)
    padded_data = keras.preprocessing.sequence.pad_sequences(data_pad, maxlen=60, dtype='int32', padding='post', truncating='pre', value=0.0)
    scaler = joblib.load(path.abspath("dollyscaler.save"))

#     print(padded_data)
    return pd.DataFrame(scaler.fit_transform(padded_data[0]), columns=['WindSpeed', 'Pressure', 'Distance', 'Direction', 'gridID'])
    

def some_test():
    data = pd.read_csv(path.abspath('checkpoint-dataframe.csv'), index_col=0, header=0)
    name = 'DOLLY-2002-4'
    hurricane = prep_hurricane(data[data['unique-key'] == name], name, ) # This is good
    hurr_data = load_hurricane(hurricane, 5)
    hurricane_temp = hurricane['gridID']

    pred = create_trajectory(hurr_data)
    #hurricane_temp = hurricane_temp

    gridScaler = MinMaxScaler(feature_range=(0, 1))
    gridScaler.fit_transform(data['gridID'].as_matrix().reshape(-1,1))
    scaler = joblib.load(path.abspath("dollyscaler.save"))
    invert_grid = gridScaler.inverse_transform(pred)
    lon, lat = grid2coord(invert_grid)



# def main():
#     some_test()

# main()
    