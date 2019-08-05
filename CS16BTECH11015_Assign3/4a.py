import numpy as np
import csv
import pandas as pd
import itertools
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import utils
import  matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import os

test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")

train_data = train_data.dropna(how = 'any')


train_data['longitude_dif'] = (train_data.dropoff_longitude - train_data.pickup_longitude).abs() 
train_data['latitude_dif'] =  (train_data.dropoff_latitude - train_data.pickup_latitude).abs()

test_data['longitude_dif'] = (test_data.dropoff_longitude - test_data.pickup_longitude).abs() 
test_data['latitude_dif'] =  (test_data.dropoff_latitude - test_data.pickup_latitude).abs()
train_X =  np.column_stack((train_data.longitude_dif,train_data.latitude_dif))

train_y  = np.array(train_data['fare_amount'])

test_X = np.column_stack((test_data.longitude_dif,test_data.latitude_dif))


lr = GradientBoostingRegressor()
lr.fit(train_X,train_y)
predictions = lr.predict(test_X)
submission = pd.DataFrame( {'key': test_data.key, 'fare_amount': predictions},columns = ['key', 'fare_amount'])
submission.to_csv('CS16BTECH11015_1.csv', index = False)
