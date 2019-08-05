import pandas as pd
import numpy as np
import io
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb

#loading data using csv
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

col = ["distance","taxi_type","months_of_activity","customer_score","customer_score_confidence","drop_location_type","ratings_given_by_cust","num_of_cancelled_trips","anon_var_1","anon_var_2","anon_var_3","sex"]

change = ["taxi_type","customer_score_confidence","drop_location_type","sex"]

X = train_data[col]
y = train_data["pricing_category"]

#Encoding categorical values from strings to integers
for columns in change:
	le = preprocessing.LabelEncoder()
	X[columns]=le.fit_transform(X[columns])

#standerdising the data
X = preprocessing.scale(X)	

#using xgbClassifier to fit the data with the written parameters
xgb_model = xgb.XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'multi:softmax',num_class=3,nthread=4,scale_pos_weight=1,seed=27)
eval_set = [(X, y)]
xgb_model.fit(X, y, early_stopping_rounds=3, eval_set=eval_set, verbose=True)

#pre-processing test_Data
test_X=test_data[col]
for columns in change:
	le = preprocessing.LabelEncoder()
	test_X[columns]=le.fit_transform(test_X[columns])

test_X = preprocessing.scale(test_X)	

#predicting values for test_data
y_predictions = xgb_model.predict(test_X)

#converting labels to float
y_predictions =[float(item) for item in y_predictions]

#writing to csv file
submission = pd.DataFrame( {'id': test_data.id, 'pricing_category': y_predictions},columns = ['id', 'pricing_category'])
submission.to_csv('CS16BTECH11033_CS16BTECH11015.csv', index = False)
