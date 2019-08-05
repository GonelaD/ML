import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
import math 
import csv
import random
from sklearn.linear_model import Ridge
from statistics import mean,stdev


with open("linregdata") as f:
	data = [tuple(line) for line in csv.reader(f, delimiter=",")]

row = ['gender','A2','A3','A4','A5','A6','A7','A8','A9']

data1 = []
data1.append(row)
for i in range(len(data)):
	data1.append(data[i])

with open("linregdata_duplicate", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(data1)

data = pd.read_csv("linregdata_duplicate")

# ------------------------3b----------------------#

Gender = {'M': 2,'F': 1,'I':0} 
data.gender = [Gender[item] for item in data.gender] 

mean=(data.mean(axis=0))
SD = (data.std(axis=0))
data.gender = (data.gender-mean.gender)/SD.gender
data.A2 = (data.A2-mean.A2)/SD.A2
data.A3 = (data.A3-mean.A3)/SD.A3
data.A4 = (data.A4-mean.A4)/SD.A4
data.A5 = (data.A5-mean.A5)/SD.A5
data.A6 = (data.A6-mean.A6)/SD.A6
data.A7 = (data.A7-mean.A7)/SD.A7
data.A8 = (data.A8-mean.A8)/SD.A8

# print data

#..............................................................#

def mylinridgereg(X,y,alpha):
	mat=np.identity(X.shape[1])
	weight_matrix=np.dot(np.linalg.inv(np.dot(X.T,X)+alpha*mat),np.dot(X.T,y))
	return weight_matrix

def mylinridgeregeval(X_test,weights):
	return np.dot(X_test,weights)

def meansquarederr(y_test,y_pred):
	sum_val = 0.0
	for row in range(len(y_test)) :
		sum_val = sum_val + math.pow(y_test[row]-y_pred[row],2)
	return sum_val/len(y_test)

def find_lambda(lam_list,test_mse_list) :
	for i in range(len(test_mse_list)) :
		if(test_mse_list[i]==min(test_mse_list)) :
			return i


#-------------------------3d-----------------------------#
y= data.pop('A9')
# data.pop('A2')
# data.pop('A6')
# data.pop('A7')
X =data.values
y= y.values


new = np.ones((len(X),1))

X = np.append(X,new,axis=1)


X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lam_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
weights_list=[]
error_list =  []
for i in range(len(lam_list)):
	weights = mylinridgereg(X_train,y_train,lam_list[i])
	weights_list.append(weights)
	y_pred_test = mylinridgeregeval(X_test,weights)
	y_pred_train = mylinridgeregeval(X_train,weights)
	mean_test = meansquarederr(y_test,y_pred_test)
	error_list.append(mean_test)
	mean_train = meansquarederr(y_train,y_pred_train)

lam = find_lambda(lam_list,error_list)
print lam_list[lam]
print weights_list[lam]
print error_list[lam]


data.pop('A2')
data.pop('A6')
data.pop('A7')
X =data.values

new = np.ones((len(X),1))

X = np.append(X,new,axis=1)


X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

weights = mylinridgereg(X_train,y_train,lam)
print weights

y_pred_test = mylinridgeregeval(X_test,weights)
y_pred_train = mylinridgeregeval(X_train,weights)
print meansquarederr(y_test,y_pred_test)
print meansquarederr(y_train,y_pred_train)


