import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
import math 
import csv
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from statistics import mean,stdev
import random

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



Gender = {'M': 2,'F': 1,'I':0} 
data.gender = [Gender[item] for item in data.gender] 

def mylinridgereg(X,y,alpha):
	mat=np.identity(X.shape[1])
	weight_matrix=np.dot(np.linalg.inv(np.dot(X.T,X)+alpha*mat),np.dot(X.T,y))
	return weight_matrix

def mylinridgeregeval(X_test,weights):
	return np.dot(X_test,weights)

def means(X):
	mean = []

	for col in range(len(X[0])) :
		sum_val = 0.0
		for rows in range(len(X)) :
			sum_val = sum_val + X[rows][col]
		mean.append(sum_val/len(X))

	return mean

def SD(X):
	stddev_list = []
	avg_list = means(X_train)
	
	for col in range(len(X_train[0])) :
		sum_dev = 0.0
		for rows in range(len(X_train)) :
			sum_dev = sum_dev + math.pow(X_train[rows][col] - avg_list[col],2)
		stddev_list.append(math.sqrt(sum_dev/len(X_train)))

	return stddev_list 

def standerdise(X_train,avg_list,stddev_list) :
	for col in range(len(X_train[0])) :
		for rows in range(len(X_train)) :
			X_train[rows][col] = (X_train[rows][col] - avg_list[col])/stddev_list[col]
	return X_train

def find_lambda(lam_list,test_mse_list) :
	for i in range(len(test_mse_list)) :
		if(test_mse_list[i]==min(test_mse_list)) :
			return lam_list[i]


def meansquarederr(y_test,y_pred):
	sum_val = 0.0
	for row in range(len(y_test)) :
		sum_val = sum_val + math.pow(y_test[row]-y_pred[row],2)
	return sum_val/len(y_test)

y= data.pop('A9')
X =data.values
y= y.values

test = [0.1,0.3,0.5,0.7,0.9]
min_test = []
lam_list1 = []
for size in test :
	lam_list = []
	train_mse_list= []
	test_mse_list = []
	for i in range(25) :
		X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=size,random_state=42)
		mean_list = means(X_train)
		stdev_list = SD(X_train)
		X_train = standerdise(X_train,mean_list,stdev_list)
		X_test = standerdise(X_test,mean_list,stdev_list)

		new = np.ones((len(X_train),1))
		X_train = np.append(X_train,new,axis=1)

		new = np.ones((len(X_test),1))
		X_test = np.append(X_test,new,axis=1)
		
		lam = random.random()
		weights = mylinridgereg(X_train,y_train,lam)
		train_mse = meansquarederr(y_train,mylinridgeregeval(X_train,weights))		
		test_mse = meansquarederr(y_test,mylinridgeregeval(X_test,weights))
		lam_list.append(lam)
		train_mse_list.append(train_mse)
		test_mse_list.append(test_mse)

	lam_list1.append(find_lambda(lam_list,test_mse_list))
	min_test.append(min(test_mse_list))
	plt.plot(lam_list,train_mse_list,'r',label = 'train_mse')
	plt.plot(lam_list,test_mse_list,'b',label = 'test_mse')
	plt.show()

plt.plot(test,min_test,'g')
plt.ylabel('Minimum test mse')
plt.xlabel('Partition Value')
plt.savefig('Minimum test mse versus the partition fraction values')
plt.show()

plt.plot(test,lam_list1,'g')
plt.ylabel('lambda corresponding to min test_mse')
plt.xlabel('Partition Value')
plt.savefig('lambda vs partition')
plt.show()