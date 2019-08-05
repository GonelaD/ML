import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
import math 
import csv
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

y= data.pop('A9')
# data.pop('A2')
# data.pop('A3')
# data.pop('A4')
X =data.values
y= y.values


new = np.ones((len(X),1))

X = np.append(X,new,axis=1)

X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.5,random_state=42)

lam = 0.7

weights = mylinridgereg(X_train,y_train,lam)
# print weights
y_pred_test = mylinridgeregeval(X_test,weights)
y_pred_train = mylinridgeregeval(X_train,weights)
print meansquarederr(y_test,y_pred_test)
print meansquarederr(y_train,y_pred_train)


a = np.linspace(2,20,100)
b = a

plt.plot(a,b,'b')
plt.scatter(y_pred_train,y_train)
plt.savefig('Predicted Vs Actual for Train data')
plt.show()

plt.plot(a,b,'r')
plt.scatter(y_pred_test,y_test)
plt.savefig('predicted Vs Actual for test Data')
plt.show()



