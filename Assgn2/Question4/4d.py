import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

train_data = np.genfromtxt("train_data")
test_data = np.genfromtxt("test_data")
#print train_data
new_array_train_data = [] 
new_array_train_label = []
new_array_test_data =[]
new_array_test_label = []

#print len(train_data)
for i in range(len(train_data)):
	if train_data[i][0]==5 or train_data[i][0]==1:
		new_array_train_data.append(train_data[i])
		new_array_train_label.append(train_data[i][0])


X_train =[[0 for x in range(2)] for y in range(len(new_array_train_data))]
for i in range(len(new_array_train_data)):
	X_train[i][0] = new_array_train_data[i][1]
	X_train[i][1] = new_array_train_data[i][2]


#X_train,X_test,y_train,y_test = train_test_split(X_train,new_array_train_label,test_size=0.5,random_state=10)




for i in range(len(test_data)):
	if test_data[i][0]==5 or test_data[i][0]==1:
		new_array_test_data.append(test_data[i])
		new_array_test_label.append(test_data[i][0])


X_test =[[0 for x in range(2)] for y in range(len(new_array_test_data))]
for i in range(len(new_array_test_data)):
	X_test[i][0] = new_array_test_data[i][1]
	X_test[i][1] = new_array_test_data[i][2]

clf = SVC(kernel='rbf',C=1000000)
clf.fit(X_train,new_array_train_label)

test_accuracy = clf.score(X_test,new_array_test_label)
print "Number of support vectors:",clf.n_support_

print "Test Error:",1-test_accuracy

train_accuracy = clf.score(X_train,new_array_train_label)
print "Train Error:",1-train_accuracy