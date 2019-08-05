import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

train_data = np.genfromtxt("gisette_train.data")
train_label= np.genfromtxt("gisette_train.labels")
valid_data = np.genfromtxt("gisette_valid.data")
valid_label = np.genfromtxt("gisette_valid.labels")


#X_train,X_test,y_train,y_test = train_test_split(train_data,train_label,test_size=0.5,random_state=10)


clf = SVC(kernel='linear')
clf.fit(train_data,train_label)


#predicted_values = clf.predict(X_test)
train_accuracy = clf.score(train_data,train_label)

print "Linear Support Vectors:",clf.n_support_


print "Train error:",1-train_accuracy

test_accuracy = clf.score(valid_data,valid_label)

print "Test Error:",1-test_accuracy

clf = SVC(kernel='poly',degree= 2,coef0=1)
clf.fit(train_data,train_label)



#predicted_values = clf.predict(X_test)
train_accuracy = clf.score(train_data,train_label)


print "Train error for poly:",1-train_accuracy

test_accuracy = clf.score(valid_data,valid_label)

print "Test Error for poly:",1-test_accuracy
print "Polynomial Support Vectors:",clf.n_support_

clf = SVC(kernel='rbf',gamma=0.001)
clf.fit(train_data,train_label)


#predicted_values = clf.predict(X_test)
train_accuracy = clf.score(train_data,train_label)


print "Train error for rbf:",1-train_accuracy

test_accuracy = clf.score(valid_data,valid_label)

print "Test Error for rbf:",1-test_accuracy
print "RBF Support Vectors:",clf.n_support_


