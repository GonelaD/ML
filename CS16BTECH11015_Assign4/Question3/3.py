import pandas as pd
import io
import numpy as np
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab as pl

train_data = np.genfromtxt('dataset1.txt')

X=train_data

# for i in range(len(train_data)):
# 	X.append(train_data[i])

clf= KMeans()
clf.fit(X)
print clf.n_clusters
y = clf.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

centers = clf.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('8 means Clustering')
plt.savefig('8 means Clustering')
plt.show()

train_data = np.genfromtxt('dataset2.txt')
X=train_data

clf= KMeans(n_clusters=4)
clf.fit(X)
print clf.n_clusters
y = clf.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

centers = clf.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.savefig('4 means Clustering')
plt.show()