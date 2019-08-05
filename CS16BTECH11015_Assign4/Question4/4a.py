import pandas as pd
import csv
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pylab as pl
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.datasets import make_swiss_roll

col = ['sepal_length','sepal_width','petal_length','petal_width','species']

train_data = pd.read_csv('iris.data', sep=",", names=col)
species = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
train_data.species = [species[item] for item in train_data.species]

y= train_data.pop('species')
X=train_data
X= StandardScaler().fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.show()

