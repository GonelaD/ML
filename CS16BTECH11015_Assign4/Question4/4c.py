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

data,y = make_swiss_roll(n_samples=200)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

plt.scatter(data_pca[:,0],data_pca[:,1],c=y)
plt.savefig('PCA_DATASET1')
plt.show()

X_tsne = TSNE(n_components=2).fit_transform(data)
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y)
plt.savefig('TSNE_DATASET1')
plt.show()

