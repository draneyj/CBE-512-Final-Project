import sklearn
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster

mols = pd.read_pickle("mols.pkl")
print(mols)
mols_numerical = mols.drop(['mol', 'ID','Prange'], axis=1)

scaler = preprocessing.MinMaxScaler()
scaler.fit(mols_numerical)
X = scaler.transform(mols_numerical)

def listClusters(labls):
    for labl in np.unique(labls):
        boobles = [labl == label_ for label_ in labls]
        print("===========================================")
        print(mols.loc[boobles, "mol"])

from sklearn.cluster import DBSCAN
numcats = []
eps = np.linspace(0.1,1,num=9)
for ep in eps:
    clustering = DBSCAN(eps=ep).fit(X)
    numcats.append(len(np.unique(clustering.labels_)))
    
print(numcats)
clustering = DBSCAN(eps=eps[1]).fit(X)
listClusters(clustering.labels_)

from sklearn.mixture import BayesianGaussianMixture as BGM
clustering = BGM(n_components=4).fit(X)
clustering.predict(X)
listClusters(clustering.predict(X))

mols["category"] = clustering.predict(X)
pd.to_pickle(mols, "mols.pkl")