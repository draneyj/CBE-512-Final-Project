import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.losses as losses

R = 8.314
# Ideal Gas Law
def ig(v,T,Pc,Tc,w):
    return (R * T / v) / Pc

# Van der Waals (VDW) 1873: critical properties
def vdw(v,T,Pc,Tc,w):
    return 8 * Tr / (3 * vr - 1) - 3 / vr ** 2

# Soave modification of Redlich-Kwong (SRK) 1972: critical properties + accentric factor
def srk(v,T,Pc,Tc,w):
    Tr = T/Tc
    alpha = (1 + (0.480 + 1.574 * w - 0.176 * w ** 2) * (1 - np.sqrt(Tr)))
    a = 1 / (9 * (2 ** (1/3) - 1)) * R ** 2 * Tc ** 2 / Pc
    b = (2 ** (1/3) - 1) / 3 * R * Tc / Pc
    return (R * T / (v - b) - a * alpha / (v*(v+b)))/Pc

# Peng-Robinson (PR) 1976: critical properties + accentric factor
def pr(v,T,Pc,Tc,w):
    Tr = T/Tc
    a = 0.457235 * R ** 2 * Tc ** 2 / Pc
    b = 0.077796 * R * Tc / Pc 
    alpha = ( 1 + (0.37464 + 1.54226 * w - 0.26992 * w ** 2)*(1-np.sqrt(Tr)))
    return (R * T / (v - b) - a * alpha / (v*(v+b)+b*(v-b)))/Pc

# Elliott, Suresh, Donohue (ESD) 1990: critical properties + accentric factor
def esd(v,T,Pc,Tc,w):
    return 0

# load model Prs
modelmses = np.load("model_mses.arr")
modelmaes = np.load("model_maes.arr")

# load test set
bigData = pd.read_pickle("bigData_unskaled.pkl")
testdata = bigData[bigData['Test']]

# predict Prs
def getmetrics(model, data):
    Y_test = data['Pr']
    Y_hat = np.zeros(len(Y_test))
    for i,row in data.iterrows():
        Y_hat[i] = model(row['V'],row['T'],row['Pc'],row['Tc'],row['Af'])
    return [losses.MAE(Y_test, Y_hat), losses.MSE(Y_test, Y_hat)]

numcats = 4

def getmetrics_allcats(model):
    maes = np.zeros(numcats)
    mses = np.zeros(numcats)
    for category in range(numcats):
        data = testdata[testdata['category']==category]
        mae, mse = getmetrics(model, data)
        maes[category] = mae
        mses[category] = mse
    return [maes, mses]
igmaes,igmses = getmetrics_allcats(ig)
vdwmaes,vdwmses = getmetrics_allcats(vdw)
srkmaes,srkmses = getmetrics_allcats(srk)
prmaes,prmses = getmetrics_allcats(pr)

# bar plot for maes
X = np.arange(numcats)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, modelmaes, width = 1.0/6.0, label = "Neural Net")
ax.bar(X + 1.0/6.0, igmaes, width = 1.0/6.0, label="Ideal Gas Law")
ax.bar(X + 2.0/6.0, vdwmaes, width = 1.0/6.0, label="Van der Waal's")
ax.bar(X + 3.0/6.0, srkmaes, width = 1.0/6.0, label="Soave-Redlich-Kwong")
ax.bar(X + 4.0/6.0, prmaes, width = 1.0/6.0, label="Peng-Robinson")
plt.xlabel("category")
plt.ylabel("mae")
fig.show()
# bar plot for mses