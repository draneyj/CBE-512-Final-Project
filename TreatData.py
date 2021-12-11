import pandas as pd
from pandas.io.pickle import to_pickle
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import random
mols =  pd.read_pickle('mols.pkl')

# create test/train split
splitp = 0.2
scaler = sklearn.preprocessing.StandardScaler()
dfs = [pd.DataFrame()] * len(np.unique(mols['category']))
for i,category in enumerate(np.unique(mols['category'])):
    dfs[i] = mols[mols['category']==category].copy(deep=True)
    numRows = dfs[i].shape[0]
    numTest = int(np.floor(splitp*numRows))
    tests = np.zeros(numRows, bool)
    tests[random.sample(range(0,numRows), numTest)] = True #select the appropriate subset to be test set
    dfs[i]['Test'] = tests

    df_train = dfs[i][[not x for x in dfs[i]['Test']]].copy(deep=True)
    scaler.fit(df_train.select_dtypes(include=np.number))
    dfs[i][dfs[i].select_dtypes(include=np.number).columns] = scaler.transform(dfs[i].select_dtypes(include=np.number))
    dfs[i]['category'] = [category] * len(dfs[i]['category'])

# for col in mols.select_dtypes(include=np.number).columns.tolist():
#     f = plt.figure()
#     for df in dfs:
#         plt.hist(df[col], density=True)
#     plt.title(col)
#     f.show()

mols_scaled = pd.concat(dfs)

bigData = pd.read_pickle('bigData.pkl')
bigData.index = range(bigData.shape[0])
bigData = bigData[bigData.index%10==0]
bigData.rename(columns = {'Temperature (K)':'T', 'Pressure (MPa)':'P',
       'Volume (l/mol)':'V'}, inplace=True)
bigData.drop(columns = ['Cp (J/mol*K)', 'Sound Spd. (m/s)', 'Density (mol/l)',
       'Joule-Thomson (K/MPa)', 'Viscosity (uPa*s)', 'Therm. Cond. (W/m*K)',
       'Phase','Internal Energy (kJ/mol)', 'Enthalpy (kJ/mol)',
       'Entropy (J/mol*K)', 'Cv (J/mol*K)', 'Phase', 'ID', 'Prange'], inplace=True)
bigData['Tr'] = bigData['T']/bigData['Tc']
bigData['Pr'] = bigData['P']/bigData['Pc']
bigData = bigData[bigData['V'] != 'infinite']
bigData['Vr'] = np.multiply(bigData['V'],bigData['Dc'])
bigData.index = range(bigData.shape[0])

def getFromMol(mol,prop):
    return mols_scaled[mols_scaled['mol']==mol][prop].values[0]

bigData['Test'] = [getFromMol(mol, 'Test') for mol in bigData['mol']]
bigData.to_pickle("bigData_unscaled.pkl")

bigData.columns
for col in mols_scaled.select_dtypes(include=np.number).columns:
    print("updating ",col)
    bigData[col] = [getFromMol(mol, col) for mol in bigData['mol']]#bigData.apply(lambda row: getFromMol(row['mol'], col), axis=1)


bigData.head()
bigData.to_pickle("bigData_scaled.pkl")
