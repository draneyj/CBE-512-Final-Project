from scrape import makeURL
from random import random
from time import sleep
from requests import get
import numpy as np
import pandas as pd

def getStateDf(ID,P):
    page = get(makeURL(P, ID))
    sleep(random() + 0.25)
    tables = pd.read_html(page.text)
    Ptable = tables[0]
    return Ptable
def getManyStates(molrow,N=300):
    ID = molrow['ID']
    Prange = molrow['Prange']
    Ps = np.linspace(Prange[0], Prange[1], N)
    smallDfs = []
    for P in Ps:
        smallDfs.append(getStateDf(ID,P))
    return pd.concat(smallDfs)
def addMolRows(molrow,singleDf):
    for column in molrow.keys():
        singleDf[column] = [molrow[column]] * len(singleDf)
    return singleDf

if __name__ == "__main__":
    mols = pd.read_pickle('mols.pkl')
    smallDfs = []
    for i, mol in mols.iterrows():
        print("getting", mol['mol'], "...")
        smallDf = getManyStates(mol, 300)
        smallDfs.append(addMolRows(mol, smallDf))
        bigDf = pd.concat(smallDfs) # I know this is slow I'm just lazy
        pd.to_pickle(bigDf,'data.pkl') # I want to save every time in case something happens...