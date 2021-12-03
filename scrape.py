from requests import get
from bs4 import BeautifulSoup
from time import sleep
from random import random
import pandas as pd
import numpy as np


def str2Range(rangeString):
    stringWithTo = rangeString[55 : rangeString.find("MPa", ) - 1]
    lower = float(stringWithTo[: rangeString.find("t")-1])
    upper = float(stringWithTo[rangeString.find(" ") :])
    return [lower,upper]

def getRange(ID):
    URL1 = "https://webbook.nist.gov/cgi/fluid.cgi?ID="
    URL2 = "&TUnit=K&PUnit=MPa&DUnit=mol%2Fl&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm&Type=IsoBar&RefState=NBP&Action=Page"
    page = requests.get(URL1+ID+URL2)
    time.sleep(random.random() + 0.25)
    soup = BeautifulSoup(page.content, "html.parser")
    rangeString = soup.find_all("li")[15].get_text()
    return str2Range(rangeString)

def makeURL(P, ID):
    URL1  = "https://webbook.nist.gov/cgi/fluid.cgi?P="
    URL2 = "&TLow=&THigh=&TInc=&Digits=5&ID="
    URL3 = "&Action=Load&Type=IsoBar&TUnit=K&PUnit=MPa&DUnit=mol%2Fl&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm&RefState=NBP"
    return URL1+str(P)+URL2+ID+URL3

def getPropTable(ID, Prange):
    P = np.round(np.mean(Prange), decimals=2)
    page = requests.get(makeURL(P, ID))
    time.sleep(random.random() + 0.25)
    tables = pd.read_html(page.text)
    proptable = tables[2]
    return proptable

def propstr2float(string):
    if not(" " in string):
        return float(string)
    return float(string[:string.find(" ")])

def table2Props(pt):
    Tc = propstr2float(pt.iloc[0,1])
    Pc = propstr2float(pt.iloc[1,1])
    Dc = propstr2float(pt.iloc[2,1])
    Af = propstr2float(pt.iloc[3,1])
    nbp = propstr2float(pt.iloc[4,1])
    dipole = propstr2float(pt.iloc[5,1])
    return [Tc, Pc, Dc, Af, nbp, dipole]

def getMW(ID):
    page = requests.get("https://webbook.nist.gov/cgi/cbook.cgi?ID="+ID)
    time.sleep(random.random() + 0.25)
    soup = BeautifulSoup(page.content, "html.parser")
    lis = soup.find_all("li")
    return float(lis[16].text[lis[16].text.rfind(" ")+1:])

if __name__ == "__main__":
    mols_long = pd.read_csv("IDs.csv")
    mols = mols_long
    numMols = mols.shape[0]

    print("fetching pressure ranges")
    mols["Prange"] = [getRange(ID) for ID in mols["ID"]]

    print("fetching properties")
    props = [table2Props(getPropTable(ID, Prange)) for ID, Prange in zip(mols["ID"], mols["Prange"])]

    print("fetching molecular weights")
    mols["MW"] = [getMW(ID) for ID in mols["ID"]]
    mols["Tc"] = [prop[0] for prop in props]
    mols["Pc"] = [prop[1] for prop in props]
    mols["Dc"] = [prop[2] for prop in props]
    mols["Af"] = [prop[3] for prop in props]
    mols["nbp"] = [prop[4] for prop in props]
    mols["dipole"] = [prop[5] for prop in props]

    pd.to_pickle(mols, "mols.pkl", protocol=4)