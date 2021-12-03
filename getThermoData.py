from scrape import makeURL
from random import random
from time import sleep
from requests import get
import pandas as pd

def getStateDF(ID,P):
    page = requests.get(makeURL(P, ID))
    sleep(random() + 0.25)
    tables = pd.read_html(page.text)
    Ptable = tables[0]
    return Ptable

if __name__ == "__main__":
    pass
# generate uniform distribution of 200 Ps from a compound
# make dataframe with info from mols in first columns and each row of PVT table
# repeat for all IDs
# save to data.pkl