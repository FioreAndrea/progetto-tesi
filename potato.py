#%%
import constant as c
import pandas as pd
import os
from os.path import join


ant_dataframe = []

for filename in os.listdir(c.ant):
    tmp = pd.read_csv(join(c.ant, filename))
    ant_dataframe.append(tmp)

ant_dataframe = pd.concat(ant_dataframe)
