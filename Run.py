#%% Import Data
import pandas as pd
import numpy as np
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
data.head(10)

# %% Working Directory
import os
os.chdir("/Users/dhhazanov/Desktop/Scripts/Utilities/")
os.getcwd()

# %% Import for Utilities Functions
from NumericYMC import RJitter,Ranks_Dictionary,MultipleNumericYMC,NumericYMC
RJitter(data['Year'],factor=0.1)
Ranks_Dictionary(data['Year'],ranks_num=10)
NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10)
# %%
