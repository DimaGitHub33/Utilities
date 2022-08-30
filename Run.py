#%% Import Data
import pandas as pd
import numpy as np
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
data.head(10)

# %% Working Directory
import os
os.chdir("/Users/dhhazanov/Documents/GitHub/Utilities/")
os.getcwd()

# %% Import for Utilities Functions
from NumericYMC import RJitter,Ranks_Dictionary,FourBasicNumericYMC,NumericYMC,FactorYMC,percentile
print("RJitter:")
print(RJitter(data['Year'],factor=0.1))
print("\n")
print("\n")
print("Ranks_Dictionary:")
print(Ranks_Dictionary(data['Year'],ranks_num=10))
print("\n")
print("\n")
print("NumericYMC:")
print(FourBasicNumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10))
print("\n")
print("\n")
print(NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.mean,Name = "Mean"))
print(NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median"))
print(NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = percentile(90),Name = "Percentile"))

print("\n")
print("\n")
print("FactorYMC:")
print(FactorYMC(VariableToConvert = 'Sport', TargetName = 'Year',Data = data, FrequencyNumber = 100, Fun = np.median, Suffix='_Median_YMC' ))
print("\n")
print("\n")

# %%
