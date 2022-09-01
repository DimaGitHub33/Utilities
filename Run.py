#%% Import Data ------------------------------------------------------------
import pandas as pd
import numpy as np
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
data.head(10)

# %% Working Directory -----------------------------------------------------
import os
os.chdir("/Users/dhhazanov/Documents/GitHub/Utilities/")
os.getcwd()

# %% Import for Utilities Functions ----------------------------------------
from YMCFunctions import RJitter,Ranks_Dictionary,FourBasicNumericYMC,FunNumericYMC,FunFactorYMC,percentile

# %% Rjitter ---------------------------------------------------------------
print("RJitter:")
print(RJitter(data['Year'],factor=0.1))

# %% Ranks Dictionary -------------------------------------------------------
print("Ranks_Dictionary:")
print(Ranks_Dictionary(data['Year'],ranks_num=10))

# %% NumericYMC -------------------------------------------------------------
## Four Basic Numeric YMC in one table
print("FourBasicNumericYMC:")
print(FourBasicNumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10))
print("\n")

## Mean
print("Mean YMC:")
print(FunNumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.mean,Name = "Mean"))
print("\n")

## Median
print("Median YMC:")
print(FunNumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median"))
print("\n")

## Percentile
print("Percentile YMC:")
print(FunNumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = percentile(90),Name = "Percentile"))
print("\n")

# %% FactorYMC -------------------------------------------------------------
## FactorYMC - Sport
print("FactorYMC - Sport:")
print(FunFactorYMC(VariableToConvert = 'Sport', TargetName = 'Year',Data = data, FrequencyNumber = 100, Fun = np.median, Suffix='_Median_YMC' ))

## FactorYMC - Event gender 
print("FactorYMC - Event gender :")
print(FunFactorYMC(VariableToConvert = 'Event gender', TargetName = 'Year',Data = data, FrequencyNumber = 100, Fun = np.median, Suffix='_Median_YMC' ))

# %% NumericYMC Class-------------------------------------------------------------
from NumericYMC import NumericYMC
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
Variable = data['Year']
Target = np.where(data['Year']>=data['Year'].mean(),1,0)
NumberOfGroups = 10
NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median")

YMC1 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.mean,Name = "Mean")
YMC2 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.median,Name = "Median")
YMC3 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = percentile(90),Name = "Percentile")

#YMC3 = NumericYMC.InstantiateFromWeb()
#print(NumericYMC.AllInstances)

print(YMC1.YMC())
print(YMC2.YMC())
print(YMC3.YMC())

# %%
