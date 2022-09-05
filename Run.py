#%% Import Data ------------------------------------------------------------
from statistics import quantiles
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

## FactorYMC - Event gender 
print("FactorYMC - Event gender :")
print(FunFactorYMC(VariableToConvert = 'Event gender', TargetName = 'Year',Data = data, FrequencyNumber = 100, Fun = percentile(90), Suffix='_Median_YMC' ))

# %% NumericYMC Class-------------------------------------------------------------
from YMC import NumericYMC
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
NumericYMC1 = NumericYMC(VariableToConvert = data['Year'],TargetVariable = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.mean,Name = "Mean",VariableName = 'Year')
NumericYMC2 = NumericYMC(VariableToConvert = data['Year'],TargetVariable = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median",VariableName = 'Year')
NumericYMC3 = NumericYMC(VariableToConvert = data['Year'],TargetVariable = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = len,Name = "Lenthg",VariableName = 'Year')
NumericYMC4 = NumericYMC(VariableToConvert = data['Year'],TargetVariable = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = percentile(90),Name = "Percentile",VariableName = 'Year')

NumericYMC1.fit()
NumericYMC2.fit()
NumericYMC3.fit()
NumericYMC4.fit()

# %% FactorYMC Class-------------------------------------------------------------
from YMC import FacotrYMC
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')

## FactorYMC - Sport
VariableToConvert = data['Sport']
TargetVariable = data['Year']
FacotrYMC1 = FacotrYMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, FrequencyNumber = 100, Fun = np.median, Name = "Median", VariableName = 'Sport')
FacotrYMC1.fit()

## FactorYMC - Event gender 
VariableToConvert = data['Event gender']
TargetVariable = data['Year']
FacotrYMC2 = FacotrYMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, FrequencyNumber = 100, Fun = np.median, Name = "Median", VariableName = 'Sport')
FacotrYMC2.fit()

## FactorYMC - Event gender 
VariableToConvert = data['Event gender']
TargetVariable = data['Year']
FacotrYMC3 = FacotrYMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, FrequencyNumber = 100, Fun = percentile(90), Name = "Percentile", VariableName = 'Sport')
FacotrYMC3.fit()

# %%
