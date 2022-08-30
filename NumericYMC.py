from tkinter import Variable
import pandas as pd
import numpy as np
# import statsmodels.formula.api as smf
# #from sklearn import metrics
# import os
# import pickle
# from datetime import datetime
# #from sklearn.ensemble import GradientBoostingClassifier
# #from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
 
# import lightgbm as LightGBM
# from sklearn.linear_model import LogisticRegression
# import lifelines
# from lifelines.utils import k_fold_cross_validation

# #from sklearn.linear_model import QuantileRegressor
# import shap
# import sqlalchemy

## ------------------------------------------------------------------------------------------------
## ----------------------------------------- Functions --------------------------------------------
## ------------------------------------------------------------------------------------------------
## Ranks Dictionary -------------------------------------------------------
def Ranks_Dictionary(temp_data, ranks_num):
    quantile_seq = np.linspace(1 / ranks_num, 1, ranks_num)
    overall_quantile = list(map(lambda x: round(np.quantile(temp_data, x), 6), quantile_seq))
    overall_quantile = pd.concat([pd.DataFrame(quantile_seq), pd.DataFrame(overall_quantile)], axis=1)
    overall_quantile.columns = ['quantile', 'value']
    overall_quantile['lag_value'] = overall_quantile['value'].shift(1)
    overall_quantile.loc[:, 'lag_value'] = overall_quantile['lag_value'].fillna(float('-inf'))
    overall_quantile.loc[:, 'value'][len(overall_quantile['value']) - 1] = float('inf')
    overall_quantile['Rank'] = list(range(1, len(overall_quantile['value']) + 1))
    overall_quantile = overall_quantile.loc[overall_quantile['value']!= overall_quantile['lag_value'], :]
    return overall_quantile


## jitter ---------------------------------------------------------------------
def RJitter(x,factor):
    z = max(x)-min(x)
    amount = factor * (z/50)
    x = x + np.random.uniform(-amount, amount, len(x))
    return(x)

## Multiple Numeric YMC -------------------------------------------------------
def MultipleNumericYMC(Variable, PercentHandicape, Suspicious, TimeDurationDays, NumberOfGroups):

    # Create dictionary for Variable
    #Dictionary = Ranks_Dictionary(Variable + np.random.uniform(0, 0.00001, len(Variable)), ranks_num=NumberOfGroups)
    Dictionary = Ranks_Dictionary(RJitter(Variable,0.00001), ranks_num=NumberOfGroups)
    Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                    Dictionary['value'],
                                                    closed='left')
    # Convert Each value in variable to rank
    Variable = pd.DataFrame({'Variable': Variable, 
                             'Suspicious': Suspicious,
                             'PercentHandicape': PercentHandicape,
                             'TimeDurationDays': TimeDurationDays})
    V = Variable['Variable']
    Variable['rank'] = Dictionary.loc[V]['rank'].reset_index(drop=True)

    # Create The aggregation for each rank
    def aggregations(x):
        Mean_YMC = np.mean(x)
        NumberOfObservation = len(x)
        DataReturned = pd.DataFrame({'Mean_YMC_PercentHandicape': [Mean_YMC],
                                     'NumberOfObservation': [NumberOfObservation]})

        return DataReturned

    # Aggregation Table
    AggregationTable = Variable.groupby('rank')['PercentHandicape'].apply(aggregations).reset_index()
    AggregationTable = AggregationTable.drop(columns=['level_1'])

    AggregationTable_YSuspicious = Variable.groupby('rank')['Suspicious'].mean().reset_index('rank')
    AggregationTable_YSuspicious.columns = ['rank' ,'Mean_YMC_YSuspicious']
    
    AggregationTable_TimeDurationDays = Variable.groupby('rank')['TimeDurationDays'].mean().reset_index()
    AggregationTable_TimeDurationDays.columns = ['rank' ,'Mean_YMC_TimeDurationDays']

    # Join Aggregation Tables
    AggregationTable = AggregationTable.join(AggregationTable_YSuspicious.set_index('rank'), how='left', on='rank')
    AggregationTable = AggregationTable.join(AggregationTable_TimeDurationDays.set_index('rank'),how='left', on='rank')

    # Merge to The Dictionary
    Dictionary = Dictionary.merge(AggregationTable, how='left', on=['rank'])

    Dictionary.loc[:, 'Mean_YMC_PercentHandicape'] = Dictionary['Mean_YMC_PercentHandicape'].fillna(np.mean(PercentHandicape.dropna()))
    Dictionary.loc[:, 'NumberOfObservation'] = Dictionary['NumberOfObservation'].fillna(0)
    Dictionary.loc[:, 'Mean_YMC_YSuspicious'] = Dictionary['Mean_YMC_YSuspicious'].fillna(np.mean(Suspicious.dropna()))
    Dictionary.loc[:, 'Mean_YMC_TimeDurationDays'] = Dictionary['Mean_YMC_TimeDurationDays'].fillna(np.mean(TimeDurationDays.dropna()))

    return Dictionary



##  Numeric YMC -------------------------------------------------------
    #Variable = data['Year']
    #Target = np.where(data['Year']>=data['Year'].mean(),1,0)
    #NumberOfGroups = 10
def NumericYMC(Variable, Target, NumberOfGroups):
    # Create dictionary for Variable
    #Dictionary = Ranks_Dictionary(Variable + np.random.uniform(0, 0.00001, len(Variable)), ranks_num=NumberOfGroups)
    Dictionary = Ranks_Dictionary(RJitter(Variable,0.00001), ranks_num=NumberOfGroups)
    Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                    Dictionary['value'],
                                                    closed='left')
    # Convert Each value in variable to rank
    Variable = pd.DataFrame({'Variable': Variable, 
                             'Target': Target})
    IntervalLocation = Variable['Variable']
    Variable['Rank'] = Dictionary.loc[IntervalLocation]['Rank'].reset_index(drop=True)
    del IntervalLocation


    # Aggregation Table
    AggregationTable = pd.DataFrame()
    AggregationTable['Rank'] = Variable['Rank'].unique()
    AggregationTable = AggregationTable.merge(Variable.groupby('Rank')['Variable'].mean().reset_index().set_axis(['Rank', 'Mean'], axis=1), how='left', on=['Rank'])
    AggregationTable = AggregationTable.merge(Variable.groupby('Rank')['Variable'].median().reset_index().set_axis(['Rank', 'Median'], axis=1), how='left', on=['Rank'])
    AggregationTable = AggregationTable.merge(Variable.groupby('Rank')['Variable'].quantile(q=0.95).reset_index().set_axis(['Rank', 'Quantile'], axis=1), how='left', on=['Rank'])
    AggregationTable = AggregationTable.merge(Variable.groupby('Rank')['Variable'].apply(lambda x: len(x)).reset_index().set_axis(['Rank', 'NumberOfObserbation'], axis=1), how='left', on=['Rank'])

    # Merge to The Dictionary
    Dictionary = Dictionary.merge(AggregationTable, how='left', on=['Rank'])

    Dictionary.loc[:, 'Mean'] = Dictionary['Mean'].fillna(np.mean(Variable['Variable'].dropna()))
    Dictionary.loc[:, 'Median'] = Dictionary['Median'].fillna(np.median(Variable['Variable'].dropna()))
    Dictionary.loc[:, 'Quantile'] = Dictionary['Quantile'].fillna(np.quantile(Variable['Variable'].dropna(),q=0.95))
    Dictionary.loc[:, 'NumberOfObserbation'] = Dictionary['NumberOfObserbation'].fillna(0)

    return Dictionary

##  Factor YMC -------------------------------------------------------
    #Variable = data['Year']
    #Target = np.where(data['Year']>=data['Year'].mean(),1,0)
    #NumberOfGroups = 10
def FactorYMC(Variable, Target):
    for VariableToConvert in FactorVariables:
    VariableToConvert="Not Sport"
    Target
    # Creating variable to transform it to YMC ------------------------
    Variable = DataModel.loc[:, ["ClaimNo_Descision","percenthandicape", "y_suspicious","time_duration_days",VariableToConvert]]
    Variable.columns = ["ClaimNo_Descision","percenthandicape", "y_suspicious","time_duration_days", "VariableToConvert"]
    Variable.loc[:, 'VariableToConvert'] = Variable['VariableToConvert'].astype(str)
    Variable.loc[:, 'VariableToConvert'] = Variable['VariableToConvert'].fillna('NULL')        

    # Group all the Not Frequent Factor to one factor group -----------
    NotFrequentFactorGroup = pd.DataFrame(Variable.groupby('VariableToConvert')['percenthandicape'].apply(lambda x: 'Rare' if len(x) <= 15 else 'Frequent')).reset_index()
    NotFrequentFactorGroup.columns = ["VariableName", "SmallGroupOrNot"]
    FrequentFactors = NotFrequentFactorGroup.loc[NotFrequentFactorGroup.SmallGroupOrNot == 'Frequent'].VariableName
    Variable.loc[:, 'VariableToConvert'] = np.where(Variable['VariableToConvert'].isin(FrequentFactors), Variable['VariableToConvert'], 'Not Frequent Factor')

    # Creating Dictionary
    Dictionary_Variable_YMC = Variable.groupby('VariableToConvert')[["percenthandicape", "y_suspicious","time_duration_days"]].apply(np.mean).reset_index()
    Dictionary_Variable_YMC.columns = ["Variable","YMC_PercentHandicape_Mean_Variable","YMC_Suspicious_Mean_Variable","time_duration_days"]
    Dictionary_Variable_YMC = Dictionary_Variable_YMC.sort_values(by='YMC_PercentHandicape_Mean_Variable', ascending=False)

    Dictionary = pd.DataFrame(data = {"VariableToConvert": DataModel[VariableToConvert].unique()})
    Dictionary['VariableToConvert'] = Dictionary['VariableToConvert'].astype(str)
    Dictionary['VariableToConvert'] = Dictionary['VariableToConvert'].fillna('NULL') 
    Dictionary['Variable'] = np.where(Dictionary['VariableToConvert'].isin(FrequentFactors), Dictionary['VariableToConvert'], 'Not Frequent Factor')
    Dictionary = Dictionary.join(Dictionary_Variable_YMC.set_index('Variable'), how='left', on='Variable')
    Dictionary = Dictionary.drop(columns = 'Variable')
    Dictionary.columns = Dictionary_Variable_YMC.columns
