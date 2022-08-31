import pandas as pd
import numpy as np
from YMC import RJitter,Ranks_Dictionary,FourBasicNumericYMC,NumericYMC,FactorYMC,percentile

#Group Mappe ------------------------------------------------------------------
def GroupMappe(Predicted, Actual, NumberOfRank,Percentile = False):
    Data = pd.DataFrame(data={'Predicted': Predicted,
                              'Actual': Actual})
    Dictionary = Ranks_Dictionary(Data['Predicted'],NumberOfRank)
    Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                    Dictionary['value'],
                                                    closed='left')
    Data['PredictedRank'] = Dictionary.loc[Data['Predicted']]['rank'].reset_index(drop=True)
    
    if(Percentile==False):
        GroupMappe = Data.groupby('PredictedRank')[['Predicted','Actual']].apply(lambda x: np.mean(x)).reset_index()
    else:
        GroupMappe = Data.groupby('PredictedRank')[['Predicted','Actual']].apply(lambda x: np.mean(100*x)).reset_index()
    
    GroupMappe.columns = ["Rank","Mean_Predicted","Mean_Actual"]
    return(GroupMappe)

def GroupMappeAccuracy(GroupMappeTable):
    Mean_Predicted = GroupMappeTable['Mean_Predicted']
    Mean_Actual = GroupMappeTable['Mean_Actual']
    return(100-100*np.average((np.abs(Mean_Predicted-Mean_Actual)+1)/(Mean_Actual+1)))