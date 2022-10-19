import pandas as pd
import numpy as np

## Ranks Dictionary ----------------------------------------------------------
"""    
Ranks Dictionary
    Args:
      temp_data:
        Array of numerical data
      ranks_num:
        Number of ranks to split the temp_data

    Returns:
      pandas data frame that describe the boundaries of ranks for input data 
      Example:
        Dictionary = Ranks_Dictionary(np.random.normal(3, 2.5, size=(1, 1000)), ranks_num=10)
        print(Dictionary)
        quantile     value  lag_value  rank
             0.1 -0.210117       -inf     1
             0.2  0.748485  -0.210117     2
             0.3  1.636300   0.748485     3
             0.4  2.210183   1.636300     4
             0.5  2.832312   2.210183     5
             0.6  3.501649   2.832312     6
             0.7  4.116645   3.501649     7
             0.8  4.895207   4.116645     8
             0.9  5.843564   4.895207     9
             1.0       inf   5.843564    10

      The output table help us to split any numerical array to array of ranks.
      Each numerical value is between value and lag_value in this table. the rank in that specific row is the rank of the value.
"""
def Ranks_Dictionary(temp_data, ranks_num):
    quantile_seq = np.linspace(1 / ranks_num, 1, ranks_num)
    overall_quantile = list(map(lambda x: round(np.quantile(temp_data, x), 6), quantile_seq))
    overall_quantile = pd.concat([pd.DataFrame(quantile_seq), pd.DataFrame(overall_quantile)], axis=1)
    overall_quantile.columns = ['quantile', 'value']
    overall_quantile['lag_value'] = overall_quantile['value'].shift(1)
    overall_quantile.loc[:, 'lag_value'] = overall_quantile['lag_value'].fillna(float('-inf'))
    overall_quantile.loc[:, 'value'][len(overall_quantile['value']) - 1] = float('inf')
    overall_quantile['rank'] = list(range(1, len(overall_quantile['value']) + 1))
    overall_quantile = overall_quantile.loc[overall_quantile['value']!= overall_quantile['lag_value'], :]
    return overall_quantile

## jitter ---------------------------------------------------------------------
"""    
RJitter
    Args:
      x:
        Array of numerical data
      factor:
        rate of jitter

    Returns:
      array of numerical data with almost the same data (the mean is the same the variance slightly grows, depend on the factor)
      Example:
        InputData=np.random.normal(0, 2.5, size=(1, 3))[0]
        print(InputData)
        [-1.58334198  0.64810107  0.37624609]
        print(RJitter(x=InputData,factor=0.1))
        [-1.58345746  0.64537898  0.37463414]

      The function slightly add variance to the input data so that we can split it, if there are to many unique values.
"""
def RJitter(x,factor):
    z = max(x)-min(x)
    amount = factor * (z/50)
    x = x + np.random.uniform(-amount, amount, len(x))
    return(x)

## Pridit ----------------------------------------------------------------------
"""    
Pridit
    Args:
      Data:
        Data frame of numerical and factorial data
        Example:
                                ID   DATE_OF_BIRTH GENDER 
                          14262240      ז      1946-11-15
                          14262455      ז      1956-04-18
                          14263677      ז      1953-03-15
                          14263727      נ      1958-02-12
                          14265052      נ      1956-04-24

      FactorVariables:
        List of all the variables that their type is factorial
        Example:
        ['GENDER', 'FAMILY_STATUS']
      NumericVariables:
        List of all the variables that their type is numerical
        Example:
        ['Number_Of_Kids', 'Age']
      FactorsVariablesOrder:
        data frame of all the factor variables and their levels order
        Example:
                 Variable               Level  Order
                   GENDER                   ז      0
                   GENDER                   נ      1
            FAMILY_STATUS                   נ      0
            FAMILY_STATUS                   ר      1
            FAMILY_STATUS                   א      2
      NumericVariablesOrder
        data frame of all the numeric variables and their sign order
        Example:
                Variable  Order
                     Age      1
                  Salery      1
                  Height      0
                  weight      1

    Returns:
      Pridit Score
      Example:
        Data = pd.read_parquet('/Downloads/ppp.parquet.gzip', engine='pyarrow')
        PriditScore = Pridit(Data)
        print(PriditScore)
        [-0.63490772, -0.15769004, -0.54438071, ..., -0.60417859,-0.42238741,  9.05145987]

"""
def Pridit(Data,FactorVariables = None, NumericVariables = None, FactorsVariablesOrder = None, NumericVariablesOrder = None):
 
    ## Fill the FactorVariables and NumericVariables list ----------------------
    if FactorVariables is None:
        FactorVariables = []
        DataTypes = Data.dtypes.reset_index().rename(columns = {'index': 'Index',0:'Type'})
        for Index,row in DataTypes.iterrows(): 
            if row['Type'] in ['object','str']:
                FactorVariables.append(row['Index'])
                
    if NumericVariables is None:
        NumericVariables = []
        DataTypes = Data.dtypes.reset_index().rename(columns = {'index': 'Index',0:'Type'})
        for Index,row in DataTypes.iterrows(): 
            if row['Type'] in ['int64','float64']:
                NumericVariables.append(row['Index'])
               
    
    ## F calculation for Factor variables  ------------------------------------
    F = pd.DataFrame()
    for VariableToConvert in FactorVariables:
        #print(VariableToConvert)
        Variable = Data[[VariableToConvert]].copy()
        Variable.columns = ["VariableToConvert"]
        Variable.loc[:,'VariableToConvert'] = Variable['VariableToConvert'].astype(str).fillna('NULL')
   
        # Frequency table
        if (len(Variable['VariableToConvert'].unique()) < 2):
            continue
           
        FrequencyTable = pd.DataFrame(Variable['VariableToConvert'].value_counts(normalize = True)).reset_index()
        FrequencyTable.columns = [VariableToConvert,'Frequency']
        
        ## Order the Factors by the FactorsVariablesOrder
        if FactorsVariablesOrder is None:
            FrequencyTable = FrequencyTable.sort_values('Frequency',ascending = True)
        else:
            Order = FactorsVariablesOrder[FactorsVariablesOrder['Variable'] == VariableToConvert].set_index('Level')
            if len(Order) == 0:
                FrequencyTable = FrequencyTable.sort_values('Frequency',ascending = True)
            else:
                FrequencyTable = FrequencyTable.join(Order,on=VariableToConvert,how='left')
                FrequencyTable['Order'] = FrequencyTable['Order'].fillna(np.mean(FrequencyTable['Order']))
                FrequencyTable = FrequencyTable.sort_values('Order',ascending = True)
            
        ##Calculating the weights after ordering the Levels
        FrequencyTable['CumSum'] = FrequencyTable['Frequency'].cumsum()
        FrequencyTable['F'] = FrequencyTable['CumSum'] - FrequencyTable['Frequency'] - (1 - FrequencyTable['CumSum'])
        FrequencyTable = FrequencyTable[[VariableToConvert,'F']]
        FrequencyTable.columns = [VariableToConvert,'FTransformation_' + VariableToConvert]
       
        #Merge to The Table
        F[VariableToConvert] = Data[VariableToConvert].astype(str)
        F = F.join(FrequencyTable.set_index(VariableToConvert),on=VariableToConvert,how='left')
        F = F.drop(VariableToConvert,axis=1)
   

    ## F calculation for numeric variables ------------------------------------
    for VariableToConvert in [NV for NV in NumericVariables if NV not in FactorVariables]:
        #print(VariableToConvert)
        Variable = Data[[VariableToConvert]].copy().astype(float)
        Variable = Variable.fillna(np.mean(Variable,axis=0))
        Variable.columns = ["VariableToConvert"]
       
        #Rank the numeric variable
        Dictionary = Ranks_Dictionary(RJitter(Variable['VariableToConvert'],0.00001), ranks_num=10)
        Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                        Dictionary['value'],
                                                        closed='left')
       
        # Convert Each value in variable to rank
        Variable['Rank'] = Dictionary.loc[Variable['VariableToConvert']]['rank'].reset_index(drop=True).astype(str)
   
        # Frequency table
        if (len(Variable['VariableToConvert'].unique()) < 2):
           continue
           
        FrequencyTable = pd.DataFrame(Variable['Rank'].value_counts(normalize = True)).reset_index()
        FrequencyTable.columns = ['Rank','Frequency']
        FrequencyTable['Rank'] = FrequencyTable['Rank'].astype(float)

        ## Order the Factors by the NumericVariablesOrder
        if FactorsVariablesOrder is None:
            FrequencyTable = FrequencyTable.sort_values('Frequency',ascending = True)
        else:
            Order = NumericVariablesOrder[NumericVariablesOrder['Variable'] == VariableToConvert]
            if len(Order) == 0:
                FrequencyTable = FrequencyTable.sort_values('Frequency',ascending = True)
            else:
                if Order['Order'][0]==0:
                    FrequencyTable = FrequencyTable.sort_values('Rank',ascending = False)
                else:
                    FrequencyTable = FrequencyTable.sort_values('Rank',ascending = True)

        ##Calculating the weights after ordering the numeric levels
        FrequencyTable['CumSum'] = FrequencyTable['Frequency'].cumsum().copy()
        FrequencyTable['F'] = FrequencyTable['CumSum'] - FrequencyTable['Frequency'] - (1 - FrequencyTable['CumSum'])
        FrequencyTable = FrequencyTable[['Rank','F']]
        FrequencyTable.columns = ['Rank','FTransformation_' + VariableToConvert]
        FrequencyTable['Rank'] = FrequencyTable['Rank'].astype(int).astype(str)
       
        #Merge to The Table
        Variable = Variable.join(FrequencyTable.set_index('Rank'),on='Rank',how='left')
        F['FTransformation_' + VariableToConvert] = Variable['FTransformation_' + VariableToConvert]
   
        
    ## Calculating the Eigenvector of the maximum eigenvalues-------------------
    F_mat = F.to_numpy()
    F_t_F = np.matmul(F_mat.T,F_mat)
    eigenvalues, eigenvectors = np.linalg.eigh(F_t_F)
    PriditScore = F_mat.dot(eigenvectors[:,np.argmax(eigenvalues)])
   
    return PriditScore
