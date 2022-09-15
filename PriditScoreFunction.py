def Pridit(Data,FactorVariables = None,NumericVariables = None):
 
    ## Fill the FactorVariables and NumericVariables list ---------------------
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
        
        Variable = Data[[VariableToConvert]].copy()
        Variable.columns = ["VariableToConvert"]
        Variable.loc[:,'VariableToConvert'] = Variable['VariableToConvert'].astype(str).fillna('NULL')
   
        # Frequency table
        if (len(Variable['VariableToConvert'].unique()) < 2):
            continue
           
        FrequencyTable = pd.DataFrame(Variable['VariableToConvert'].value_counts(normalize = True)).reset_index()
        FrequencyTable.columns = [VariableToConvert,'Frequency']
        FrequencyTable = FrequencyTable.sort_values('Frequency',ascending = True)
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
        Variable = Data[[VariableToConvert]].copy().astype(float)
        Variable = Variable.fillna(np.mean(Variable))
        Variable.columns = ["VariableToConvert"]
       
        #Rank the numeric variable
        Dictionary = Ranks_Dictionary(RJitter(Variable['VariableToConvert'],0.00001), ranks_num=10)
        Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                        Dictionary['value'],
                                                        closed='left')
       
        # Convert Each value in variable to rank
        Variable['Rank'] = Dictionary.loc[Variable['VariableToConvert']]['rank'].reset_index(drop=True).astype(str).fillna('NULL')
   
        # Frequency table
        if (len(Variable['VariableToConvert'].unique()) < 2):
            continue
           
        FrequencyTable = pd.DataFrame(Variable['Rank'].value_counts(normalize = True)).reset_index()
        FrequencyTable.columns = ['Rank','Frequency']
        FrequencyTable = FrequencyTable.sort_values('Frequency',ascending = True)
        FrequencyTable['CumSum'] = FrequencyTable['Frequency'].cumsum()
        FrequencyTable['F'] = FrequencyTable['CumSum'] - FrequencyTable['Frequency'] - (1 - FrequencyTable['CumSum'])
        FrequencyTable = FrequencyTable[['Rank','F']]
        FrequencyTable.columns = ['Rank','FTransformation_' + VariableToConvert]
        FrequencyTable['Rank'] = FrequencyTable['Rank'].astype(str)
       
        #Merge to The Table
        Variable = Variable.join(FrequencyTable.set_index('Rank'),on='Rank',how='left')
        F['FTransformation_' + VariableToConvert] = Variable['FTransformation_' + VariableToConvert]
   
        
    ## Calculating the Eigenvector --------------------------------------------
    F_mat = F.to_numpy()
    F_t_F = np.matmul(F_mat.T,F_mat)
    eigenvalues, eigenvectors = np.linalg.eigh(F_t_F)
    PriditScore = F_mat.dot(eigenvectors[:,np.argmax(eigenvalues)])
   
    return PriditScore
 