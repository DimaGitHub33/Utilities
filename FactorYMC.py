##  NumericYMC -------------------------------------------------------
import numpy as np
import pandas as pd
class FacotrYMC:
    #Class Attribute
    PurposeOfClass = "This class calculate the factor ymc" 
    AllInstances = []

    def __init__(self,VariableToConvert, TargetName, Data, FrequencyNumber = 100, Fun = np.mean, Suffix='_Mean_YMC'):
        # Run validation to the recieved arguments
        assert FrequencyNumber>0, f"FrequencyNumber {FrequencyNumber} is not greater of 0"

        # Assign to self object
        self.VarVariableToConvertiable = VariableToConvert
        self.TargetName = TargetName
        self.Data = Data
        self.FrequencyNumber = FrequencyNumber
        self.Fun = Fun
        self._Suffix = Suffix

        # Action to execute
        FacotrYMC.AllInstances.append(self)
    
    #property decoretor = Read - only Attribute
    @property
    def Name(self):
        return self._Suffix

    # @Name.setter
    # def Name(self,value):
    #     self._Name = value

    # Class method to initiate a class for examples
    @classmethod
    def InstantiateFromWeb(cls):
        Data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
        VariableToConvert = 'Sport'
        TargetName = 'Year'
        Data = Data
        FrequencyNumber = 100
        Fun = np.median
        Suffix='_Median_YMC'
        return FacotrYMC(VariableToConvert = VariableToConvert, TargetName = TargetName,Data = Data, FrequencyNumber = FrequencyNumber, Fun = Fun, Suffix = Suffix)

    def fit(self):
        VariableToConvert = self.VarVariableToConvertiable
        TargetName = self.TargetName
        Data = self.Data
        FrequencyNumber = self.FrequencyNumber 
        Fun = self.Fun 
        Suffix = self._Suffix
        
        # Creating variable to transform it to YMC ------------------------
        Variable = Data.loc[:, [TargetName,VariableToConvert]].set_axis(['TargetName','VariableToConvert'], axis=1)
        Variable.loc[:, 'VariableToConvert'] = Variable['VariableToConvert'].astype(str).fillna('NULL')  
    

        # Group all the Not Frequent Factor to one factor group -----------
        NotFrequentFactorGroup = pd.DataFrame(Variable.groupby('VariableToConvert')['TargetName'].apply(lambda x: 'Rare' if len(x) <= FrequencyNumber else 'Frequent')).reset_index()
        NotFrequentFactorGroup.columns = ["VariableName", "SmallGroupOrNot"]
        FrequentFactors = NotFrequentFactorGroup.loc[NotFrequentFactorGroup.SmallGroupOrNot == 'Frequent'].VariableName
        Variable.loc[:, 'VariableToConvert'] = np.where(Variable['VariableToConvert'].isin(FrequentFactors), Variable['VariableToConvert'], 'Not Frequent Factor')

        # Creating Dictionary
        Dictionary_Variable_YMC = Variable.groupby('VariableToConvert')["TargetName"].apply(Fun).reset_index()
        Dictionary_Variable_YMC.columns = ["Variable",TargetName+Suffix]
        Dictionary_Variable_YMC = Dictionary_Variable_YMC.sort_values(by=TargetName+Suffix, ascending=False)

        Dictionary = pd.DataFrame(data = {"VariableToConvert": Data[VariableToConvert].unique()})
        Dictionary['VariableToConvert'] = Dictionary['VariableToConvert'].astype(str).fillna('NULL') 
        Dictionary['Variable'] = np.where(Dictionary['VariableToConvert'].isin(FrequentFactors), Dictionary['VariableToConvert'], 'Not Frequent Factor')
        Dictionary = Dictionary.join(Dictionary_Variable_YMC.set_index('Variable'), how='left', on='Variable')
        Dictionary = Dictionary.drop(columns = 'Variable')
        Dictionary.columns = Dictionary_Variable_YMC.columns

        return Dictionary

    ## Name of the instance (reprecent of the instance)
    def __repr__(self):
        return f"YMC {self.__class__.__name__}('{self.Name}')"

    ## Read only property
    @property
    def ClassDescription(self):
        return "This class return factor ymc to a table with traget and variable to convert"


data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
FactorYMC1 = FacotrYMC(VariableToConvert = 'Event gender', TargetName = 'Year',Data = data, FrequencyNumber = 100, Fun = np.median, Suffix='_Median_YMC' )
FactorYMC1.AllInstances
FactorYMC1.fit()

FactorYMC2 = FacotrYMC.InstantiateFromWeb()
FactorYMC2.fit()