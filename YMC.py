import numpy as np
import pandas as pd
class YMC:
    #Class Attribute
    PurposeOfClass = "YMC Transformation" 
    JitterRate = 0.00001
    AllInstances = []

    def __init__(self,VariableToConvert, TargetVariable, Fun = np.mean, Name = "Mean"):

        # Run validation to the recieved arguments
        assert isinstance(Name, str), f"Name {Name} is not a string"

        # Assign to self object
        self.VariableToConvert = VariableToConvert
        self.TargetVariable = TargetVariable
        self.Fun = Fun
        self._Name = Name

        # Action to execute
        YMC.AllInstances.append(self)
    
    #property decoretor = Read - only Attribute
    @property
    def Name(self):
        return self._Name

    # @Name.setter
    # def Name(self,value):
    #     self._Name = value

    # Class method to initiate a class for examples
    @classmethod
    def InstantiateFromWeb(cls):
        Data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
        VariableToConvert = Data['Year']
        TargetVariable = np.where(Data['Year']>=Data['Year'].median(),1,0)
        Fun = np.mean
        Name = "Mean"
        return YMC(VariableToConvert  = VariableToConvert, TargetVariable = TargetVariable,Fun = Fun,Name = Name)


    # Static method for claculating ranks dictionary
    @staticmethod
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

    
    # Jitter a numeric variable
    @staticmethod
    def RJitter(x,factor):
        z = max(x)-min(x)
        amount = factor * (z/50)
        x = x + np.random.uniform(-amount, amount, len(x))
        return(x)

    ## Name of the instance (reprecent of the instance)
    def __repr__(self):
        return f"YMC {self.__class__.__name__}('{self.Name}')"

    ## Read only property
    @property
    def ClassDescription(self):
        return "General YMC class"



### Implement NumericYMC Class that inherit YMC ------------------------------------------------------------------------ 
class NumericYMC(YMC):
    #Class Attribute
    PurposeOfClass = "Split to ranks and calculate YMC for specific variable" 

    def __init__(self,VariableToConvert: float, TargetVariable: float, NumberOfGroups = 10, Fun = np.mean, Name = "Mean", VariableName = 'VariableName'):
        
        ##Call to super function to have acces to all attributes
        super().__init__(VariableToConvert, TargetVariable, Fun, Name)

        # Run validation to the recieved arguments
        assert NumberOfGroups>0, f"NumberOfGroups {NumberOfGroups} is not greater of 0"
        assert isinstance(VariableName, str), f"VariableName {VariableName} is not a string"
        assert isinstance(Name, str), f"Name {Name} is not a string"

        # Assign to self object
        self.NumberOfGroups = NumberOfGroups
        self.VariableName = VariableName

    # Class method to initiate a class for examples
    @classmethod
    def InstantiateFromWeb(cls):
        Data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
        VariableToConvert = Data['Year']
        TargetVariable = np.where(Data['Year']>=Data['Year'].median(),1,0)
        VariableName = "Year"
        NumberOfGroups = 10
        Name = "Median"
        return NumericYMC(VariableToConvert = VariableToConvert,TargetVariable = TargetVariable,NumberOfGroups = NumberOfGroups,Fun = np.median, Name = Name, VariableName = VariableName)

    def fit(self):
        VariableToConvert = self.VariableToConvert
        TargetVariable = self.TargetVariable
        NumberOfGroups = self.NumberOfGroups
        Fun = self.Fun
        Name = self.VariableName + '_' + self.Name + '_NumericYMC'
        
        # Create dictionary for Variable
        Dictionary = YMC.Ranks_Dictionary(YMC.RJitter(VariableToConvert,YMC.JitterRate), ranks_num = NumberOfGroups)
        Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                        Dictionary['value'],
                                                        closed='left')
        # Convert Each value in variable to rank
        Variable = pd.DataFrame({'Variable': VariableToConvert, 
                                 'Target': TargetVariable})
        IntervalLocation = Variable['Variable']
        Variable['Rank'] = Dictionary.loc[IntervalLocation]['Rank'].reset_index(drop=True)
        del IntervalLocation

        # Aggregation Table
        Dictionary = Dictionary.merge(Variable.groupby('Rank')['Target'].apply(Fun).reset_index().set_axis(['Rank', Name], axis=1), how='left', on=['Rank'])

        #Fill NA with the Function outcomes on all the variable
        Dictionary.loc[:, Name] = Dictionary[Name].fillna(Fun(Variable['Target'].dropna()))

        return Dictionary

    ## Read only property
    @property
    def ClassDescription(self):
        return "This class return numeric ymc"


### Implement FacotrYMC Class that inherit YMC ------------------------------------------------------------------------ 
class FacotrYMC(YMC):
    #Class Attribute
    PurposeOfClass = "FactorYMC class" 

    def __init__(self,VariableToConvert , TargetVariable, FrequencyNumber = 100, Fun = np.mean, Name = "Mean", VariableName = 'VariableName'):
        
        ##Call to super function to have acces to all attributes
        super().__init__(VariableToConvert, TargetVariable, Fun, Name)

        # Run validation to the recieved arguments
        assert FrequencyNumber>0, f"NumberOfGroups {FrequencyNumber} is not greater of 0"
        assert isinstance(VariableName, str), f"NTargetNameame {VariableName} is not a string"
        assert isinstance(Name, str), f"NTargetNameame {Name} is not a string"

        # Assign to self object
        self.FrequencyNumber = FrequencyNumber
        self.VariableName = VariableName


    # Class method to initiate a class for examples
    @classmethod
    def InstantiateFromWeb(cls):
        Data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
        VariableToConvert = Data['Sport']
        TargetVariable =  np.where(data['Year']>=data['Year'].median(),1,0)
        FrequencyNumber = 100
        Fun = np.median
        Name = "Mean"
        VariableName = "Sport"
        return FacotrYMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, FrequencyNumber = FrequencyNumber, Fun = Fun, Name = Name, VariableName = VariableName)

    def fit(self):
        VariableToConvert = self.VariableToConvert
        TargetVariable = self.TargetVariable
        FrequencyNumber = self.FrequencyNumber 
        Fun = self.Fun 
        VariableName = self.VariableName
        Suffix = VariableName + '_' + self.Name + '_FactorYMC' 
        
        # Creating variable to transform it to YMC ------------------------
        Variable = pd.DataFrame({'TargetVariable':TargetVariable,
                                 'VariableToConvert':VariableToConvert})
        Variable.loc[:, 'VariableToConvert'] = Variable['VariableToConvert'].astype(str).fillna('NULL')  
    
        # Group all the Not Frequent Factor to one factor group -----------
        NotFrequentFactorGroup = pd.DataFrame(Variable.groupby('VariableToConvert')['TargetVariable'].apply(lambda x: 'Rare' if len(x) <= FrequencyNumber else 'Frequent')).reset_index()
        NotFrequentFactorGroup.columns = ["VariableName", "SmallGroupOrNot"]
        FrequentFactors = NotFrequentFactorGroup.loc[NotFrequentFactorGroup.SmallGroupOrNot == 'Frequent'].VariableName
        Variable.loc[:, 'VariableToConvert'] = np.where(Variable['VariableToConvert'].isin(FrequentFactors), Variable['VariableToConvert'], 'Not Frequent Factor')

        # Creating Dictionary
        Dictionary_Variable_YMC = Variable.groupby('VariableToConvert')["TargetVariable"].apply(Fun).reset_index()
        Dictionary_Variable_YMC.columns = ["VariableToConvert",Suffix]
        Dictionary_Variable_YMC = Dictionary_Variable_YMC.sort_values(by=Suffix, ascending=False)

        Dictionary = pd.DataFrame(data = {"Variable": VariableToConvert.unique()})
        Dictionary['Variable'] = Dictionary['Variable'].astype(str).fillna('NULL') 
        Dictionary['VariableToConvert'] = np.where(Dictionary['Variable'].isin(FrequentFactors), Dictionary['Variable'], 'Not Frequent Factor')
        Dictionary = Dictionary.join(Dictionary_Variable_YMC.set_index('VariableToConvert'), how='left', on='VariableToConvert')
        Dictionary = Dictionary.drop(columns = 'VariableToConvert')
        Dictionary.columns = [VariableName,Suffix]
        Dictionary = Dictionary.sort_values(by=Suffix, ascending=False)

        return Dictionary

    ## Read only property
    @property
    def ClassDescription(self):
        return "This class return factor ymc"
    

# ## Example of instantiating YMC class-------------------------------------------------------------------------------
# data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
# VariableToConvert = data['Year']
# TargetVariable = np.where(data['Year']>=data['Year'].median(),1,0)
# Fun = np.mean
# Name = "Mean"
# YMC1 = YMC.InstantiateFromWeb()
# YMC2 = YMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, Fun = np.median, Name='Median')
# YMC.AllInstances


# ##Example of instantiating NumericYMC class------------------------------------------------------------------------
# data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
# NumericYMC1 = NumericYMC(VariableToConvert = data['Year'],TargetVariable = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median",VariableName = 'Year')
# NumericYMC2 = NumericYMC(VariableToConvert = data['Year'],TargetVariable = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.mean,Name = "Mean",VariableName = 'Year')
# NumericYMC3 = NumericYMC.InstantiateFromWeb()
# NumericYMC1.fit()
# NumericYMC2.fit()
# NumericYMC3.fit()

# ##Example of nstantiating factor class-----------------------------------------------------------------------------
# data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
# VariableToConvert = data['Sport']
# TargetVariable = np.where(data['Year']>=data['Year'].mean(),1,0)
# FacotrYMC1 = FacotrYMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, FrequencyNumber = 100, Fun = np.mean, Name = "Mean", VariableName = 'Sport')
# FacotrYMC2 = FacotrYMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, FrequencyNumber = 100, Fun = np.median, Name = "Median", VariableName = 'Sport')
# FacotrYMC3 = FacotrYMC(VariableToConvert = VariableToConvert, TargetVariable = TargetVariable, FrequencyNumber = 100, Fun = np.max, Name = "Max", VariableName = 'Sport')
# FacotrYMC1.fit()
# FacotrYMC2.fit()
# FacotrYMC3.fit()