##  NumericYMC -------------------------------------------------------
import numpy as np
import pandas as pd
class NumericYMC:
    #Class Attribute
    PurposeOfClass = "This class calculate the numeric ymc" 
    JitterRate = 0.00001
    AllInstances = []

    def __init__(self,Variable: float, Target: float, NumberOfGroups = 10,Fun = np.mean,Name = "Mean"):
        # Run validation to the recieved arguments
        assert NumberOfGroups>0, f"NumberOfGroups {NumberOfGroups} is not greater of 0"

        # Assign to self object
        self.Variable = Variable
        self.Target = Target
        self.NumberOfGroups = NumberOfGroups
        self.Fun = Fun
        self._Name = Name

        # Action to execute
        NumericYMC.AllInstances.append(self)
    
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
        Variable = Data['Year']
        Target = np.where(Data['Year']>=Data['Year'].median(),1,0)
        return NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 5,Fun = np.mean,Name = "Mean")


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


    def fit(self):
        Variable = self.Variable
        Target = self.Target
        NumberOfGroups = self.NumberOfGroups
        Fun = self.Fun
        Name = self.Name
        
        # Create dictionary for Variable
        Dictionary = NumericYMC.Ranks_Dictionary(NumericYMC.RJitter(Variable,NumericYMC.JitterRate), ranks_num=NumberOfGroups)
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
        Dictionary = pd.DataFrame()
        Dictionary['Rank'] = Variable['Rank'].unique()
        Dictionary = Dictionary.merge(Variable.groupby('Rank')['Variable'].apply(Fun).reset_index().set_axis(['Rank', Name], axis=1), how='left', on=['Rank'])

        #Fill NA with the Function outcomes on all the variable
        Dictionary.loc[:, Name] = Dictionary[Name].fillna(Fun(Variable['Variable'].dropna()))

        return Dictionary

    ## Name of the instance (reprecent of the instance)
    def __repr__(self):
        return f"YMC {self.__class__.__name__}('{self.Name}')"

    ## Read only property
    @property
    def ClassDescription(self):
        return "This class return numeric ymc to specific target and variable to convert"



data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
Variable = data['Year']
Target = np.where(data['Year']>=data['Year'].mean(),1,0)
NumberOfGroups = 10
NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median")

YMC1 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.mean,Name = "Mean")
YMC1.fit()

YMC2 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.median,Name = "Median")
YMC3 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = percentile(90),Name = "Percentile")