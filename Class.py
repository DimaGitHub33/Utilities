##  NumericYMC -------------------------------------------------------
import numpy as np
import pandas as pd
from YMCFunctions import RJitter, Ranks_Dictionary
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

    # Static method just for implement in the future 
    @staticmethod
    def is_integer(Num):
        if isinstance(Num,float):
            return Num.is_integer()
        elif isinstance(Num,int):
            return True
        else:
            return False


    def YMC(self):
        Variable = self.Variable
        Target = self.Target
        NumberOfGroups = self.NumberOfGroups
        Fun = self.Fun
        Name = self.Name
        
        # Create dictionary for Variable
        Dictionary = Ranks_Dictionary(RJitter(Variable,NumericYMC.JitterRate), ranks_num=NumberOfGroups)
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
    def ReadOnlyProperty(self):
        return "secific string to fill in the future"


# Example 
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
Variable = data['Year']
Target = np.where(data['Year']>=data['Year'].mean(),1,0)
NumberOfGroups = 10
NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median")

YMC1 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.mean,Name = "Mean")
YMC2 = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.median,Name = "Median")
YMC3 = NumericYMC.InstantiateFromWeb()

YMC3.Name = "New"
YMC3.YMC()
##All the Instances
for Instances in NumericYMC.AllInstances:
    print(Instances)

print(NumericYMC.AllInstances)




print(NumericYMC.__dict__)## All the atributes for class level
print(YMC.__dict__)## All the atributes for instance level



### Implement Updated Class that inherit NumericYMC ------------------------------------------------------------------------ 
class NumericYMCImproved(NumericYMC):
    #Class Attribute
    PurposeOfClass = "Improve NumericYMC class" 

    def __init__(self,Variable: float, Target: float, NumberOfGroups = 10,Fun = np.mean,Name = "Mean",NewFeature="Somthing"):
        
        ##Call to super function to have acces to all attributes
        super().__init__(Variable, Target, NumberOfGroups, Fun, Name)

        # Run validation to the recieved arguments
        assert isinstance(NewFeature, str), f"NewFeature {NewFeature} is not string"

        # Assign to self object
        self.NewFeature = NewFeature




data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
Variable = data['Year']
Target = np.where(data['Year']>=data['Year'].mean(),1,0)
NumberOfGroups = 10
ImprovedYMC1 = NumericYMCImproved(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.mean,Name = "Mean",NewFeature="Improved")
ImprovedYMC2 = NumericYMCImproved(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.mean,Name = "Median",NewFeature="Improved")

##All the Instances
for Instances in NumericYMCImproved.AllInstances:
    print(Instances)