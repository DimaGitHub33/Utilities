class YMC:
    def __init__(self, Name):
        print(f"YMC Instatnce Created For: {Name}")
        self.name = Name
    def Fun1(self, x, y):
        return x*y


ymc1 = YMC("YMC")
ymc1.name = "Factor YMC"
ymc1.Variables = 5
ymc1.a = 2
ymc1.b = 4
ymc1.Fun1(ymc1.a,ymc1.b)
print(ymc1)



ymc2 = YMC()
ymc2.name = "Factor YMC"
ymc2.Variables = 5
ymc2.a = 2
ymc2.b = 4
ymc2.Fun1(ymc2.a,ymc2.b)
print(ymc2)


##  NumericYMC -------------------------------------------------------
import numpy as np
import pandas as pd
from YMC import RJitter, Ranks_Dictionary
class NumericYMC:
    #Class Attribute
    PurposeOfClass = "This class calculate the numeric ymc" 
    JitterRate = 0.00001

    def __init__(self,Variable: float, Target: float, NumberOfGroups = 10,Fun = np.mean,Name = "Mean"):
        # Run validation to the recieved arguments
        assert NumberOfGroups>0, f"NumberOfGroups {NumberOfGroups} is not greater of 0"

        # Assign to self object
        self.Variable = Variable
        self.Target = Target
        self.NumberOfGroups = NumberOfGroups
        self.Fun = Fun
        self.Name = Name

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


# Example 
data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
Variable = data['Year']
Target = np.where(data['Year']>=data['Year'].mean(),1,0)
NumberOfGroups = 10
#NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median")

YMC = NumericYMC(Variable  = Variable,Target = Target,NumberOfGroups = 10,Fun = np.mean,Name = "Mean")
YMC.YMC()
print(NumericYMC.PurposeOfClass)
print(YMC.PurposeOfClass)

print(NumericYMC.__dict__)## All the atributes for class level
print(YMC.__dict__)## All the atributes for instance level