import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def CreateAPartyOfPeople(NumberOfPeople):
    # Create an empty array
    RandomNumbers = []

    # Populate the array with 100 random numbers between 0 and 364
    for _ in range(NumberOfPeople):
        RandomNumbers.append(random.randint(0, 364))

    # Print the array to verify the numbers
    return(RandomNumbers)

def CreateBirthdayDictinary(Party):
    # Create an empty dictionary
    BirthdayDictionary = {}
    for i in range(365):
        BirthdayDictionary[i] = 0 

    # Iterate over the list and update the dictionary
    for num in Party:
        BirthdayDictionary[num] += 1

    BirthdayDictionary = pd.DataFrame(list(BirthdayDictionary.items()), columns=['Day', 'Count'])
    return(BirthdayDictionary)

def CalculateSameDayBirthdayProbability(NumberOfPeople,NumberOfIterations):
    SimulationResults = []

    for _ in range(NumberOfIterations):
        Party = CreateAPartyOfPeople(NumberOfPeople)
        BirthdayDictionary = CreateBirthdayDictinary(Party)
        #Probability = 100.0*sum(BirthdayDictionary['Count'][BirthdayDictionary['Count']>1])/BirthdayDictionary['Count'].sum()
        Flag = any(BirthdayDictionary['Count']>1)
        SimulationResults.append(Flag)
    
    Probability = np.mean(SimulationResults)
    return(Probability)


ProbabilityDictionary = {}
for i in range(1,100):
    ProbabilityDictionary[i] = round(CalculateSameDayBirthdayProbability(NumberOfPeople = i,NumberOfIterations = 300),2)


ProbabilityDictionary = pd.DataFrame(list(ProbabilityDictionary.items()), columns=['NumberOfPeople', 'Probability'])
ProbabilityDictionary.loc[ProbabilityDictionary['NumberOfPeople']==15,:]

# Plot the table
plt.plot(ProbabilityDictionary['NumberOfPeople'], ProbabilityDictionary['Probability'], marker='o', linestyle='-')
plt.xlabel('NumberOfPeople')
plt.ylabel('Probability')
plt.grid(True)
plt.show()


