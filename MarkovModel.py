### Load libraries
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import string
from sklearn.model_selection import train_test_split

##Remove the scientific notations
np.set_printoptions(suppress=True)

# Word2Index Function----------------------------------------- 
def Word2IndexFun(Sequences):
    Index = 1
    Word2Index = {'<unk>': 0}
    for Sequence in Sequences:
        for Word in Sequence:
            #print(token)
          if Word not in Word2Index:
            Word2Index[Word] = Index
            Index += 1
    return Word2Index
 

# Convert data into integer format -----------------------------------------
def ConvertDataIntoIntegerFormat(Sequences,Word2Index):
    SequencesToInt = []
    
    for Sequence in Sequences:
      #print(Sequence)
      SequencesAsInt = [Word2Index.get(Word, 0) for Word in Sequence]
      SequencesToInt.append(SequencesAsInt)
  
    return SequencesToInt


# initialize A and pi matrices - for both classes -----------------------------
def ComputeMarkovModelObjects(IntSequences,Word2Index):
    V = len(Word2Index)
    A = np.ones((V, V)).astype(float) ##Markov State Transition for specific label
    pi = np.ones(V).astype(float) ##Markov Initial State Distribution for specific label 

    # Compute counts for A and pi for Specific label  -------------------------
    for IntSequence in IntSequences:
        LastIndex = None
        for Index in IntSequence:
            if LastIndex is None:
                # it's the first word in a sentence
                pi[Index] += 1
            else:
                # the last word exists, so count a transition
                A[LastIndex, Index] += 1
        
            # update last idx
            LastIndex = Index
            
            
    # Normalize A and pi so they are valid probability matrices ---------------
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    return A,pi
    

# Compute priors ---------------------------------------------------------
def ComputePriors(Labels):
    p0 = sum(np.array(Labels)==0) / len(Labels)
    p1 = sum(np.array(Labels)==1) / len(Labels)
    return p0, p1




# Compute LogLike lihkliood ---------------------------------------------------------
# logA = Markov matrix sequence probabilities for specific labeled chain (log transformation)
# logpi = Initials state probabilities for specific labeled chain (log transformation)
# logp = Overall probability of the label (log transformation)
# Input = Sequence of events
def ComputeLogLikelihood(Input,logA,logPi,logP):
    
    # if this is the first event in the Input sequence we will use the logpi for the first state probabilities
    # else we will take the probability of the chain from one step to the second from the markov model matrix
    LastIndex = None
    LogProb = 0
    for Index in Input:
      if LastIndex is None:
        # it's the first token
        LogProb += logPi[Index]
      else:
        LogProb += logA[LastIndex, Index]
      
      # update last_idx
      LastIndex = Index
    
    return LogProb + logP



def FitMarkovModel(Sequences,Labels):
    # Word2Index Function------------------------------------------------------
    Word2Index = Word2IndexFun(TrainText)   

    # Convert data into integer format ----------------------------------------
    IntSequences = ConvertDataIntoIntegerFormat(Sequences,Word2Index)

    #Markov Model transition and Initial probabilities ------------------------
    A0,pi0 = ComputeMarkovModelObjects([t for t, y in zip(IntSequences, Labels) if y == 0],Word2Index) 
    A1,pi1 = ComputeMarkovModelObjects([t for t, y in zip(IntSequences, Labels) if y == 1],Word2Index) 

    # Compute priors ----------------------------------------------------------
    p0, p1 = ComputePriors(Labels = YTrain) 

    #Preparing the Output -----------------------------------------------------
    MarkovModel = A0,pi0,A1,pi1,Word2Index,p0, p1
    
    return MarkovModel


# Prediction ---------------------------------------------------------
def PredictMarkovModel(PredictSequence,MarkovModel):
    ClasifiersPredictions = []
    
    A0,pi0,A1,pi1,Word2Index,p0, p1 = MarkovModel
    PredictIntSequences = ConvertDataIntoIntegerFormat(Sequences = PredictSequence,Word2Index = Word2Index)
    for Sequence in PredictIntSequences:
        #print(Sequence)
        Label0 = ComputeLogLikelihood(Input = Sequence,logA = np.log(A0),logPi = np.log(pi0),logP = np.log(p0))
        Label1 = ComputeLogLikelihood(Input = Sequence,logA = np.log(A1),logPi = np.log(pi1),logP = np.log(p1))
        
        if Label0 >= Label1:
            Out = 0
        else:
            Out = 1
            
        ClasifiersPredictions.append(Out)
    
    return ClasifiersPredictions
        

################################ Run #########################################

# collect data into lists ------------------------------------
InputText = [] ## Sequence of Sequences
Labels = [] ## Sequence of labels (0 or 1)

##Filling InputText and Labels
for ClaimNo in TimeSequenceQuery['claim_no'].unique():
    SubSequence = TimeSequenceQuery.loc[TimeSequenceQuery['claim_no']==ClaimNo,]
    InputText.append(SubSequence['descriptions1'])
    Labels.append(SubSequence['suspicious'].unique()[0])

# Train Test ------------------------------------------------
TrainText, TestText, YTrain, YTest = train_test_split(InputText, Labels)

##Markov Model Objet with transition probabilities, initial probabilities and Wordt2Index
MarkovModel = FitMarkovModel(Sequences = TrainText,Labels = YTrain)

##Predict Outpus
Output = PredictMarkovModel(PredictSequence = TestText,MarkovModel = MarkovModel)

##Check the model
from sklearn.metrics import confusion_matrix, f1_score
cm_test = confusion_matrix(YTest, Output)
100*cm_test/len(YTest)

print(f"f1 score Test: {f1_score(YTest, Output)}")
