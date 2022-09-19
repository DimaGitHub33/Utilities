#%% ------------------------------------------------------------
import numpy as np
import string
 
np.random.seed(1234)
 
#%% ------------------------------------------------------------
# collect data into lists ------------------------------------
InputText = [] ## Sequence of Sequences
Labels = [] ## Sequence of labels (0 or 1)
 
##Filling InputText and Labels
for ClaimNo in TimeSequenceQuery['claim_no'].unique():
    SubSequence = TimeSequenceQuery.loc[TimeSequenceQuery['claim_no']==ClaimNo,]
    InputText.append(SubSequence['descriptions1'])
    Labels.append(SubSequence['suspicious'].unique()[0])
   
#%% ------------------------------------------------------------
Initial = {} # start of a phrase
FirstOrder = {} # second word only
SecondOrder = {} # probability of the third word given two privious words
 
#%% ------------------------------------------------------------
##I don't use it in the script
def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))
 
#%% ------------------------------------------------------------
def Add2Dictionary(Dictionary, Index, Value):
    if Index not in Dictionary:
        Dictionary[Index] = []
    Dictionary[Index].append(Value)
 
 
#%% ------------------------------------------------------------
for Line in InputText:
    #print(line)
    #tokens = remove_punctuation(line.rstrip().lower()).split()
    Line = np.array(Line)
    T = len(Line)
    for i in range(T):
        Word = Line[i]
        if i == 0:
            # Distributions of the first Words
            Initial[Word] = Initial.get(Word, 0.) + 1
        else:
            ##Distribution of all not the First words
            PreviousWord = Line[i-1]
            if i == T - 1:
               # Measure probability of ending the line
                Add2Dictionary(SecondOrder, (PreviousWord, Word), 'END')
            if i == 1:
                # Measure distribution of second word given only first word
                Add2Dictionary(FirstOrder, PreviousWord, Word)
            else:
                # Measure distribution of third word given the two last words
                SecondPreviousWord = Line[i-2]
                Add2Dictionary(SecondOrder, (SecondPreviousWord, PreviousWord), Word)
 
#%% ------------------------------------------------------------
# Normalize the distributions
InitialTotal = sum(Initial.values())
for t, c in Initial.items():
    #print(t)
    #print(c)
    Initial[t] = c / InitialTotal
 
#%% ------------------------------------------------------------
# convert [cat, cat, cat, dog, dog, dog, dog, mouse, ...]
# into {cat: 0.5, dog: 0.4, mouse: 0.1}
 
def List2ProbDictionary(ts):
    # turn each list of possibilities into a dictionary of probabilities
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t, c in d.items():
        d[t] = c / n
    return d
 
#%% ------------------------------------------------------------
for SpecificWord, AllTheWordsThatGoAfterTheSpecificWord in FirstOrder.items():
    #print(SpecificWord)
    #print(AllTheWordsThatGoAfterTheSpecifiWord)
    # Replace list with dictionary of probabilities
    FirstOrder[SpecificWord] = List2ProbDictionary(AllTheWordsThatGoAfterTheSpecificWord)
 
#%% ------------------------------------------------------------
for SpecificTwoWord, AllTheWordsThatGoAfterTheSpecificTwoWord in SecondOrder.items():
    #print(SpecificTwoWord)
    #print(AllTheWordsThatGoAfterTheSpecificTwoWord)
    # Replace list with dictionary of probabilities
    SecondOrder[SpecificTwoWord] = List2ProbDictionary(AllTheWordsThatGoAfterTheSpecificTwoWord)
 
#%% ------------------------------------------------------------
def SampleWord(d):
    # print "d:", d
    p0 = np.random.random()
    # print "p0:", p0
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t
    assert(False) # should never get here
 
#%% ------------------------------------------------------------
def generate():
    for i in range(4): # generate 4 lines
        sentence = []
       
        # Initial word
        w0 = SampleWord(Initial)
        sentence.append(w0)
       
        # Sample second word
        w1 = SampleWord(FirstOrder[w0])
        sentence.append(w1)
       
        # second-order transitions until END
        while True:
            w2 = SampleWord(SecondOrder[(w0, w1)])
            if w2 == 'END':
                break
            sentence.append(w2)
            w0 = w1
            w1 = w2
        print(' '.join(sentence))
 
#%% ------------------------------------------------------------
generate()
 
 