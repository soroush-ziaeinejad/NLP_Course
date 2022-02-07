##import pytrec_eval
##import json
import numpy as np
import time
import pickle
from nltk.corpus import wordnet
##import nltk
##nltk.download('wordnet')
wordnet.all_synsets()

# for pytrec_eval
def DictonaryGeneration(topCorrections, Misspelleds, Corrects):
    Correction = {}
    for c in range(len(Misspelleds)):
        Correction[Misspelleds[c]] = {}
        comm = topCorrections[c]
        for n in range(len(comm)):
            Correction[Misspelleds[c]][comm[n]] = 1
    Truth = {}
    for c in range(len(Corrects)):
        Truth[Misspelleds[c]] = {}
        Truth[Misspelleds[c]][Corrects[c]] = 1
    return  Correction, Truth


def MED_DP(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1]+1,   # Insert
                                   dp[i-1][j]+1,   # Remove
                                   dp[i-1][j-1]+2) # Replace
 
    return dp[m][n]
## Dataset Preparation
dataset = [i.strip() for i in open("missp.dat").readlines()]
correct = []
misspelled = []
for word in dataset:
    if word[0]=='$':
        previous_word = word[1:]
    else:
        correct.append(previous_word.lower())
        misspelled.append(word.lower())
print(f'Dataset loaded with {len(dataset)} entries and {len(correct)} unique words.')
Dictionary = []
for i in wordnet.all_synsets():
    Dictionary.append(i.name().split('.')[0])
Dictionary = np.asarray(Dictionary)
print("len dictionary:", len(Dictionary))
Dictionary = np.unique(Dictionary)
print("len dictionary after removing duplicates:", len(Dictionary))
limit = 5#len(misspelled)
AVG_SATK = []
K=10
TOPS = []
for mcidx,ms in enumerate(misspelled):
    tic = time.time()
##    if mcidx == limit:
##        break
    distances = []
    beg = time.time()
    for cidx, c in enumerate(Dictionary):
##        if cidx%40000==0:
##            print(ms, c, cidx, time.time()-beg)
            #MED(ms,c,len(ms),len(c))
        distances.append(MED_DP(ms,c,len(ms),len(c)))
    distances = np.asarray(distances)
##    idx = distances.argsort()
    idx = distances.argpartition(range(K))[:K]
    kTops = Dictionary[np.array(idx)]
    TOPS.append(kTops)
    print(f'Most similar words to {ms}: {kTops}')
    print(f'Most similar edits: {distances[np.array(idx[:K])]}')
    print(f'Ground Truth: {correct[mcidx]}')
    print(f'{mcidx}/{len(misspelled)}')
    print(f'{time.time()-tic} seconds')
    print('-------------------------------------------')
q, run = DictonaryGeneration(TOPS, misspelled[:limit], correct[:limit])
with open(f'corrections_at_k10.pkl', 'wb') as f:
    pickle.dump(q, f)
with open(f'golden_standard_at_k10.pkl', 'wb') as f:
    pickle.dump(run, f)
'''
with open('corrections_at_k10.pkl', 'rb') as f:
    q = pickle.load(f)
with open('golden_standard_at_k10.pkl', 'rb') as f:
    run = pickle.load(f)
evaluator = pytrec_eval.RelevanceEvaluator(q, {'success_1', 'success_5', 'success_10'})
print(json.dumps(evaluator.evaluate(run), indent=1))
'''
