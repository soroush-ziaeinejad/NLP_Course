from nltk.util import bigrams
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.corpus import brown
import numpy as np
import pytrec_eval
import json
##import pyterrier as pt
##pt.init()

brownDS = brown.words(categories='news')
text = [w.lower() for w in brownDS]
def DictonaryGeneration(topCorrections, Misspelleds, Corrects):
    Correction = {}
    for c in range(len(Misspelleds)):
        Correction[Misspelleds[c]] = {}
        comm = topCorrections[c]
        for n in range(len(comm)):
##            print(Correction)
            Correction[Misspelleds[c]][comm[n]] = 1
    Truth = {}
    for c in range(len(Corrects)):
        Truth[Misspelleds[c]] = {}
        Truth[Misspelleds[c]][Corrects[c]] = 1
    return  Correction, Truth
def LM(ng, text):
    padded_bigrams = list(pad_both_ends(text, n=ng))
    train, vocab = padded_everygram_pipeline(ng, [text])
    lm = MLE(ng)
    lm.fit(train, vocab)
    return lm

def dataProcessing(path):
    datContent = [i.strip() for i in open(path, encoding='utf8').readlines()]
    misspelleds = []
    corrects = []
    sentences = []
    for i in datContent:
        row = i.split()
        if len(row)>1:
            misspelleds.append(row[0])
            corrects.append(row[1])
            sentences.append(row[2:])
    return misspelleds, corrects, sentences
def evaluatorfunc(topN,k,M,C):
    T = np.asarray(topN)
    q, run = DictonaryGeneration(T[:,:k], M, C)
    evaluator = pytrec_eval.RelevanceEvaluator(q, {f'success_{k}'})
    a = json.dumps(evaluator.evaluate(run), indent=1)
    b = evaluator.evaluate(run)
    return b
N = [1,2,3,5,10]
##LMs = []
##for n in N:
##    print('n: ', n)
##    lm = LM(n,text)
##    LMs.append(lm)
##    print(lm.score("a"))
##    print(lm.score("the"))
##    print(lm.score("grand", ['the', 'fulton', 'county']))
##    print('-----------------------------')


file = 'APPLING1DAT.643'
M, C, S = dataProcessing(file)
TopTens = []
text2 = list(set(text))
nptext = np.asarray(text2)
for nn,n in enumerate(N):
    print(f'gram: {nn}/{len(N)}')
    TopTens.append([])
    lm = LM(n,text)
    for ss,sentence in enumerate(S):
        starindex = sentence.index('*')
        input = sentence[max(0,starindex-n+1):starindex]
        scores = []
        for t in nptext:
            score = lm.score(t, input)
            scores.append(score)
        if ss%40==0:
            print(f'{ss}/{len(S)}')
        scores = np.asarray(scores)
        idx = np.flip(scores.argsort()[-10:])
        
##        idx2 = np.argpartition(scores, -10)[-10:]
##        idx2[np.argsort(scores[idx2])]
        kTops = nptext[np.array(idx)]
        TopTens[-1].append(kTops)
    T = np.asarray(TopTens[-1])
    
    a = evaluatorfunc(T[:,:1],1,M,C)
    d1 = []
    for i in a.keys():
        d1.append(a[i]['success_1'])
    print(f'{n}-gram mean success_1:',sum(d1)/len(d1))
    
    b = evaluatorfunc(T[:,:5],5,M,C)
    d5 = []
    for i in b.keys():
        d5.append(b[i]['success_5'])
    print(f'{n}-gram mean success_5:',sum(d5)/len(d5))
    
    c = evaluatorfunc(T[:,:10],10,M,C)
    d10 = []
    for i in c.keys():
        d10.append(c[i]['success_10'])
    print(f'{n}-gram mean success_10:',sum(d10)/len(d10))
    print('----------------------------------')
    
