import nltk
from nltk.corpus import brown, wordnet
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cs
import matplotlib.pyplot as plt
import copy
################ Dataset ################
def data():
##    brownDS = brown.words(categories='news')
    brownDS = brown.sents(categories=['news'])#, 'editorial', 'reviews', 'mystery',
                                      #'fiction','hobbies'])
    dd =[]
    for i in brownDS:
        dd.append(" ".join(i))
    print(len(dd))
##    Dataset1 = [w.lower() for w in brownDS]
##    Dataset1 = np.asarray(brownDS)
##    print("len brown:", len(Dataset1)) #100,554
##    Dataset1 = np.unique(Dataset1)
##    print("len brown after removing duplicates:", len(Dataset1)) #13,112
    return dd
####    nltk.download('wordnet')
##    wordnet.all_synsets()
##    Dataset2 = []
##    for i in wordnet.all_synsets():
##        Dataset2.append(i.name().split('.')[0])
##    Dataset2 = np.asarray(Dataset2)
##    print("len wordnet:", len(Dataset2)) #117,659
##    Dataset2 = np.unique(Dataset2)
##    print("len wordnet after removing duplicates:", len(Dataset2)) #86,555
##
##    dataset = np.concatenate((Dataset1,Dataset2))
##    print("len dataset:", len(dataset)) #99,667
##    dataset = np.unique(dataset)
##    print("len dataset after removing duplicates:", len(dataset)) #93,293, 6,374 Common words
##    return dataset

def notten(a):
  c = 0
  for ii,i in enumerate(a):
    if len(a[i])!=10:
      c+=1
  return c

def transitive(topk,W1,W2,Sim,vocab):
    for word in topk.keys():
        if len(topk[word]) >= 10:
            topk[word] = sorted(topk[word], reverse=True)[:10]
        else:
            topk[word] = sorted(topk[word], reverse=True)
            tw = np.asarray(topk[word])
            addedWords = []
            for sim,word2 in topk[word]:
                if word2 in W1:
                    indices = [i for i, x in enumerate(W1) if x == word2]
                    for ind in indices:
                        if W2[ind] not in tw[:,1] and W2[ind]!=word and W2[ind] not in addedWords:
                            addedWords.append([Sim[ind]*(sim/max(Sim)),W2[ind]])
                            #topk[word].append([Sim[ind]*(sim/max(Sim)),W2[ind]])
                if word2 in W2:
                    indices = [i for i, x in enumerate(W2) if x == word2]
                    for ind in indices:
                        if W1[ind] not in tw[:,1] and W1[ind]!=word and W1[ind] not in addedWords:
                            addedWords.append([Sim[ind]*(sim/max(Sim)),W1[ind]])
                            #topk[word].append([Sim[ind]*(sim/max(Sim)),W1[ind]])
            addCounter = 0
            addedWords = sorted(addedWords, reverse=True)
            while len(topk[word])<10 and addCounter<len(addedWords):
                topk[word].append(addedWords[addCounter])
                addCounter+=1
            #if len(topk[word]) >= 10:
                #topk[word] = sorted(topk[word], reverse=True)[:10]
    return topk


def transitivityAnalysis(topk,W1,W2,Sim,vocab, render=True):
    lens = []
    lens.append(notten(topk))
    topk = transitive(topk,W1,W2,Sim,vocab)
    while notten(topk) != lens[-1]:
        lens.append(notten(topk))
        topk = transitive(topk,W1,W2,Sim,vocab)
    if render:
        plt.plot(range(len(lens)),lens)
        plt.ylabel('# similar words != 10')
        plt.xlabel('# transitive operations')
        plt.title('# required transitive operations')
        plt.show()
    return topk


def golden_top_K(W1,W2,Sim,vocab):
    topk = {}
    for i in range(len(W1)):
        if W1[i] in vocab:
            first = W1[i]
            second = W2[i]
            if first not in topk.keys():
                topk[first] = [[Sim[i], second]]
            else:
                topk[first].append([Sim[i], second])
        if W2[i] in vocab:
            first = W2[i]
            second = W1[i]
            if first not in topk.keys():
                topk[first] = [[Sim[i], second]]
            else:
                topk[first].append([Sim[i], second])
##    transitiveTopk = transitive(topk,W1,W2,Sim,vocab)
    return topk
        

##    topk = {}
##    WW1 = np.concatenate((W1,W2))
##    WW2 = np.concatenate((W2,W1))
##    Sim = np.concatenate((Sim,Sim))
##    W1 = WW1
##    W2 = WW2
##    print(W1.shape)
##    print(W2.shape)
##    print(Sim.shape)
##    for i,w1 in enumerate(W1):
##        if w1 in vocab:
##            if w1 not in topk.keys():
##                topk[w1] = []
##            indexesL2R = np.where(np.array(W1) == w1)[0]
##            for ind in indexesL2R:
##                if [W2[ind],Sim[ind]] not in topk[w1]:
##                    topk[w1].append([W2[ind],Sim[ind]])
##            if len(topk[w1]) > 10:
##                print(w1, topk[w1])
##    return topk


C = data()
f = open("SimLex-999/SimLex-999.txt", encoding = 'utf-8')
W1 = []
W2 = []
Sim = []
for line in f.readlines()[1:]:
    fields = line.strip().split('\t')
    W1.append(fields[0].lower())
    W2.append(fields[1].lower())
    Sim.append(float(fields[3]))
f.close()
W1 = np.asarray(W1)
W2 = np.asarray(W2)
Sim = np.asarray(Sim)



tfIdfVectorizer=tfidf(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(C)
d = pd.DataFrame(tfIdf.T.todense())
cos = cs(d)
a = tfIdfVectorizer.vocabulary_.keys()
vocab = []
for i in a:
    vocab.append(i)
gg = golden_top_K(W1,W2,Sim,vocab)
print(notten(gg))
GroundTruth = transitivityAnalysis(copy.deepcopy(gg),W1,W2,Sim,vocab)
print(notten(GroundTruth))
##df = pd.DataFrame(tfIdf.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
##df = df.sort_values('TF-IDF', ascending=False)
##print (df.head(25))

##vectorizer = tfidf()
##X = vectorizer.fit_transform(C)
