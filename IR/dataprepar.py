import path
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
import tflearn
import random
import pickle
import json

stemmer = LancasterStemmer() #take the word and return the root of it 
stopWords = set(stopwords.words('english')) #stop words
ignorewords = ['?',"!",":p",":)",":*",":/"]

def intentsmaker(pathTojasonfile):
    #return data in the jason file
    with open(pathTojasonfile) as json_data:
        intents = json.load(json_data)
    return(intents)

def datamaker(intents):
    #work throw the dataset and return 3 list 
    #word classes, documents
    words,clases,documents,intentTitlt = [],[],[],[]
    #words ['Hi', 'Is', 'anyone', 'there'
    #clases tags ['1', '2', '5', '6', '7', '8', '9']
    #document list mte3 tuple (word,tag)=(word,clases)  [(['Hi'], '1')

    for intent in intents['intents']:
        #print(intent)
        for patt in intent['patterns']:
            #print('patt',patt)
            w = nltk.word_tokenize(patt) #split fil nltk 
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in clases:
                clases.append(intent['tag'])
        intentTitlt.append(intent["intentTitle"])
        
    words = [stemmer.stem(w.lower()) for w in words if w not in ignorewords and w not in stopWords  ]
    return((words,clases,documents,intentTitlt))
        




#intents=intentsmaker("data.json")
#words,clases,documents=datamaker(intents)
#print("words",words)
#print("cla",clases)
#print("doc",documents)