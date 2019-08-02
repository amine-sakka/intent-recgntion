import nltk
from nltk.stem.lancaster import LancasterStemmer
from IR import Mymodel 
import numpy as np
import tflearn
import random

import pickle
import json

import time

nbOflayers=3
nbOfEpoch=26000

with open('IR/content.json') as json_data:
    intents = json.load(json_data)
data= pickle.load(open("IR/trained_data", "rb"))
ERROR_THRESHOLD = 0.25

stemmer = LancasterStemmer()
data = pickle.load(open("IR/trained_data", "rb"))
words = data['words']
clases = data['classes']
train_x = data['trainX']
train_y = data['trainY']
intentsTitle=data["titles"]

#loading the model 
jarvis=Mymodel.MyTfModel(train_x,train_y,nbOflayers,nbOfEpoch,'IR/model.tflearn')
model=jarvis.getmodel()

def identfyIntent(sentence):
    #yrajlik intent 
    du=jarvis.classify(sentence,data)[0]
    #print(du)
    if du[1]>0.5:
        return(intentsTitle [int(du[0])-1])
    else:
        return("NULL")
        
ch=input("give a senence \n")
print(identfyIntent(ch))
