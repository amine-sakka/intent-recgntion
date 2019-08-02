import dataprepar as dpp
from nltk.stem.lancaster import LancasterStemmer
import random
import Mymodel
import numpy as np
import pickle
import path

nbOflayers=3
nbOfEpoch=26000
stemmer = LancasterStemmer()

intents=dpp.intentsmaker("content.json")
words,classes,documents,intentTitle=dpp.datamaker(intents)
print("document",documents)

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "Docs")
print(len(classes), "Classes", classes)
print(len(words), "Split words", words)
print(len(intentTitle), "intentTitle", intentTitle)
training = []
output = []
output_empty = [0] * len(classes) 

for doc in documents:
    bag = []
    pattern_words = doc[0]
    #print("patt",pattern_words)
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])


#print(training)
random.shuffle(training)
training = np.array(training)

trainX = list(training[:, 0])
trainY = list(training[:, 1])

jarvis=Mymodel.MyTfModel(trainX,trainY,nbOflayers,nbOfEpoch,"")
model=jarvis.getmodel()
model.save('model.tflearn')



with open('trained_data', 'wb') as f:
    pickle.dump({'words': words, 'classes': classes, 'trainX': trainX, 'trainY': trainY,"titles":intentTitle}, f)
