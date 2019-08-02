import numpy as np
import tensorflow as tf
import tflearn
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
import json

stemmer = LancasterStemmer()
ERROR_THRESHOLD = 0.25
stemmer = LancasterStemmer()


class MyTfModel:
    
    def __init__(self,trainX,trainY,nbOfHiddenLayers,nbOfEpoch,path):
        if path =="":
            tf.reset_default_graph()
            net = tflearn.input_data(shape=[None, len(trainX[0])])
            
            for i in range(nbOfHiddenLayers):
                net = tflearn.fully_connected(net, 8)

            net = tflearn.fully_connected(net, len(trainY[0]), activation='softmax')
            net = tflearn.regression(net)
            self.model = tflearn.DNN(net, tensorboard_dir='train_logs')
            self.model.fit(trainX, trainY, n_epoch=nbOfEpoch, batch_size=500, show_metric=True)
        else:
            net = tflearn.input_data(shape=[None, len(trainX[0])])
        
            for i in range(nbOfHiddenLayers):
                net = tflearn.fully_connected(net, 8)

            net = tflearn.fully_connected(net, len(trainY[0]), activation='softmax')
            net = tflearn.regression(net)
            self.model = tflearn.DNN(net, tensorboard_dir='train_logs')
            self.model.load(path)

   
    def getmodel(self):
        return(self.model)

    def clean_up_sentence(self,sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return (sentence_words)

    def bow(self,sentence, words, show_details=False):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def classify(self,sentence,data):
        words = data['words']
        clases = data['classes']
        train_x = data['trainX']
        train_y = data['trainY']
        intentsTitle=data["titles"]
        results = self.model.predict([self.bow(sentence, words)])[0]
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((clases[r[0]], r[1]))
        return return_list
        
    
