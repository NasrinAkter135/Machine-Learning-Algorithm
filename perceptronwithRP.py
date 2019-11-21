# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 02:29:53 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:50:43 2019

@author: Dell
"""
import numpy as np
import pandas as pn
import math
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import repeat
maindata=pn.read_csv("./data/perceptron_data.csv")
maindata=np.asarray(maindata)
X=maindata[:,:maindata.shape[1]-1]
y=maindata[:,-1:]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)
class  PerceptronN:
    value = None
    def __init__(self, maxiteration=None):
        self.maxiteration = maxiteration
        
    def train(self, features, labels):
        n = features.shape[1]
        w = np.random.rand(n+1)
        for label in labels:
            for feature in features:
                b=np.append(feature,1)
                value = np.dot(b, w)
                if value >= 0:
                    predictlabel=1
                    if not(predictlabel==label):
                         w=np.add(w,b)
                   
                       
                
    
                else:
                    predictlabel=0
                    if not(predictlabel==label):
                        w=np.subtract(w,b)
                        
           
        return w
    def fit(self, feature,labels):
        self.value = self.train(feature, labels)
    def predict(self, x):
        print("yesssssssssssssssss")
        print(x.shape[0])
        predictions = [] 
        w=self.value
        print(w.shape[0])
        for eachrowprediction in x:
            print(" i am in loop")
            print(eachrowprediction.shape[0])
            b=np.append(eachrowprediction,1)
            print("this is b")
            print(b)
            print(b.shape[0])
            predictvalue = np.dot(w,b)
            if predictvalue >= 0:
                predictlabel=1
                predictions.append(predictlabel)
            else :
                predictlabel=0
                predictions.append(predictlabel)
        print("prediction len")
        print(len(predictions))
        return predictions
def main(X_train,X_test,y_train,y_test):
     perceptron= PerceptronN()
     perceptron.fit(X_train,y_train)
     perceptron.train(X_train,y_train)
     predict=perceptron.predict(X_test)
     print(predict) 
     conf_matrix = confusion_matrix(y_test,predict)
     print(conf_matrix)
     print(f1_score(y_test,predict))
     print(accuracy_score(y_test,predict))
                    
main(X_train,X_test,y_train,y_test)