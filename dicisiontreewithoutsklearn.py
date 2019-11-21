# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:42:15 2019

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
attributes=[]
maindata=pn.read_csv("./data/data.csv")
maindata=np.asarray(maindata)
traindata,testdata = train_test_split(maindata,test_size=0.2,random_state=0)
for i in range(traindata.shape[1]):
    attributes.append(i)

class Node:
    def __init__(self, attribute=None, attribute_values=None, child_nodes=None, decision=None):
        self.attribute = attribute
        self.attribute_values = attribute_values
        self.child_nodes = child_nodes
        self.decision = decision
class DecisionTree:

    root = None
    
    @staticmethod
    def plurality_values(data): # return the value in the labels variable(in the skeleton code) that occurs the most.
        s = []
        labels = data[:, data.shape[1] - 1]  # store the last column in labels
        values=np.unique(data[:, labels])
        for value in values:
            count=0
            for label in labels:
                if label==value:
                    count=count+1
            s.append((count,value))
        s.sort(reverse=True)
        highlabel=s.pop()
        high=highlabel[1]
        return high
    @staticmethod        
    def all_zero(data): # return True if all the values in the labels variable is 0, otherwise return False
            count=0
            labels = data[:, data.shape[1] - 1]  # store the last column in labels
            value=data.shape[0]
            for label in labels:
                if label==0:
                     count=count+1
                else:
                     return False
            if count>=value:
                return True                

    @staticmethod
    def all_one(data):#return True if all the values in the labelsvariable is 1, otherwise return False
        labels = data[:, data.shape[1] - 1]  # store the last column in labels
        count=0
        value=labels.shape[0]
        for label in labels:
            if label==1:
                 count=count+1
            else:
                 return False
        if count>=value:
            return True
        
    @staticmethod
    def importance(data,attributes):#calculate the information gain to find out the best attribute
        decisionlabels = data[:, data.shape[1] - 1]  # store the last column in labels
        decisionvalues=np.unique(data[:, decisionlabels])
        parententropy = 0
        for value in decisionvalues:
            count=0
            for label in decisionlabels:
                if value==label:
                    count=count+1
            fraction = float(count/len(data[decisionlabels]))
            parententropy+=-fraction*np.log2(fraction)
        gain=[]
        for attribute in attributes:
           
            attributelabels=data[:,attribute]
            size = attributelabels.shape[0]
            attributevalues=np.unique(data[:, attributelabels])
            totalnumbers=0
            attributeentropy=0
            for singlevalue in attributevalues:
                for value in decisionvalues:
                    pncount=0
                    singleattributenum = 0
                    for n in range(size):
                        if((data[n][attribute])==singlevalue):
                            singleattributenum = singleattributenum+1
                            if(data[n][data.shape[1]-1]==value):
                                pncount=pncount+1
                    if not (pncount==0):
                        fraction = float(pncount/singleattributenum)
                        attributeentropy+=float(-fraction*np.log2(fraction))
                totalnumbers+=((singleattributenum/len(attributelabels))*attributeentropy)    
                 
            infogain=parententropy-totalnumbers
            gain.append((infogain,attribute))
        gain.sort(reverse=True)
        gainattribute=gain.pop()
        highattribute=gainattribute[1]
        return highattribute
    def train(self, data, attributes, parent_data):
        data = np.array(data)
        parent_data = np.array(parent_data)
        attributes = list(attributes)

        if data.shape[0] == 0:  # if x is empty
            return Node(decision=self.plurality_values(parent_data))

        elif self.all_zero(data):
            return Node(decision=0)

        elif self.all_one(data):
            return Node(decision=1)

        elif len(attributes) == 0:
            return Node(decision=self.plurality_values(data))

        else:
            a = self.importance(data, attributes)
            tree = Node(attribute=a, attribute_values=np.unique(data[:, a]), child_nodes=[])
            attributes.remove(a)
            for vk in np.unique(data[:, a]):
                new_data = data[data[:, a] == vk, :]
                subtree = self.train(new_data, attributes, data)
                tree.child_nodes.append(subtree)

            return tree

    def fit(self, data):
        self.root = self.train(data, list(range(data.shape[1] - 1)), np.array([]))

    def predict(self, data):
        predictions = []
        for i in range(data.shape[0]):
            current_node = self.root
            while True:
                if current_node.decision is None:
                    current_attribute = current_node.attribute
                    current_attribute_value = data[i, current_attribute]
                    if current_attribute_value not in current_node.attribute_values:
                        predictions.append(random.randint(0, 1))
                        break
                    idx = list(current_node.attribute_values).index(current_attribute_value)

                    current_node = current_node.child_nodes[idx]
                else:
                    predictions.append(current_node.decision)
                    break

        return predictions

def main(traindata,testdata,attributes):
     decision= DecisionTree()
     decision.fit(traindata)
     decision.train(traindata, attributes , 0)
     predict=decision.predict(testdata)
     print(predict)
     testpredict=testdata[:, testdata.shape[1] - 1] 
     conf_matrix = confusion_matrix(testpredict,predict)
     print(conf_matrix)
     print(f1_score(testpredict,predict))
     print(accuracy_score(testpredict,predict))
                    
main(traindata,testdata,attributes)