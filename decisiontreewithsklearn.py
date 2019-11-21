# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:06:17 2019

@author: Dell
"""
import numpy as np
import pandas as pn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
maindata=pn.read_csv("./data/data.csv")
maindata=np.asarray(maindata)
X = maindata[:,0:19]
y = maindata[:,19]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))