# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 02:38:47 2019

@author: Dell
"""
import numpy as nps
import pandas as pn
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
maindata=pn.read_csv("./data/perceptron_data.csv")
maindata=np.asarray(maindata)
X=maindata[:,:maindata.shape[1]-1]
y=maindata[:,-1:]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)
n_iter=40
eta0=0.1
psklearn = Perceptron(n_iter=n_iter,eta0=eta0, random_state=0)
psklearn.fit(X_train, y_train)  
y_pred = psklearn.predict(X_test)
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
