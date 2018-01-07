#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:09:59 2018
@author: Brettmccausland
"""
                                    #to import libraries command enter
#Data Preproccessing
                                    #Importing the libraries
import numpy as np                  #include for mathmatics libraries
import matplotlib.pyplot as plt     #ploting library
import pandas as pd                 #import data sets

#importing the Datasets
dataset = pd.read_csv('Data.csv')   #to execute highlight and command enter
X = dataset.iloc[:,:-1].values      #[take all the lines , except the last one]
y = dataset.iloc[:,3].values        #Indepedent value vector

#taking care of missing librarys
#using the strategy of replacing the missing values with the mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3]) #taking indexes 1 and 2 , it doesnt take the upper bound
X[:,1:3]=imputer.transform(X[:,1:3])

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split #import library
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

"""#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()         #create object of labelencoder class
X[:,0] =labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()"""


"""labelencoder_y = LabelEncoder();
y = labelencoder_y.fit_transform(y)"""

#feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""













