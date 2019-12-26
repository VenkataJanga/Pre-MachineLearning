# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 22:38:23 2019

@author: Venkata Sai
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.under_sampling import NearMiss
from collections import Counter

from imblearn.combine import SMOTETomek

from imblearn.over_sampling import RandomOverSampler

path = 'C:/Users/Venkata Sai/Downloads/'
data = pd.read_csv(path+'creditcard.csv',sep=',')
print(data.head())

print(data.isnull().any())

print(data.info())

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(X.shape,y.shape)
print(X.head())

print(data["Class"].value_counts())
#data["Class"].value_counts().plot(kind = 'bar')

#split the fraud data and non fraud data from data set

fraud_data = data[data['Class']==1]
non_fraud_data = data[data['Class']==0]
print(fraud_data.shape);print();print(non_fraud_data.shape)

nm = NearMiss()

X_res,y_res = nm.fit_sample(X,y)
print(X_res.shape, y_res.shape)


print("Orginal data set  shape {}".format(Counter(y)))
print("Resampled  data set  shape {}".format(Counter(y_res)))

smt = SMOTETomek()
X_train,y_train = smt.fit_sample(X,y)
print(X_train.shape, y_train.shape)
print("Orginal data set  shape using SMOTETomek {}".format(Counter(y)))
print("Resampled  data set  shape using SMOTETomek {}".format(Counter(y_train)))


os = RandomOverSampler(ratio = 0.5)
X_train_res,y_train_res = os.fit_sample(X,y)


print(X_train_res.shape, y_train_res.shape)
print("Orginal data set  shape using RandomOverSampler {}".format(Counter(y)))
print("Resampled  data set  shape using RandomOverSampler {}".format(Counter(y_train_res)))
