# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('final_project_dataset_1.csv')
x=dataset.iloc[:,0:6].values
y=dataset.iloc[:,[6]].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x=LabelEncoder()
labelencoder_x.fit_transform(x[:,1])
x[:,1]=labelencoder_x.fit_transform(x[:,1])
ct=ColumnTransformer([("sex",OneHotEncoder(),[1])],remainder='passthrough')
X=ct.fit_transform(x)
X=X[:,1:]

labelencoder_x1=LabelEncoder()
labelencoder_x1.fit_transform(x[:,4])
x[:,4]=labelencoder_x1.fit_transform(x[:,4])
ct1=ColumnTransformer([("smoker",OneHotEncoder(),[4])],remainder='passthrough')
X1=ct.fit_transform(x)
X1=X1[:,1:]

labelencoder_x2=LabelEncoder()
labelencoder_x2.fit_transform(x[:,5])
x[:,5]=labelencoder_x2.fit_transform(x[:,5])
ct2=ColumnTransformer([("country",OneHotEncoder(),[5])],remainder='passthrough')
X2=ct.fit_transform(x)
X2=X2[:,1:]

y = np.array(y, dtype=int)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.2,random_state=0)

'''from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)'''
y=np.array(y).reshape(-1,1)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
#預測測試項
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

plt.scatter(y_test,y_pred)
plt.plot(y_test,y_test,color='red')
plt.title('Random Forest')
plt.xlabel('Actual')
plt.ylabel('Predicted')

