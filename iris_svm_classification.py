"""IMPLEMENTED USING GOOGLE COLAB"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn import svm  
from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from collections import Counter



iris=pd.read_csv('/content/IRIS.csv')

iris

sns.pairplot(data=iris, hue='species', palette='magma', height=3)

# Separating the independent variables from dependent variables

x=iris.iloc[:,:-1]
y=iris.iloc[:,4]
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30, random_state=30)

scaler= Normalizer().fit(x_train)
#print(x_train)
normalized_x_train= scaler.transform(x_train)
normalized_x_test= scaler.transform(x_test) 
#print(normalized_x_train)
#print(normalized_x_test)

from sklearn.svm import SVC
svc = svm.SVC(C=1.0, cache_size=200, class_weight=None,
  decision_function_shape='ovr', gamma='auto', kernel='rbf', max_iter=-1)

lin_svc = svm.SVC(C=1.0, cache_size=200, class_weight=None,
  decision_function_shape='ovr', gamma='auto', kernel='linear', max_iter=-1)

poly_svc = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly', max_iter=-1)

sigmoid_svc = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', gamma='auto', kernel='sigmoid', max_iter=-1)

svc.fit(x_train, y_train)
lin_svc.fit(x_train, y_train)
poly_svc.fit(x_train, y_train)
sigmoid_svc.fit(x_train, y_train)

pred=svc.predict(x_test)
lin_pred=lin_svc.predict(x_test)
poly_pred=poly_svc.predict(x_test)
sigmoid_pred=poly_svc.predict(x_test)

# Importing the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print(confusion_matrix(y_test,lin_pred))
print(confusion_matrix(y_test,poly_pred))
print(confusion_matrix(y_test,sigmoid_pred))

print("SVM with RBF Kernel \n" + classification_report(y_test, pred))

print("SVM with linear Kernel \n" + classification_report(y_test, lin_pred))

print("SVM with poly Kernel \n" + classification_report(y_test, poly_pred))

print("SVM with sigmoid Kernel \n" + classification_report(y_test, sigmoid_pred))