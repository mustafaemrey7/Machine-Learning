# -*- coding: utf-8 -*-
"""example_a0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SIBeeEAdIvhqOW9awiDbR2KfK6A1P9u_

# BBM409 Assignment_0

Group Member: Hanifi Aslankarayiğit

Group Member: Mustafa Emre Yıldırım
"""

""" import necessary libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report





""" upload dataset from your PC, you can use different methods"""
#from google.colab import files
#data = files.upload()
df = pd.read_csv('star_classification.csv')
df.head(20)

"""## Pre-processing"""

#X= df.iloc[:,:-5].values
#X= df[['obj_ID','alpha','delta','u','g','r','i','z','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','redshift','plate','MJD','fiber_ID']]
#X = df.drop(columns=['class'])
X = df[['obj_ID','alpha', 'delta', 'u', 'g', 'r', 'i', 'z','redshift']]#We can minimize runtime by minimizing the number of unique features we choose.
#X= df.drop(['class'],axis=1)
#X= df.iloc[:,:2].values
#X= df[['run_ID','spec_obj_ID']]
#y= df.iloc[:,13].values
y=df['class']
#y=df['class']

"""Explain why you use those methods, etc.

## Split the dataset

80% training & 20% test  or 5-fold cross validation
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#The purpose of this code is to standardize or normalize your features, making them have similar scales and
#ensuring that your machine learning model is not influenced by the different units or magnitudes of the input features.

"""## Classification Methods

In this homework, we use kNN, Naive Bayes ...

### kNN
"""

# import KNeighbors Classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
#n_neighbors determines the number of nearest neighbors to consider when making a classification decision. In this case, the KNN classifier will consider the 5 nearest neighbors.
classifier.fit(X_train,y_train)#KNN classifier uses the training data to "learn" patterns and relationships between the features and the corresponding labels. It effectively memorizes the training data

y_pred = classifier.predict(X_test)
#It is used to make predictions on a set of unseen data points using the trained KNN classifier.

"""Explain your results, draw plots, tables etc."""

print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)

"""### Weighted KNN"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
#n_neighbors determines the number of nearest neighbors to consider when making a classification decision. In this case, the KNN classifier will consider the 5 nearest neighbors.
classifier.fit(X_train,y_train)#KNN classifier uses the training data to "learn" patterns and relationships between the features and the corresponding labels. It effectively memorizes the training data

y_pred = classifier.predict(X_test)
#It is used to make predictions on a set of unseen data points using the trained KNN classifier.
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)

"""### Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
# Gaussian Naive Bayes is a probabilistic classifier based on the Bayes' theorem, which is commonly used for classification tasks.
clf=GaussianNB()
#This line initializes an instance of the Gaussian Naive Bayes classifier and assigns it to the variable clf. This clf object will be used to train and make predictions with the Gaussian Naive Bayes model.
clf.fit(X_train,y_train)
#This line trains the Gaussian Naive Bayes classifier using the provided training data.
#When you call clf.fit(X_train, y_train), the Gaussian Naive Bayes classifier estimates the statistical parameters (mean and variance) of each feature for each class in the training data.
#It assumes that the features are normally distributed within each class (hence "Gaussian"), and it uses this information to calculate the likelihood of observing a specific combination of features for each class.
#It also estimates the prior probabilities of each class.

y_pred=clf.predict(X_test)
#It is used to make predictions on a set of unseen data points using the trained Naive bayes classification
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)

"""### Random Forest

"""

from sklearn.ensemble import RandomForestClassifier

#The RandomForestClassifier is an ensemble learning method based on decision trees, which is used for classification tasks.
rforest = RandomForestClassifier()
#Random Forest algorithm creates a collection of decision trees, where each tree is trained on a bootstrapped subset of the training data (bagging) and selects a random subset of features to split on (feature bagging).
#The ensemble of decision trees works together to make predictions.
rforest.fit(X_train,y_train)

rf_predictions = rforest.predict(X_test)
#It is used to make predictions on a set of unseen data points using the trained Random Forest classification
print("Classification Report for Random Forest:")
print(classification_report(y_test, rf_predictions))
cm=confusion_matrix(y_test,y_pred)
print(cm)

"""###Svm"""

from sklearn.svm import SVC

svm_model = SVC()
#The SVC class stands for Support Vector Classification and is used for binary and multi-class classification tasks.
#The SVM algorithm uses this information to learn a decision boundary that separates different classes.
svm_model.fit(X_train,y_train)
svm_predictions = svm_model.predict(X_test)
#It is used to make predictions on a set of unseen data points using the trained SVM classification
print("Classification Report for SVM:")
print(classification_report(y_test, svm_predictions))
cm=confusion_matrix(y_test,y_pred)
print(cm)

"""###Explain your results, draw plots, tables etc."""

colors = {'QSO': 'r', 'GALAXY': 'g', 'STAR': 'b'}
color_labels = [colors[label] for label in y_test]

plt.scatter(X_test[:, 1], X_test[:, 2], c=color_labels)
plt.xlabel("Unique feature1")
plt.ylabel("Unique feature2")

plt.show()

"""We can prefer knn or naive bayes instead of svm and random forest in classifying and predicting larger data. The reason for this is that svm and random forest algorithms run slower on big data.
Confusion matrix provides a detailed breakdown of true positives, true negatives, false positives, and false negatives, which can help you understand where the model is making errors.
"""