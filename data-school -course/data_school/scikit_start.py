#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intro scipy adn iris dataset
- supervised learning problem: predict the species of an iris using the measurements

"""
import sklearn
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()
print (iris.data)
print (iris.feature_names)
print (iris.target)
print (iris.target_names)
x=iris.data
y=iris.target

print(x.shape)
print(y.shape)

knn=KNeighborsClassifier(n_neighbors=1)
print(knn)

#fit the model with data
knn.fit(x,y)
knn.predict([[3,5,4,2]])
X_new=[[3,5,4,2],[5,4,3,2]]
a=knn.predict(X_new)
print(a)

#using a different classification model: regression:
    
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
#logreg = LogisticRegression()

# fit the model with data
#logreg.fit(x, y)

# predict the response for new observations
#b=logreg.predict(X_new)
#print(b)

#EVALUATION:
    ##1: Train and test on the entire dataset¶
#MODEL:Logistic regression
#logreg = LogisticRegression()
#logreg.fit(x, y)
#logreg.predict(x)
# store the predicted response values
#y_pred_reg = logreg.predict(x)

# check how many predictions were generated
#print(len(y_pred_reg))

# compute classification accuracy for the logistic regression model
from sklearn import metrics
#print(metrics.accuracy_score(y, y_pred_reg))

y_pred_knn = knn.predict(x)
print(metrics.accuracy_score(y, y_pred_knn))

#Evaluation procedure #2: Train/test split

# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
#kan we find a better value for k?
#try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt


# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

# instantiate the model with the best known parameters
knn = KNeighborsClassifier(n_neighbors=11)

# train the model with X and y (not X_train and y_train)
knn.fit(x, y)

# make a prediction for an out-of-sample observation
knn.predict([[3, 5, 4, 2]])