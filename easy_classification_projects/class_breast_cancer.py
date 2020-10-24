#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:54:09 2020

@author: marcusnotohansen

page:https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
 
classification with naive bayes on breast cancer_ 3 steps:
    1)Organizing Data into Sets
    2) choosing model adn evaluate model
    3) evaluate model´s accuracy

"""

import sklearn
from sklearn.datasets import load_breast_cancer


# Load dataset
data = load_breast_cancer()


# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Look at our data
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])


from sklearn.model_selection import train_test_split


# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

'''
test_size parameter. In this example, we now have a test set (test) that represents 33% of the original dataset.
'''

# gaussian naive bayes:
    
from sklearn.naive_bayes import GaussianNB


# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)  

# Make predictions
preds = gnb.predict(test)
print(preds)

#Evaluating the Model’s Accuracy

from sklearn.metrics import accuracy_score


# Evaluate accuracy
print(accuracy_score(test_labels, preds))




