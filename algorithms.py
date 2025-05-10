# Importing libraries
import pandas as pd
import numpy as np
import re

#import for visualization
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import warnings
warnings.filterwarnings('ignore')





def gaussianNBModel(trial, output=False) :
    # Split data into features & target
    X = trial.drop('class', axis=1)
    y = trial['class']

    # Make training & est sets - 30% of data goes to test set, 70% to training - can change random_state to any # - 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train a NB Classifier (assuming it follows a normal distribution)
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Make prediction
    y_pred = nb_model.predict(X_test)
    
    
    # analysis
    a = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    if output == True: 
        print("Accuracy:", a)
        print("Classification Report:\n", cr)
        print("Confusion Matrix:\n", cm)


    return a, cr


def RandomForestModel(trial, output=False) :
    # Split data into features & target
    X = trial.drop('class', axis=1)
    y = trial['class']

    # Make training & test sets - 30% of data goes to test set, 70% to training - can change random_state to any # - 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train a RF Classifier
    rf_model = RandomForestClassifier(max_depth=10, n_estimators=300, bootstrap=False, min_samples_split=2, min_samples_leaf=1, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # Make prediction
    y_pred = rf_model.predict(X_test)

    # analysis
    a = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    if output == True: 
        print("Accuracy:", a)
        print("Classification Report:\n", cr)
        print("Confusion Matrix:\n", cm)

    return a, cr


def SupportVectorMachines(trial, output=False) :
    # Split data into features & target
    X = trial.drop('class', axis=1)
    y = trial['class']

    # Make training & test sets - 30% of data goes to test set, 70% to training - can change random_state to any # - 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # scale our data 
    sc = StandardScaler().fit(X_train)
    Xtr, Xte = sc.transform(X_train), sc.transform(X_test)

    # Train a SVM Classifier 
    svm = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    svm.fit(Xtr, y_train)
    y_pred = svm.predict(X_test)

    # Make prediction
    y_pred = svm.predict(Xte)

    # analysis
    a = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    if output == True: 
        print("Accuracy:", a)
        print("Classification Report:\n", cr)
        print("Confusion Matrix:\n", cm)

    return a, cr


def logistic_regression(trial, output=False) :

    # Split data into features & target
    X = trial.drop('class', axis=1)
    y = trial['class']

    # Make training & test sets - 30% of data goes to test set, 70% to training - can change random_state to any # - 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # scale our data 
    sc = StandardScaler().fit(X_train)
    Xtr, Xte = sc.transform(X_train), sc.transform(X_test)

    # Train a SVM Classifier 
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    # Make prediction
    y_pred = model.predict(Xte)

    # analysis
    a = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    if output == True: 
        print("Accuracy:", a)
        print("Classification Report:\n", cr)
        print("Confusion Matrix:\n", cm)

    return a, cr


def print_accuracy(trials, model):
    mean_accuracy = 0
    for trial in trials: 
        if model == 'GNB':
            trial_accuracy, trial_CR = gaussianNBModel(trial)
            mean_accuracy += trial_accuracy
        elif model == 'RF':
            trial_accuracy, trial_CR = RandomForestModel(trial)
            mean_accuracy += trial_accuracy
        elif model == 'SVM':
            trial_accuracy, trial_CR = SupportVectorMachines(trial)
            mean_accuracy += trial_accuracy
        elif model == 'LR':
            trial_accuracy, trial_CR = logistic_regression(trial)
            mean_accuracy += trial_accuracy
    mean_accuracy/=25
    return mean_accuracy


def analyze(trials, model):
    mean_accuracy = 0
    total_accuracy = []

    # accuracy for all 25 trials  
    for trial in trials: 
        if model == 'GNB':
            trial_accuracy, trial_CR = gaussianNBModel(trial)
            total_accuracy.append(trial_accuracy)
            mean_accuracy += trial_accuracy
        elif model == 'RF':
            trial_accuracy, trial_CR = RandomForestModel(trial)
            total_accuracy.append(trial_accuracy)
            mean_accuracy += trial_accuracy
        elif model == 'SVM':
            trial_accuracy, trial_CR = SupportVectorMachines(trial)
            total_accuracy.append(trial_accuracy)
            mean_accuracy += trial_accuracy
        elif model == 'LR':
            trial_accuracy, trial_CR = logistic_regression(trial)
            total_accuracy.append(trial_accuracy)
            mean_accuracy += trial_accuracy
    mean_accuracy/=25
    print(f'{model} Accuracy: {mean_accuracy/100:.2%}')

    # finding the top 3 trials for best accuracy 
    top3 = total_accuracy
    max1 = (0,0) #index, value
    max2 = (0,0) 
    max3 = (0,0) 
    size = len(top3)
    for i in range(size):
        if top3[i] > max1[1]:
            max1 = (i, top3[i]) 
    top3.remove(max1[1])

    size = len(top3)
    for i in range(size):
        if top3[i] > max2[1]:
            max2 = (i, top3[i]) 
    top3.remove(max2[1])

    size = len(top3)
    for i in range(size):
        if top3[i] > max3[1]:
            max3 = (i, top3[i]) 
    top3.remove(max3[1])
            
    a = (max1[1]+max2[1]+max3[1])/3
    print(f'Accuracy of top 3 trials: {a/100:.2%}')
    print(f"Top 3 trials: {max1[0]}, {max2[0]}, {max3[0]}")
    print(f'Top Accuracy: {max1[1]}')
