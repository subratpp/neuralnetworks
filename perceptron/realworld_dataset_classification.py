#!/usr/bin/env python
# coding: utf-8

# ## Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv

# This import is written by me which contains functions related to perceptrons
from perceptron import *


# ## =================1. Binary Classification: IRIS Dataset
# Class 0: IRIS-Setosa and Class 1: IRIS-Versicolor
# The actual IRIS data contains 3 classes, one of the class is removed to make the dataset for binary classification.
# The classes are linearly separable.
print(">>>>>>>>>>>>>>>>>>>>>>>>>>> IRIS: Binary Classification")

#Load data
data = []
with open('datasets/iris.data','r')as f:
    file = csv.reader(f)
    for row in file:
        if row[4] == 'Iris-setosa':
            row[4] = '0'
            float_list = list(map(float, row)) 
            data.append(float_list)
        
        elif row[4] == 'Iris-versicolor':
            row[4] = '1'
            float_list = list(map(float, row)) 
            data.append(float_list)
data = np.array(data)
np.random.shuffle(data) #Shuffle the input data
num_of_sample = len(data)

#Generate Test and Training Data from the Sample
test_data = data[:int(num_of_sample*0.15)]
training_data = data[int(num_of_sample*0.15):]

#Train the perceptron
no_of_inputs = 4
no_of_outputs = 1
epoch = 50
learning_rate=0.01
perceptron1 = Perceptron(no_of_inputs, no_of_outputs, epoch, learning_rate)
perceptron1.train(training_data)

#Test the perceptron
accuracy = perceptron1.test(test_data)
print(f"Model Accuracy for IRIS(2 class only) Dataset:{accuracy}%")


# ## ==================2. Multiclass Classification: Seeds Dataset (3 classes)
# The seeds dataset don't have linearlly separable classes. Linearly separable multiclass dataset is hard to find.
print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Seeds Dataset: Multiclass Classification- NOT linearly separable")

#load data
data = np.loadtxt("datasets/seeds_dataset.txt")
#The dataset has class level 1,2,3 which is changed to label 0,1,2 respectively to work with
data[0:, 7:] = data[0:, 7:] - 1
# 20% Test Data and 80% Training Data
np.random.shuffle(data) #Shuffle the input data
num_of_sample = len(data)

#Genearte training and testing data
test_data = data[:int(num_of_sample*0.2)]
training_data = data[int(num_of_sample*0.2):]

#Train the perceptron
no_of_inputs = 7
no_of_outputs = 3
epoch = 500
learning_rate=0.01
perceptron2 = Perceptron(no_of_inputs, no_of_outputs, epoch, learning_rate)
perceptron2.train(training_data)

#Test the perceptron
accuracy = perceptron2.test(test_data)
print(f"Model Accuracy for Seeds Dataset:{accuracy}%")


# ## ===================3. Multiclass Classification: Generated Dataset(3 Classes)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Generated Data: Multiclass Classification - Linearly Separable")

#generate data
data = generate_dataset(300, 3)
np.random.shuffle(data) #Shuffle the input data
num_of_sample = len(data)

#Generate Test and Training Data from the Sample
# 20% Test Data and 80% Training Data
test_data = data[:int(num_of_sample*0.2)]
training_data = data[int(num_of_sample*0.2):]

#Train the perceptron
no_of_inputs = 2
no_of_outputs = 3
epoch = 150
learning_rate=0.01
perceptron3 = Perceptron(no_of_inputs, no_of_outputs, epoch, learning_rate)
perceptron3.train(training_data)

#Test the perceptron
accuracy = perceptron3.test(test_data)
print(f"Model Accuracy for Generated 3 Class Dataset:{accuracy}%")

#>>>>>>>>>>> END
