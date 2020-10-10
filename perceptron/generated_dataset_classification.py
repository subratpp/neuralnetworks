#Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv

# This import is written by me which contains functions related to perceptrons
from perceptron import *


# ==============================================Generated Dataset: Binary Classification
#generate data
data = generate_dataset(300, 2) #function defined in mylibrary
np.random.shuffle(data) #Shuffle the input data
num_of_sample = len(data)

#Generate Test and Training Data from the Sample
# 20% Test Data and 80% Training Data
test_data = data[:int(num_of_sample*0.2)]
training_data = data[int(num_of_sample*0.2):]

#Train the perceptron
no_of_inputs = 2
no_of_outputs = 1
epoch = 100
learning_rate = 0.01
perceptron = Perceptron(no_of_inputs, no_of_outputs, epoch, learning_rate)
perceptron.train(training_data)

#Test the perceptron
accuracy = perceptron.test(test_data)
print(f"Model Accuracy for Generated 2 class dataset:{accuracy}%")
