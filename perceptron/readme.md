
# Perceptron from Scratch
Name: Subrat Prasad Panda
Problem: To build perceptron from scratch and use *i.IRIS* and *ii.self generated dataset* for training and testing of model.

## Dependencies:
- numpy
- matplotlib
- csv

## Execute:
```console
python3 realworld_dataset_classification.py
python3 generated_dataset_classification.py
python3 perceptron.py
```
*Note: Please remove the plot functions in case of any ERROR in plotting. Comment out the sections for the plotting in the program which is marked with comments.*

**Note: perceptron.py is tested for an example test case which generates perceptron for AND logic.*

## Output:
- Output will be printed on the terminal
- Accuracy will get printed for test dataset
- Percentage error plots while training the perceptron

## Description
1. perceptron.py
	- The program for perceptron is written in perceptron.py which includes the perceptron class.
	- It also includes the function definition to generate dataset.

2. realworld_dataset_classification.py
	- One Real world dataset is used(seeds_dataset) which has 3 classes. Accuracy 100%
	- One self generated dataset is used for multiclass (3class) classification. Accuracy 100%
	- IRIS Dataset is used for Binary Classification. Accuracy around 80%-90% which varies with each execution.

3. generated_dataset_classification.py
	- Generated own dataset which is used for binary classification. Accuracy 100%

4. The percentage error in each epoch is plotted during training.

## Use perceptron.py Library:
```python
form perceptron import *
no_of_inputs = 4
no_of_outputs = 1
epoch = 50
learning_rate=0.01

perceptron1 = Perceptron(no_of_inputs, no_of_outputs, epoch, learning_rate) #create perceptron object
perceptron1.train(training_data) #train perceptron

accuracy = perceptron1.test(test_data) #Test the perceptron #test perceptron
print(f"Model Accuracy for IRIS(2 class only) Dataset:{accuracy}%") #print accuracy of model on test data
```
