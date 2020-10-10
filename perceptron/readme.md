
# Perceptron from Scratch
Name: Subrat Prasad Panda

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

4. The percentage error in each epoch is plotted during training

