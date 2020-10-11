#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

#For Plotting
import matplotlib.cm as cm 
import matplotlib.pyplot as plt

#For clearing the screen
#from IPython.display import clear_output



class deepNN:
    '''
    Class for Multilayer Neural Networks
    1. Relu Activation Function at Hidden Layer
    2. Softmax for outer layer
    '''
    
    #Initialize the weights matrixes and the layers
    def __init__(self, layer_dim, learning_rate, gamma, epoch):
        self.layer_dim = layer_dim # number of nodes in each layer as list eg [784,500,400,300,100,10]
        self.num_of_layers = len(layer_dim) #number of layers including input and output layer
        self.num_of_features = layer_dim[0] #number of input features 784
        self.num_of_classes = layer_dim[-1] #number of output features 10
        self.num_of_samples = 0 #total number of data samples eg 42000
        self.epoch = epoch #number of training iterations
        self.learning_rate = learning_rate #learning rate of gradient descend
        self.gamma = gamma #momentum factor
        
        #Weights of input
        self.weights = {}
        self.velocity = {} #to store momentum during training
        
        # Create dictionary for each layer {W1, b1, W2, b2 ..} and initialize randomly with normal distribution
        # Layer eg: a-l1->b-l2->c here a,b and c are nodes and l1 and l2 layer parameteres. W and b is defined on l1/l2
        # He-Normal initilazation with std = 2/fan_in
        for l in range(1, self.num_of_layers):
            self.weights['W' + str(l)] = np.random.normal(0, np.sqrt(2/self.layer_dim[l-1]), (self.layer_dim[l], self.layer_dim[l-1]) ) #He Weight Initialization for ReLu
            self.weights['b' + str(l)] = np.zeros((self.layer_dim[l], 1))
            self.velocity['V' + str(l)] = np.zeros((self.layer_dim[l], self.layer_dim[l-1]))
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Architecture Notations @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # (prevlayer_op: X)A_prev--> |W,b|--> |Z = WX+b|--> |activation(Z)| --> | A | -->(next layer) 
    # Z = W.A + B where A is the input from previous layer
    # A = activation(Z) output at each layer: Relu(Z)
    # Y_hat = Output at the final layer of multi layer NN
    # Y = Actual Data
    # X = Input data of features at input layer
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    #=============== Utility to plot error
    @staticmethod
    def error_plot(error_data, filename):
        x_axis = list(range(1,len(error_data)+1))
        #print(x_value)
        #print(cost_plot)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Error')
        plt.plot(x_axis, error_data, color='g')
        plt.savefig(filename + "_traininingPlot.pdf", bbox_inches='tight')
        print("If executing from the terminal: Please CLOSE the Figure to continue...")
        plt.show()
    
    
    #================1. Activation Function used in the NN
    # Relu in internal layers and Softmax in output layer
    @staticmethod
    def relu(Z):    
        return np.maximum(0, Z) #Return A: Activation or next layer o/p
    
    
    #Softmax at outer layer
    @staticmethod
    def softmax(Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis = 0) #Return A: Activation or next-layer o/p
    
    
    #=================2. Function to compute Affine Tranformation and Activation 
    #forward propagation
    @staticmethod
    def affine_forward(A, W, b):
        # A: From previous layer W: weights # b: Bias
        Z = np.dot(W, A) + b # Z = WA + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        return Z

    
    def activation_forward(self, A_prev, W, b, activation):
        if activation == "relu":
            Z = self.affine_forward(A_prev, W, b)  # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            A = self.relu(Z) 
        
        elif activation == "softmax":
            Z = self.affine_forward(A_prev, W, b) # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            A = self.softmax(Z)
        
        # cache = ( (cacheAWb), cacheZ )@ cacheAWb: stores A_prev, W and b @cacheZ: Stores Z of a layer 
        cache = ( (A_prev, W, b), (Z) ) #All the information of this layer is stored
        return A, cache
    
    
    #================== 3. Forward Propagation
    def forward_propagation(self, X):
        self.num_of_samples = X.shape[1] #Define the Number of samples to be processed together
        caches = [] #evaluation data in each layer
        A = X #X: Data at the input layer: Data with all sample is feed
        L = self.num_of_layers - 1
        # Relu at the Hidden Layers
        for l in range(1, L):
            A_prev = A  #make A as A of previous layer
            A, cache = self.activation_forward(A_prev, self.weights['W' + str(l)], self.weights['b' + str(l)], activation = "relu")
            caches.append(cache)
        # Softmax at the output layer Y_hat
        Y_hat, cache = self.activation_forward(A, self.weights['W' + str(L)], self.weights['b' + str(L)], activation = "softmax")
        caches.append(cache)               
        return Y_hat, caches
    
    
    #==================4. Cross Entropy Loss function over the all samples 
    def loss_function(self, Y_hat, Y):
        cross_entropy_loss = (-1 / self.num_of_samples) * np.sum( np.multiply(Y, np.log(Y_hat)) ) #+ np.multiply(1 - Y, np.log(1 - Y_hat)) )
        return cross_entropy_loss  #loss = sum(Y*log(Y_hat))
    
    
    #==================5. Back Propagation: 
    #@@@@@@@@@@@@@@@@@@ Notations: derivative with respect to error(cross_entropy_error)
    #dZ = delE/delZ
    #dA = delE/delA
    #dW = delE/delW
    #db = delE/delb
    #@@@@@@@@@@@@@@@@@@
    #=======================i> Derivative of Activaiton Function
    @staticmethod
    def relu_derivative(dA, cacheZ):
        dZ = dA.copy() # 1 for the positive values and multiplied with dA i.e dZ.dA
        dZ[cacheZ < 0] = 0 #zero for negative values of Z  #Z = cacheZ
        assert (dZ.shape == cacheZ.shape)
        return dZ

    #========================ii> Backprop
    # Gradient at Z = WA + b
    def affine_backward(self, dZ, cacheAWb):
        A_prev, W, b = cacheAWb 
        m = self.num_of_samples 
        dW = (1/m) * np.dot(dZ, A_prev.T)  #average gradient from all samples and # dE/dw = dE/dz * A^
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True) #average gradients from all samples
        dA_prev = np.dot(W.T, dZ) #compute dE/dA_prev, it will be passed to previous layer evaluation W^ * dE/dZ
        return dA_prev, dW, db
    
    
    # Gradient at A = Relu(Z)
    def activation_backward(self, dA, cache, activation):
        cacheAWb, cacheZ = cache
        if activation == "relu":
            dZ = self.relu_derivative(dA, cacheZ)
            dA_prev, dW, db = self.affine_backward(dZ, cacheAWb)  
        elif activation == "softmax":
            dZ = dA.copy() #skipped dY/dZ computation above
            dA_prev, dW, db = self.affine_backward(dZ, cacheAWb)
        return dA_prev, dW, db

    
    def back_propagation(self, Y_hat, Y, caches):
        gradients = {}
        L = self.num_of_layers - 1 #number of parameters in layers
        dZ = Y_hat - Y #derivative dE/dz with sofmax and crossentropy combined instead of going for dE/dY * dY/dZ
        # cache = ( (cacheAWb), cacheZ )@ cacheAWb: stores A_prev, W and b @cacheZ: Stores Z of a layer 
        
        #=====1. Output layer gradients using "sofmax derivative"
        gradients["dA"+str(L)], gradients["dW"+str(L)], gradients["db"+str(L)] = self.activation_backward(dZ, caches[L-1], activation = "softmax")
        
        #=====2. Hidden layer gradients using "relu derivative"
        for l in reversed(range(L-1)):
            gradients["dA"+str(l + 1)], gradients["dW"+str(l + 1)], gradients["db"+str(l + 1)] = self.activation_backward(gradients["dA"+str(l + 2)], caches[l], activation = "relu")
        
        return gradients
    
    #================= 6. Update the Weigth Matrix
    #update function for weights and bias
    def update_weights(self, gradients):
        for l in range(self.num_of_layers - 1):
            self.velocity["V" + str(l+1)] = ( self.gamma * self.velocity["V" + str(l+1)] ) + ( (1-self.gamma) * gradients["dW" + str(l+1)] )
            self.weights["W" + str(l+1)] -= self.learning_rate * self.velocity["V" + str(l+1)]
            self.weights["b" + str(l+1)] -= self.learning_rate * gradients["db" + str(l+1)]
            
    #================= 7. Train Model
    def train_model(self, train_data, train_label, actual_class, verbose=False, filename="model"):
        print("training...")
        fp = open(filename +"_error_train.txt", "a")
        # train_data:X and train_label:Y
        error_each_epoch = np.zeros(self.epoch)
        for i in range(0, self.epoch):
            Y_hat, caches = self.forward_propagation(train_data)
            error = self.loss_function(Y_hat, train_label)
            percentage_accuracy = np.mean( np.argmax(Y_hat, axis=0) == actual_class )*100
            gradients = self.back_propagation(Y_hat, train_label, caches)
            self.update_weights(gradients) 
            error_each_epoch[i] = error
            if verbose:
#                 clear_output(wait=True)
                print(f"Epoch:{i} | Cross Entropy Error: {error} | Accuacy: {percentage_accuracy}%")
                fp.write(f"Epoch:{i} | Cross Entropy Error: {error} | Accuacy: {percentage_accuracy}%" + '\n')
		        
        #Plot the error
        fp.close()
        self.error_plot(error_each_epoch, filename)
   
    #================= 8. Test Model: It should print Accuracy
    def test_model(self, test_data, test_label, actual_class, filename="model"):
        Y_hat, caches = self.forward_propagation(test_data)
        error = self.loss_function(Y_hat, test_label)
        percentage_accuracy = np.mean( np.argmax(Y_hat, axis=0) == actual_class )*100
        print(f"Accuracy on Test Data: {percentage_accuracy}% | Cross Entropy Error: {error}")
        fp = open(filename +"_error_test.txt", "a")
        fp.write(f"Accuracy on Test Data: {percentage_accuracy}% | Cross Entropy Error: {error}" + '\n')
        fp.close()
        return error, percentage_accuracy


#============================================= END =======================================================

# #===================TESTING CODE===================
# samples = np.array([[2, 3, 1, -1],
#                     [1, 2, -2, 3] ])
# labels = np.array([ [1, 0, 0 ,1],
#                     [0, 1, 1, 0] ])
# actual_labels = np.array([0, 1, 1, 0])
# test = deepNN([2, 1, 1], 0.1, gamma=0.3, epoch=20 )
# # vars(test)
# test.train_model(samples, labels, actual_labels, verbose = True)
# test.forward_propagation(np.array([[-1],[0]]))
# #==================================================

