import numpy as np
import matplotlib.pyplot as plt

# Perceptron class with all the functions
class Perceptron:

    def __init__(self, inputs_num, outputs_num, epoch, learning_rate):
        self.inputs_num = inputs_num
        self.outputs_num = outputs_num
        self.epoch = epoch
        self.learning_rate = learning_rate
        #Weights of input
        self.weights = np.zeros((outputs_num, inputs_num))
        #bias
        self.bias = np.zeros(outputs_num)
    
    # Vector Dot product function
    def dot_product(self, A, B):
        len_a = len(A)
        len_b = len(B)
        if len_a !=len_b:
            print("Error: Vector size mismatch")
            return
        product = 0
        for i in range(len_a):
            product += A[i]*B[i]
        return product

    #activation function 
    def activation(self, summation):
        if summation > 0:
            out = 1
        else:
            out = 0            
        return out
    
    #compute output
    def predict(self, inputs):
        #1. Predict for single output node: binary class classification
        if self.outputs_num == 1:
            summation = self.dot_product(inputs, self.weights[0]) + self.bias[0]
            output = self.activation(summation)
        
        #2. Predict for mulitple output node: Multiclass classfication
        else:
            output = 0
            sum_prev = -1000 #large negative value
            for i in range(self.outputs_num):
                summation = self.dot_product(inputs, self.weights[i]) + self.bias[i]
                # The Node with maximum sum is set to 1
                if i > 0 and summation > sum_prev:
                    output = i   
                sum_prev = summation   

        return output
     
    def test(self, test_data):
        correct_clf = 0 #number of correct classification
        total_clf = len(test_data) #total number of classifications done
        for data in test_data:
            inputs = data[:self.inputs_num]
            label = data[self.inputs_num] 
            prediction = self.predict(inputs)  
            if int(prediction) == int(label):
                correct_clf +=1
        accuracy = (correct_clf/total_clf)*100
        return accuracy

    #Train the weights
    def train(self, training_inputs):
        print("Start of Training...")
        plt.ion()
        for ep in range(self.epoch):
            np.random.shuffle(training_inputs) #suffle the training data
            error = 0
            for data in training_inputs:
                inputs = data[:self.inputs_num]
                try:
                    label = data[self.inputs_num]
                except:
                    print(data)
                prediction = self.predict(inputs)
                #Do nothing if prediction is correct
                if int(label) == int(prediction):
                    continue
                #increment error as the prediction was not good
                error += 1
                #1. Traning for binary classification: Single output
                if self.outputs_num == 1:
                    self.weights[0] += self.learning_rate * (label - prediction) * inputs
                    self.bias[0] += self.learning_rate * (label - prediction)
                
                #2. Training for multi class classification: multiple outputs
                else:
                    #Change weigths for each output node in case of missclassification    
                    #add weigths to that node which ought to be correct
                    self.weights[int(label)] += self.learning_rate * inputs
                    self.bias[int(label)] += self.learning_rate
                
                    #reduce weights from wrongly predicted nodes
                    self.weights[int(prediction)] -= self.learning_rate * inputs
                    self.bias[int(prediction)] -= self.learning_rate
            #compute percentage error in each epoch
            percentage_error = error/len(training_inputs)
            #print(f"Epoch{ep}: {percentage_error}")
            #plot percentage error for each epoch
#=======PLOT Section: Comment this for error in plotting
            plt.xlim([-5, self.epoch])
            plt.scatter(ep, percentage_error, s=8)
            plt.title("Training Error Plot")
            plt.xlabel("epoch")
            plt.ylabel("Percentage Error(%)")
            plt.pause(0.005)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
#=======PLOT Section: Ends
        print("End of Training")

# In[1]:


# Function to Generate the two dimensional dataset with two or three classes
def generate_dataset(num_of_samples, num_of_class):
    """
    The function generated a linearly separated 2D Dataset with two classes.
    num_of_samples: number of samples in each classes
    mean: Mean values for the distribution
    cov: Covariable for the normal distribution
    
    Functions used:
    np.random.multivariabe_normal() function is used to generate the data.
    """
    
    #======Generate: Data for the Class 0
    mean = [5, 0] 
    cov = [[10, 0], [0, 5]]  # diagonal covariance
    data1 = np.random.multivariate_normal(mean, cov, num_of_samples) #generate class 0 data
    data1 = np.append(data1, [[0]]*len(data1), axis=1) #add class label to the data

    #======Generate:Data for the Class 1
    mean = [40, 30]
    cov = [[5, 0], [0, 10]]  # diagonal covariance
    data2 = np.random.multivariate_normal(mean, cov, num_of_samples)
    data2 = np.append(data2, [[1]]*len(data2), axis=1)
    
    if num_of_class == 3:
        #======Generate:Data for the Class 2
        mean = [40, -30]
        cov = [[7, 0], [0, 10]]  # diagonal covariance
        data3 = np.random.multivariate_normal(mean, cov, num_of_samples)
        data3 = np.append(data3, [[2]]*len(data2), axis=1)

    #======Generate: Mix two classes of data randomly
    data = np.append(data1, data2, axis = 0) #append the data from 2 classes
    if num_of_class == 3:
        data = np.append(data, data3, axis = 0) #append the data from 2 classes
    np.random.shuffle(data) #shuffle the dataset

#======Plot Dataset: Comment this if any error in plotting generated
    print("Plotting the Dataset...")
    plt.ion()
    plt.plot(data1[0:,:1], data1[0:, 1:2], 'x')
    plt.plot(data2[0:,:1], data2[0:, 1:2], '+')
    if num_of_class == 3:
        plt.plot(data3[0:,:1], data3[0:, 1:2], '1')
    plt.legend(['Class 0', 'Class 1', 'Class 3'], loc='upper left', fontsize = 12, prop = {'weight':'bold', 'size':  10})
    plt.title(f"Generated Dataset: Two Dimensions", fontsize = 12, weight='bold')
    plt.xlabel("feature 1", fontsize = 12, weight='bold')
    plt.ylabel("feature 2", fontsize = 12, weight='bold')
#     plt.axis('equal')
    plt.grid(True)
    plt.plot(block=False)
    plt.pause(2)
    plt.close()
#=======Plot Section: Ends   
    return data

#================================================================MAIN Function===================================================
#Main Funtion is to test a simple case of AND function which is linear
#Training AND function using perceptron Test Code
if __name__=="__main__": 
    perceptron = Perceptron(2, 1, 50, 0.01)
    #Create Training Data [input1, input2, label]
    training_inputs = []
    training_inputs.append(np.array([1, 1, 1]))
    training_inputs.append(np.array([1, 0, 0]))
    training_inputs.append(np.array([0, 1, 0]))
    training_inputs.append(np.array([0, 0, 0]))
    training_inputs = np.array(training_inputs)

    #Train the perceptron
    perceptron.train(training_inputs)

    #Test Data
    test_inputs = []
    test_inputs.append(np.array([1, 0, 0]))
    test_inputs.append(np.array([1, 1, 1]))
    test_inputs.append(np.array([0, 1, 0]))
    test_inputs.append(np.array([0, 1, 0]))
    test_inputs.append(np.array([0, 0, 0]))
    test_inputs.append(np.array([1, 0, 0]))
    test_inputs = np.array(test_inputs)

    #Test the perceptron with test data
    accuracy = perceptron.test(test_inputs)
    print("The perceptron library testing: Successfull")
    print("Example: AND function generation")
    print(f"accuracy of model: {accuracy}")

   
