#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multilayernn import * #Import from own library
import csv #for reading csv directly to numpy


# # IRIS

# ##================================== 1. Data Processing and One Hot Encoding

# In[2]:


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
        
        else: #virginica
            row[4] = '2'
            float_list = list(map(float, row)) 
            data.append(float_list)

data = np.array(data) #convert to numpy
np.random.shuffle(data) #Shuffle the input data


# In[3]:


#separating labels and pixels
all_labels= data[:, -1].copy().astype(int)
all_data= data[:, :-1].copy()
# The characteristics of MNIST data pixels = 784 samples = 42000 classes = 10


# In[4]:


#Convert to onehot encoding
features = 4
samples = len(all_data)
classes = 3
all_data = all_data.T
all_label=np.zeros((classes, samples))
for col in range (samples):
    all_label[all_labels[col],col]=1
#Scaling Down of dataset
all_data = all_data/5 #scaled the data


# In[5]:


#Generate Test and Training Data from the Sample 80% vs 20%
train_data = all_data[:, int(samples*0.2):]
train_label = all_label[:, int(samples*0.2):]
train_actual_label = all_labels[int(samples*0.2):]

test_data = all_data[:, :int(samples*0.2)]
test_label = all_label[:, :int(samples*0.2)]
test_actual_label = all_labels[:int(samples*0.2)]


# ## ==============================2. Training of Model

# Hypermeters:
# 1. Tune the right weights as improper weights will cause exploding outputs
# 2. Tune the learning rate and gamma
# 3. Tune the number of epoch to be trained

# In[6]:


#Create Mulit Layer Network
nodes_per_layer = [4, 8, 3]
iris_nn = deepNN(nodes_per_layer, learning_rate = 0.2, gamma = 0.7, epoch=1000)


# In[7]:


#Train the network
iris_nn.train_model(train_data, train_label, train_actual_label, verbose = True, filename="accuracy/iris/irisdata")


# ## ============================3. Testing of Model

# In[10]:


test_error, test_accuracy = iris_nn.test_model(test_data, test_label, test_actual_label, filename="accuracy/iris/irisdata")


# ## Conclusion:
# Check accuracy folder for all the error and accuracy data.

# <hr>
