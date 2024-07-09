#
#   Example 7: Centralized HFL (horizontal FL) using the neural network model and MNIST dataset
#   Author: student Marko Nikolovski made this example within his B.Sc. thesis (at UNI Novi Sad, May 2024, mentor Miroslav Popovic)
#
#   Description:
#   Read the description of this example in the MPT-FLA paper, see link in README.md
#   We use numpy for python (PC version) and ulab for Micropython (Pico version)
#
#   Run this example (after: cd src/examples2) for noNodes==3: 
#       launch mp_async_example7_NN_MNIST.py 3 id 0 192.168.160.12
#

import numpy as np
#import ulab
#from ulab import numpy as np
import sys
import random
import asyncio
import gc
    
from ptbfla_pkg.mp_async_ptbfla import *

def one_hot(Y):
        
    y_max = int(np.max(Y)) + 1
    y_size = len(Y[0])
    one_hot_Y = np.zeros((y_size, y_max))
    for i in range(y_size):
        for j in range(y_max):
            if j == Y[0, i]:
                one_hot_Y[i, j] = 1
    return one_hot_Y.T

def sigmoid(z):
    # Sigmoid activation function (returns a number between 0 and 1)
    return 1 / (1 + np.exp(-z))

def define_neurons(X, Y):
    Y = one_hot(Y) 
    input_layer_neurons = X.shape[0] # size of input layer
    hidden_layer_neurons = 13 # hidden layer of size 10
    ouput_layer_neurons = Y.shape[0] # size of output layer
    return (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)   

def initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons):

    # Initialize the weights and biases for the nodes in the hidden layer
    weights_hidden = [[random.uniform(-1, 1) * 0.1 for x in range(input_layer_neurons)] for x in range(hidden_layer_neurons)]
    biases_hidden = [[0.0] * 1 for x in range(hidden_layer_neurons)] # Zeroed biases

    # Initialize the weights and biases for the nodes in the output layer
    weights_output = [[random.uniform(-1, 1) * 0.1 for x in range(hidden_layer_neurons)] for x in range(ouput_layer_neurons)]
    biases_output = [[0.0] * 1 for x in range(ouput_layer_neurons)]

    parameters = (weights_hidden, biases_hidden, weights_output, biases_output)

    return parameters

def forward_propagation(X, parameters):

    weights_hidden = parameters[0]
    biases_hidden = parameters[1]
    weights_output = parameters[2]
    biases_output = parameters[3]

    # Raw output of hidden layer 
    unactivated_hidden = np.dot(weights_hidden, X) + biases_hidden

    # Tanh activated output of hidden layer
    activated_hidden = np.tanh(unactivated_hidden)
    
    # Raw output of output layer
    unactivated_output = np.dot(weights_output, activated_hidden) + biases_output

    # Sigmoid activated output of output layer
    activated_output = sigmoid(unactivated_output)

    outputs = {"unactivated_hidden": unactivated_hidden,
               "activated_hidden": activated_hidden,
               "unactivated_output": unactivated_output,
               "activated_output": activated_output}
    
    return activated_output, outputs

def backward_propagation(parameters, outputs, X, Y):
    
    m = X.shape[1]
    Y = one_hot(Y)

    # Extract the weights, biases and neuron outputs
    activated_hidden = outputs["activated_hidden"]
    activated_output = outputs["activated_output"]
    
    weights_hidden = parameters[0]  
    weights_output = parameters[2]
   
    # Backprop calculations 
    dunactivated_output = activated_output - Y
    dweights_output = (1 / m) * np.dot(dunactivated_output, activated_hidden.T)
        
    dbiases_output = (1 / m) * np.sum(dunactivated_output, axis=1)
    dbiases_output = [[val] for val in dbiases_output]
    dbiases_output = np.array(dbiases_output)
        
    dunactivated_hidden = np.dot(weights_output.T, dunactivated_output)
    one_minus_pow_activated_hidden = 1 - (activated_hidden ** 2)    
    dunactivated_hidden = dunactivated_hidden * one_minus_pow_activated_hidden
    
    dweights_hidden = (1 / m) * np.dot(dunactivated_hidden, X.T)
    
    dbiases_hidden = (1 / m) * np.sum(dunactivated_hidden, axis=1)
    dbiases_hidden = [[val] for val in dbiases_hidden]
    dbiases_hidden = np.array(dbiases_hidden)
    
    gradients = {"dweights_hidden": dweights_hidden, "dbiases_hidden": dbiases_hidden, "dweights_output": dweights_output,"dbiases_output": dbiases_output}
    
    return gradients

def gradient_descent(parameters, gradients):

    LEARNING_RATE = 0.07
    
    weights_hidden = parameters[0]
    biases_hidden = parameters[1]
    weights_output = parameters[2]
    biases_output = parameters[3]
   
    # Extract gradients to apply 
    dweights_hidden = gradients["dweights_hidden"]
    dbiases_hidden = gradients["dbiases_hidden"]
    dweights_output = gradients["dweights_output"]
    dbiases_output = gradients["dbiases_output"]

    # Apply gradients to weights and biases
    weights_hidden = weights_hidden - LEARNING_RATE * dweights_hidden
    biases_hidden = biases_hidden - LEARNING_RATE * dbiases_hidden
    weights_output = weights_output - LEARNING_RATE * dweights_output
    biases_output = biases_output - LEARNING_RATE * dbiases_output
    
    parameters = (weights_hidden, biases_hidden, weights_output, biases_output)
    
    return parameters

def train(X, Y, parameters, num_iterations=300):

    print("\nTraining begins...") 
    print("Total number of iterations is " + str(num_iterations))
    parameters = ListToNympy(parameters)

    for i in range(0, num_iterations):
        # Get output of the model with inputs and current parameters
        activated_output, outputs = forward_propagation(X, parameters)
        
        # Calculate new gradients 
        gradients = backward_propagation(parameters, outputs, X, Y)
        
        # Apply gradients and calculate new parameters
        parameters = gradient_descent(parameters, gradients)
        
        if i%100 == 0:
            print(str(i) + "/" + str(num_iterations))

    print(str(num_iterations) + "/" + str(num_iterations))
    print("Traning complete\n")

    return NumpyToList(parameters)

def prediction(parameters, X_test, Y_test):

    parameters = ListToNympy(parameters)
    
    # Obtain output of model given the test data 
    activated_output, outputs = forward_propagation(X_test, parameters)
    
    correct_predictions = np.sum(np.argmax(activated_output, 0) == Y_test) # Correct predictions where the predicted digit equals the actual digit
    total_digits = Y_test.size # Total amount of test digits

    print("Correct Predictions: " + str(correct_predictions)) 
    print("Total Digits Tested: " + str(total_digits))
    print("Accuracy: " + str(round((correct_predictions / total_digits) * 100, 1)) + "%\n")
        
def read_data(filename, start_line, end_line):

    data = []    
    with open(filename, "rb") as file:
        for i, line in enumerate(file):
            if i >= start_line and i < end_line:
                if line.strip():
                    parts = line.strip().split(b',')
                    int_parts = [int(part) for part in parts]
                    data.append(int_parts)
            elif i >= end_line:
                break
    return data
   
def NumpyToList(numpy_parameters):

    list_parameters = []
    for parametar in numpy_parameters:
        list_parameters.append(parametar.tolist())
    return list_parameters

def ListToNympy(list_parameters):

    numpy_parameters = []
    for parametar in list_parameters:
        numpy_parameters.append(np.array(parametar))
    return numpy_parameters

def fl_cent_client_processing(parameters, privateData, srv_model):

    X_train = privateData[0]
    y_train = privateData[1]
    parameters = train(X_train, y_train, srv_model)
    return parameters
    
def fl_cent_server_processing(privateData, models):
    
    weights_hidden_mean = 0
    biases_hidden_mean = 0
    weights_output_mean = 0
    biases_output_mean = 0
    
    num_clients = len(models)
    
    for model in models:
        model = ListToNympy(model)
        weights_hidden_mean += model[0]
        biases_hidden_mean += model[1]
        weights_output_mean += model[2]
        biases_output_mean += model[3]
    
    parameters = (weights_hidden_mean / num_clients, biases_hidden_mean / num_clients, weights_output_mean / num_clients, biases_output_mean / num_clients)
    
    return NumpyToList(parameters)
          
async def main():

    # Parse command line arguments
    if len(sys.argv) != 5:
        # Args:
        #   noNodes - number of nodes, nodeId - id of a node, flSrvId - id of the FL server, masterIpAdr - master's IP address
        print('Program usage: python mp_async_example7_NN_MNIST.py noNodes nodeId flSrvId masterIpAdr')
        print('Example: noNodes==3, nodeId=0..2, flSrvId==0, masterIpAdr, i.e. 3 nodes (id=0,1,2), server is node 0:')
        print('python mp_async_example7_NN_MNIST.py 3 0 0 192.168.160.12, python mp_async_example7_NN_MNIST.py 3 1 0 192.168.160.12,\n',
              'python mp_async_example7_NN_MNIST.py 3 2 0 192.168.160.12')
        exit()
    
    # Process command line arguments
    noNodes = int( sys.argv[1] )
    nodeId = int( sys.argv[2] )
    flSrvId = int( sys.argv[3] )
    masterIpAdr = sys.argv[4]
    print(noNodes, nodeId, flSrvId, masterIpAdr)
    
    random.seed(3)
    
    if nodeId == 1:
        train_data = read_data("train_micro.txt", 0, 50)
    elif nodeId == 2:
        train_data = read_data("train_micro.txt", 50, 100)
    else:
        train_data = read_data("train_micro.txt", 140, 1140)
          
    Y_train = [row[0] for row in train_data]
    X_train = [[pixel / 255 for pixel in row[1:]] for row in train_data]
    Y_train = [Y_train]

    del train_data
    gc.collect()
    
    Y_train = np.array(Y_train)
    X_train = np.array(X_train).T

    test_data = read_data("train_micro.txt", 100, 140)
    
    Y_test = [row[0] for row in test_data]
    X_test = [[pixel / 255 for pixel in row[1:]] for row in test_data]
    Y_test = [Y_test]

    del test_data
    gc.collect()

    Y_test = np.array(Y_test)
    X_test = np.array(X_test).T
    
    # Start-up: create PtbFla object
    ptb = PtbFla(noNodes, nodeId, flSrvId, masterIpAdr)
    await ptb.start()
    
    # Set private data (training data for the clients and None for the server)
    if nodeId != flSrvId:
        print("the final localData =", nodeId)
        pData = [X_train, Y_train]
        lData = None
    else:
        (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons) = define_neurons(X_train, Y_train)
        start = initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)  
        pData = None
        lData = train(X_train, Y_train, start, 1000)
        prediction(lData, X_test, Y_test)
           
    # Call fl_centralized with noIterations = 1
    model = await ptb.fl_centralized(fl_cent_server_processing, fl_cent_client_processing, lData, pData, 1)
    
    prediction(model, X_test, Y_test)   
        
    del ptb
    pkey = input("press any key to continue...")

def run_main():
    #asyncio.run(main()) # this riases RuntimeError: Event loop is closed
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main()) # this reports "Task was destroyed but it is pending!" two times

if __name__ == '__main__':
    run_main()