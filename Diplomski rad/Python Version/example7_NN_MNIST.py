#
#   Example 7: Centralized HFL (horizontal FL) using the neural network model and MNIST dataset
#   Author: student Marko Nikolovski made this example within his B.Sc. thesis (at UNI Novi Sad, May 2024, mentor Miroslav Popovic)
#
#   Description:
#       The example was made using the 4-phases development paradigm analogously to example4.
#       The row sequential code is here:
#       https://github.com/SohamP2812/MNIST-Neural-Network-from-Scratch/blob/main/MNIST_NN.py
#
#   Run this example (after: cd src/examples2) for noNodes==3: launch example7_NN_MNIST.py 3 id 2
#   Run this example (after: cd src/examples2) for noNodes==4: launch example7_NN_MNIST.py 4 id 3
#
#   Below are the supplementary functions and the 4 main functions corresponding to the
#   development phases of this example:
#       1. seq_base_case() - Sequential base case
#       2. seq_horizontal_federated() - Sequential HFL:
#       3. seq_horizontal_federated_with_callbacks() - Sequential HFL with callbacks
#       4. main() - PTB-FLA code with the same callbacks as in the pase 3.
#

import sys
import numpy as np

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Initialize an array with dimensions of Amount of Total Labels by Amount of Distinct Labels (10 digits in this case) 
    one_hot_Y[np.arange(Y.size), Y] = 1 # For each row in the zeroed array, set the value at the corresponding digit as an index to 1 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def sigmoid(z):
    # Sigmoid activation function (returns a number between 0 and 1)
    return 1 / (1 + np.exp(-z))

def define_neurons(X, Y):
    Y = one_hot(Y)  
    input_layer_neurons = X.shape[0] 
    hidden_layer_neurons = 128 
    ouput_layer_neurons = Y.shape[0] 
    return (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)   

def initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons):

    weights_hidden = [[random.uniform(-1, 1) * 0.1 for x in range(input_layer_neurons)] for x in range(hidden_layer_neurons)]
    biases_hidden = [[0.0] * 1 for x in range(hidden_layer_neurons)]

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


    m = Y.shape[1] 
    Y = one_hot(Y)

    # Calculate Costs
    logs = np.multiply(np.log(activated_output), Y) + np.multiply((1 - Y), np.log(1 - activated_output))
    cost = - np.sum(logs) / m
    cost = float(np.squeeze(cost))
                                    
    return cost

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
    dbiases_output = (1 / m) * np.sum(dunactivated_output, axis=1, keepdims=True)
    dunactivated_hidden = np.multiply(np.dot(weights_output.T, dunactivated_output), 1 - np.power(activated_hidden, 2))
    dweights_hidden = (1 / m) * np.dot(dunactivated_hidden, X.T) 
    dbiases_hidden = (1 / m) * np.sum(dunactivated_hidden, axis=1, keepdims=True)
    
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

def train(X, Y, parameters, num_iterations=100):

    print("\nTraining begins...") 
    print("Total number of iterations is " + str(num_iterations))
    parameters = ListToNympy(parameters)
    
    for i in range(0, num_iterations):
    
        activated_output, outputs = forward_propagation(X, parameters)
        
        gradients = backward_propagation(parameters, outputs, X, Y)
     
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
    print("Accuracy: " + str(np.round((correct_predictions / total_digits) * 100, 1)) + "%\n")
        
def data_split(filename, num_clients):

    with open(filename, "r") as file:
        # Pročitajte linije iz fajla
        lines = file.readlines()

    # Inicijalizujte praznu listu za podatke
    data = []

    for line in lines:
        # Podelite liniju na delove koristeći zarez kao separator
        parts = line.strip().split(",")
        # Konvertujte svaki deo u float vrednost
        int_parts = [int(part) for part in parts]
        # Dodajte konvertovane delove u listu podataka
        data.append(int_parts)

    #fja koja deli data na test i train i kasnije train deli num_clients broj delova

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) 

    test_data = data[0:1000].T
    Y_test = test_data[0] # Testing Labels
    X_test = test_data[1:n]
    X_test = X_test / 255
    
    Y_test = Y_test.T
    Y_test = np.array([Y_test])
    
    train_data = data[1000:m]
    train = np.array_split(train_data, num_clients)
    
    X_train_list = []
    Y_train_list = []
    
    for x in train:
        x = x.T
        
        Y_data = x[0]
        Y_data = Y_data.T 
        Y_data = np.array([Y_data])
        
        X_data = x[1:]
        X_data = X_data / 255
        
        X_train_list.append(X_data)
        Y_train_list.append(Y_data)
          
    train_test = (X_train_list, Y_train_list, X_test, Y_test)

    return train_test

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

def fl_cent_client_processing(parameters, privateData, msgsrv):

    X_train = privateData[0]
    y_train = privateData[1]
    parameters = train(X_train, y_train, parameters)
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

def seq_base_case():
    
    np.random.seed(3)

    # Read MNIST Dataset in from CSV file
    print("Reading CSV Data...")
    with open("train_mali.csv", "r") as file:
        # Pročitajte linije iz fajla
        lines = file.readlines()

    # Inicijalizujte praznu listu za podatke
    data = []

    for line in lines:
        # Podelite liniju na delove koristeći zarez kao separator
        parts = line.strip().split(",")
        # Konvertujte svaki deo u float vrednost
        int_parts = [int(part) for part in parts]
        # Dodajte konvertovane delove u listu podataka
        data.append(int_parts)

    # Convert data to a numpy array
    print("Converting Data...")
    data = np.array(data)

    # Get dimensions of input data
    m, n = data.shape

    # Randomize sequencing of input data
    np.random.shuffle(data) 

    print("Splitting Data in Training and Testing...")
    # Isolate first 500 digits for testing/validation

    test_data = data[0:1000].T
    Y_test = test_data[0] # Testing Labels
    X_test = test_data[1:n] # Testing Features
    X_test = X_test / 255 # Normalize pixel data

    ## Isolate rest of digits for training
    train_data = data[1000:m]
    train_data = train_data.T

    Y_train = train_data[0] # Training Labels
    X_train = train_data[1:n] # Training Features
    X_train = X_train / 255 # Normalize pixel data

    # Transpose Labels for compatibility with NN
    Y_train = Y_train.T
    Y_test = Y_test.T

    # Convert Labels to Numpy Array
    Y_train = np.array([Y_train])
    Y_test = np.array([Y_test])

    # Train the model and save the weights and biases
    print("Building Model...")
    (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons) = define_neurons(X_train, Y_train)

    # Initialize weights and biases for the amount of neurons specified previously 
    parameters = initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)
        
    parameters = train(X_train, y_train, parameters)
    
    prediction(parameters, X_test, Y_test)
           
def seq_horizontal_federated(num_clients):

    models = []
    weights_hidden_mean = 0
    biases_hidden_mean = 0
    weights_output_mean = 0
    biases_output_mean = 0

    np.random.seed(3)
    
    print("------------- Seq horizontal federated -------------")

    X_train_list, Y_train_list, X_test, Y_test = data_split("train_small.csv", num_clients)

    (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons) = define_neurons(X_train_list[0], Y_train_list[0])
    start_parameters = initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)    
    
    for n in range(num_clients):
        model = train(X_train_list[n], Y_train_list[n], start_parameters)        
        models.append(ListToNympy(model))
        
    for n in range(num_clients):
        weights_hidden_mean += models[n][0]
        biases_hidden_mean += models[n][1]
        weights_output_mean += models[n][2]
        biases_output_mean += models[n][3]

    parameters = (weights_hidden_mean / num_clients, biases_hidden_mean / num_clients, weights_output_mean / num_clients, biases_output_mean / num_clients)

    for n in range(num_clients):
        print("MODEL "+ str(n+1))
        prediction(models[n], X_test, Y_test)

    print("MODEL MEAN")
    prediction(parameters, X_test, Y_test)
    
    return parameters

def seq_horizontal_federated_with_callbacks(num_clients):
    
    np.random.seed(3)
    
    print("------------- Seq horizontal federated with callbacks -------------")

    X_train_list, Y_train_list, X_test, Y_test = data_split("train_small.csv", num_clients)
    
    (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons) = define_neurons(X_train_list[0], Y_train_list[0])

    msgsrv = []
    models = []
    start_parameters = initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)
        
    for n in range(num_clients):        
        model = fl_cent_client_processing(start_parameters, [X_train_list[n], Y_train_list[n]], msgsrv)
        models.append(model)

    avg_model = fl_cent_server_processing(None, models)
    
    parameters = seq_horizontal_federated(num_clients)
    
    assert np.array_equal(parameters[0], avg_model[0]) and \
           np.array_equal(parameters[1], avg_model[1]) and \
           np.array_equal(parameters[2], avg_model[2]) and \
           np.array_equal(parameters[3], avg_model[3]), "parameters and avg_model must be equal!"
           

from ptbfla_pkg.ptbfla import *

# Run this example (after: cd src/examples2) for noNodes==3: launch example7_NN_MNIST.py 3 id 2
# Run this example (after: cd src/examples2) for noNodes==4: launch example7_NN_MNIST.py 4 id 3
def main():

    # Parse command line arguments
    if len(sys.argv) != 4:
        # Args: noNodes nodeId flSrvId
        #   noNodes - number of nodes, nodeId - id of a node, flSrvId - id of the FL server
        print("Program usage: python example7_NN_MNIST.py noNodes nodeId flSrvId")
        print("Example: noNodes==3, nodeId=0..2, flSrvId==2, i.e. 3 nodes (id=0,1,2), server is node 2:")
        print("python example7_NN_MNIST.py 3 0 2",
              "\npython example7_NN_MNIST.py 3 1 2\npython example7_NN_MNIST.py 3 2 2")
        exit()
    
    # Process command line arguments
    noNodes = int( sys.argv[1] )
    nodeId = int( sys.argv[2] )
    flSrvId = int( sys.argv[3] )
    print(noNodes, nodeId, flSrvId)
    
    np.random.seed(3)
    

    X_train_list, Y_train_list, X_test, Y_test = data_split("train_small.csv", noNodes-1)
    
    # Start-up: create PtbFla object
    ptb = PtbFla(noNodes, nodeId, flSrvId)
    
    # Set local data    
    (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons) = define_neurons(X_train_list[0], Y_train_list[0])
    lData = initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)
  
    # Set private data (training data for the clients and None for the server)
    if nodeId != flSrvId:
        print("the final localData =", nodeId)
        pData = [X_train_list[nodeId], Y_train_list[nodeId]]
    else:
        pData = None
    
    # Call fl_centralized with noIterations = 1
    model = ptb.fl_centralized(fl_cent_server_processing, fl_cent_client_processing, lData, pData, 1)
    
    prediction(model, X_test, Y_test) # Get all the predictions by the model given the test data set   
            
    # Must be
    if nodeId == flSrvId:
            
        parameters = seq_horizontal_federated(noNodes-1)
        
        assert np.array_equal(parameters[0], model[0]) and \
               np.array_equal(parameters[1], model[1]) and \
               np.array_equal(parameters[2], model[2]) and \
               np.array_equal(parameters[3], model[3]), "parameters and avg_model must be equal!"
    # Shutdown
    del ptb
    pkey = input("press any key to continue...")


if __name__ == "__main__":
    #seq_base_case()
    #seq_horizontal_federated(2)
    #seq_horizontal_federated_with_callbacks(2)
    main()