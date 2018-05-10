# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(0)

# Course: COMP-551: Applied Machine Learning
# Instructor: Joelle Pineau
# Assignment 3: Modified Digits
# Team: Andrei Chubarau, Fred Glozman, Li Xue
#
#
# Fully connected feedforward neural network
# With Sigmoid neurons
# Trained with backpropagation 
#
# Hyper-parameters: 
# - number of neurons in the input layer
# - number of hidden layers
# - number of neurons in each hidden layer
# - number of neurons in the output layer
# - learning rate
# - number of training iterations.
#
#
# REFERENCES
#
# The neural network implemented here is similar to those described in
#
# lecture 14 of COMP-551: Applied Machine Learning  
# found here: http://www.cs.mcgill.ca/~jpineau/comp551/Lectures/14NeuralNets.pdf
# 
# and in the article titled Implementing a Neural Network from Scratch in Python – An Introduction
# found here: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
#
# We used both of the recourses above as a reference and guide while implementing the following class.
# Lecture 14 was used to understand the structure of neural networks and the mathematics behind them.
# The article on the wildml website referenced above served as tutorial which assisted 
# with the technical implementation that follows.
#
# Note that the sources above are also referenced in the written report.
#
class NeuralNetwork:
    
    # Initialize the fully connected feedforward neural network
    # numberOfInputNeurons: number of neurons in the input layer
    # numberOfHiddenLayerNeurons: number of neurons in the hidden layers (example: [numberOfNeuronsInHiddenLayer1, numberOfNeuronsInHiddenLayer2, ...])
    # numberOfOutputNeurons: number of neurons in the output layer    
    # classifications: every possible classification of the input space   
    def __init__(self, numberOfInputNeurons, numberOfHiddenLayerNeurons, numberOfOutputNeurons, classifications):
        self.classifications = classifications
        self.layers = [numberOfInputNeurons] + numberOfHiddenLayerNeurons + [numberOfOutputNeurons]
        self.weights = []
        self.bias = []
        
        for i in range(len(self.layers) - 1):
            numberOfNeuronsInCurrentLayer = self.layers[i]
            numberOfNeuronsInNextLayer = self.layers[i + 1]
        
            # Initialize the weights to a small value. Initialize the bias term weights.
            self.weights.append(np.random.randn(numberOfNeuronsInCurrentLayer, numberOfNeuronsInNextLayer))            
            self.bias.append(np.ones((1, numberOfNeuronsInNextLayer)))        
         
    # Returns the final classification prediction of the neural network for a given input
    # x: input which is fed through the neural network
    # networkOutputs: precomputed outputs of every neuron at every layer (optional)
    def predictFinalOutput(self, x, networkOutputs=None):
        # if x was not already propagated through the network, do it now
        if networkOutputs is None:
            networkOutputs = self.predict(x)
        
        # predicted class is the class represented by the output neuron reporting the highest probability
        predictedClassificationIndex = int(np.argmax(networkOutputs[len(networkOutputs) - 1]))
        predictedClassification = self.classifications[predictedClassificationIndex]
        
        return predictedClassification
    
    # Returns the outputs of every neuron at every layer
    # x: input which is fed through the neural network
    def predict(self, x):
        
        # Propagate the input x throughout the neural network
        outputs = [x] 
        for i in range(len(self.weights)):            
            z = outputs[i].dot(self.weights[i]) + self.bias[i]
            o = 1 / (1 + np.exp(-z))
            outputs.append(o)
                
        # compute the probability of every neuron in the output layer
        outputs[len(outputs) - 1] = outputs[len(outputs) - 1] / np.sum(outputs[len(outputs) - 1], axis=1, keepdims=True)
        
        return outputs
     
    # Trains the neural network using backpropagation
    # trainX: training examples
    # trainY: training labels
    # validationX: validation examples
    # validationY: validation labels
    # learningRate: learning rate
    # epochs: number of times to train the neural network on the training set 
    def train(self, trainX, trainY, validationX, validationY, learningRate, epochs):
        numberOfTrainingExamples = len(trainX)
        
        # Train the network for many iterations
        for epoch in range(epochs):
            
            corrections = [None for i in range(len(self.layers))]
            weightAdjustments = [None for i in range(len(self.weights))]
            biasAdjustments = [None for i in range(len(self.weights))]            
            
    
            # Forward propagate the examples in trainX
            networkOutputs = self.predict(trainX)
            
            # Back propagate the error through the network 
            outputLayerCorrection = networkOutputs[len(networkOutputs) - 1]
            outputLayerCorrection[range(numberOfTrainingExamples), np.array([self.classifications.index(y) for y in trainY])] -= 1                                
            corrections[len(corrections) - 1] = outputLayerCorrection
            
            for j in reversed(range(len(self.layers) - 1)):    
                nextLayerCorrections = corrections[j + 1]
                biasAdjustments[j] = (-learningRate) * (np.sum(nextLayerCorrections, axis=0, keepdims=True))
                corrections[j] = (1 - networkOutputs[j]) * networkOutputs[j] * nextLayerCorrections.dot(self.weights[j].T)
                weightAdjustments[j] = (-learningRate) * ((networkOutputs[j].T).dot(nextLayerCorrections) + 0.0001 * self.weights[j])
            
            # Perform gradient descent on the weights and the weights of the bias nodes
            # Update the weights of the network
            for j in range(len(self.layers) - 1):
                self.bias[j] += biasAdjustments[j]
                self.weights[j] += weightAdjustments[j]
                
            # Compute the new accuracy of the network on the training set and validation set
            if epoch % (epochs * 0.1) == 0:
                self.printAccuracy(trainX, trainY, validationX, validationY, epoch)
    
    # Prints the training and validation accuracy of the network
    # trainX: training set examples 
    # trainY: training set labels
    # validationX: validation set examples
    # validationY: validation set labels
    def printAccuracy(self, trainX, trainY, validationX, validationY, epoch=''):
        if trainX is not None and trainY is not None:
            print str(epoch) + ' Training Accuracy = ' + str(self.test(trainX, trainY))
        if validationX is not None and validationY is not None:
            print str(epoch) + ' Validation Accuracy = ' + str(self.test(validationX, validationY))
    
    # Outputs the accuracy of the network
    # X: examples to classify
    # y: expected classifications of examples in X
    def test(self, X, y):
        # Correct & incorrect prediction counters
        correctPredictionCount = 0
        incorrectPredictionCount = 0

        # Predict the class of every example, and compare it to the expected classification
        for example, expectedClassification in zip(X, y):

            # Predict the class of the example 
            prediction = self.predictFinalOutput(example)
            
            # If the predicted class matches the expected classification increment correctPredictionCount
            # otherwise, increment incorrectPredictionCount
            if prediction == expectedClassification:
                correctPredictionCount += 1
            else:
                incorrectPredictionCount += 1

        # accuracy = (number of examples classified correctly) / (total number of examples classified)
        accuracy = float(correctPredictionCount) / (incorrectPredictionCount + correctPredictionCount)

        return accuracy        