# -*- coding: utf-8 -*-

# Course: COMP-551: Applied Machine Learning
# Instructor: Joelle Pineau
# Assignment 3: Modified Digits
# Team: Andrei Chubarau, Fred Glozman, Li Xue

from NeuralNetwork import NeuralNetwork
import numpy as np
np.random.seed(0)

# The code in this file performs the following operations:
# - Load data from processed_train_x.csv and train_y.csv
# - Instantiate an instance of the NeuralNetwork class found in NeuralNetwork.py.
#   This is the class which contains our implementation of a 
#   Fully connected feedforward neural network.
# - Train the neural network instance on the training data. 
# - Print the accuracies throughout training iterations.
# 
# This file serves as an example which illustrates how we applied 
# our NeuralNetwork class to the Modified Digits problem.

# Files from which and to which data will flow
TRAIN_X_SLIM_FILE = 'processed_train_x_slim.csv'
TRAIN_Y_SLIM_FILE = 'train_y_slim.csv'
TRAIN_X_FILE = 'processed_train_x.csv'
TRAIN_Y_FILE = 'train_y.csv'
TEST_X_FILE = 'test_x.csv'
TEST_Y_FILE = 'test_y.csv'

# Every possible classification for this problem
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

# Load the training data from processed_train_x.csv and train_y.csv
print 'Loading the training set...'
X = np.genfromtxt(TRAIN_X_FILE, delimiter=",", skip_header=0)
X = np.array([[1 if i > 200 else 0 for i in x] for x in X])
y = np.genfromtxt(TRAIN_Y_FILE, delimiter=",", skip_header=0, dtype=np.int8).flatten()
print 'Finished loading the training set'

# Split the training data into a training set and a validation set
trainX = np.array(X[:int(len(X)*0.70)])
trainY = np.array(y[:int(len(y)*0.70)])
validationX = np.array(X[int(len(X)*0.70):])
validationY = np.array(y[int(len(y)*0.70):])

# Instantiate a neural network with 
# 4096 neurons at the input layer
# 40 neurons at the output layer
# 3 hidden layers with 64, 32, and 16 neurons in each
nn = NeuralNetwork(4096, [64, 32, 16], len(CLASSES), CLASSES)

# Train the neural network instantiate above
# Use trainX and trainY to train the network
# Use validationX and validationY to validate the accuracy of the network
# Use a learning rate of 0.0001 when training the network
# Train the network for 10000 iterations
# Training and validation accuracies will be printed every 1000 iterations
nn.train(trainX, trainY, validationX, validationY, 0.0001, 10000)