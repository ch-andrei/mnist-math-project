import numpy as np

np.random.seed(1)

class NeuralNetwork:

    # Initialize the fully connected feed-forward neural network
    # numberOfInputNeurons: number of neurons at the input layer
    # numberOfHiddenLayerNeurons: number of neurons in each hidden layer
    # numberOfOutputNeurons: number of neurons at the output layer
    def __init__(self, numberOfInputNeurons, numberOfHiddenLayerNeurons, numberOfOutputNeurons):
        print 'Initializing the fully connected feed-forward neural network...'

        self.layerSizes = [numberOfInputNeurons] + numberOfHiddenLayerNeurons + [numberOfOutputNeurons]
        self.numberOfLayers = len(self.layerSizes)
        self.numberOfInputNeurons = numberOfInputNeurons
        self.numberOfOutputNeurons = numberOfOutputNeurons
        self.layers = []

        # initialize weights
        for layerIndex in range(self.numberOfLayers - 1):
            currentLayerSize = self.layerSizes[layerIndex]
            nextLayerSize = self.layerSizes[layerIndex + 1]
            weights = np.random.random((currentLayerSize, nextLayerSize))
            self.layers.append(weights)

    # Applies the Sigmoid function to the input
    # z: input to which we apply the Sigmoid function
    # Output: 1 / (1 + e^(-z))
    def sigmoid(self, z):
        result = np.power((1 + np.exp(-z)), -1)
        return result

    # Classifies the data-sets in X
    # X: data-sets to classify
    # resultProcessor: function applied to every output unit
    # Output: output of the network
    def predictOutput(self, X, resultProcessor):
        layerOutputs = self.predict(X)
        outputLayer = layerOutputs[len(layerOutputs) - 1]
        predictedOutput = [resultProcessor(x) for x in outputLayer]
        return predictedOutput

    # Classifies the data-sets in X
    # X: data-sets to classify
    # Output: output of the network at each neuron in each layer
    def predict(self, X):
        layerOutputs = [X]

        for layerIndex in range(1, self.numberOfLayers):
            previousLayerOutputs = layerOutputs[layerIndex - 1]
            previousLayerWeights = self.layers[layerIndex - 1]
            linearCombination = previousLayerOutputs.dot(previousLayerWeights)
            outputs = self.sigmoid(linearCombination)
            layerOutputs.append(outputs)

        return layerOutputs

    # Trains the NN using back-propagation
    # inputs: input data-sets
    # expectedOutputs: expected outputs
    # numberOfEpochs: number of training iterations
    # learningRate: weight adjustment factor
    def train(self, inputs, expectedOutputs, numberOfEpochs, learningRate, printError=True):
        print 'Training...'

        for epoch in range(numberOfEpochs):

            # Step 1: Forward Propagation
            # Pass the data-sets in X through the network
            layerOutputs = self.predict(inputs)
            networkOutput = layerOutputs[len(layerOutputs) - 1]

            # Step 2: Back Propagation
            # Back propagate the error with respect to the expected output in y
            # (y - O)
            outputError = expectedOutputs - networkOutput
            if epoch % (numberOfEpochs*0.1) == 0 and printError:
                error = np.mean(np.abs(outputError))
                print 'Error: ' + str(error)

            corrections = [[] for i in range(self.numberOfLayers)]

            outputCorrections = networkOutput * (1 - networkOutput) * outputError
            corrections[len(corrections) - 1] = outputCorrections

            for layerIndex in reversed(range(1, self.numberOfLayers - 1)):
                layerOutput = layerOutputs[layerIndex]
                correction = layerOutput * (1 - layerOutput) * corrections[layerIndex + 1].dot(self.layers[layerIndex].T)
                corrections[layerIndex] = correction

            # Step 3: Gradient Descent
            # Update the weights of the network
            for layerIndex in range(self.numberOfLayers - 1):
                weightAdjustment = learningRate * layerOutputs[layerIndex].T.dot(corrections[layerIndex + 1])
                self.layers[layerIndex] += weightAdjustment
