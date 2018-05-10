# -*- coding: utf-8 -*-

import numpy as np

from keras.optimizers import SGD

from keras.models import Model

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras import Input
from keras.callbacks import EarlyStopping

import featureExtractor as fe
from data.datasetGenerator import TrioDatasetGenerator

########################################################################################################################

img_rows = img_cols = 224
channels = 3
numClasses = 40
learningRate = 1e-4

########################################################################################################################

# global flag to control training (on/off)
doTrain = False
doEvaluationPostTraining = True # predict on test samples

# load weights from existing file before retraining
doReinforceTrain = True

# stage 1: train on augmented dataset
doTrainStage1 = False

# stage 2: train on the original dataset with noisy background
doTrainStage2 = True

########################################################################################################################
# originally, this NN was trained on a gtx 1050ti with 4GB VRAM for several days
# including sessions of reinforcement learning after the initial training

generatorSamplesPerBatch = 8 # lower for decreased memory usage
trainingBatchCount = 2000 # samples per epoch = generatorFlowBatchSize * trainingBatchCount
validationBatchCount = 200

# stage 1
epochsStage1 = 20 # max number of epochs
patienceEpochsStage1 = 3 # epochs with no improvement until training is stopped

# stage 2
epochsStage2 = 20
patienceEpochsStage2 = 3

########################################################################################################################

def resnet50_model():
    # build the VGG16 network
    input_tensor = Input(shape=(img_rows, img_cols, channels))

    # base_model = applications.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    base_model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor,
                                                pooling=None, classes=numClasses)
    print('Model loaded.')

    base_model_num_layers = len(base_model.layers)

    # build a classifier model to put on top of the convolutional model
    x = base_model.output
    x = Flatten(input_shape=base_model.output.shape)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(numClasses, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x)

    nonTrainableLayers = len(model.layers) - base_model_num_layers - 74

    # set the first N layers to non-trainable (weights will not be updated)
    for layer in model.layers[:-nonTrainableLayers]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)

    modelName = "resNet-cl_{}-in_{}x{}-l_{}-lr_{}-math".format(
        numClasses, img_rows, img_cols, len(model.layers), learningRate)

    return modelName, model

def preprocessingFunction(x, min=200, max=240):
    """
    linearize input given a range. Assumes the original input is in range [0-255].
    Example: remap image from [0, 255] to min=200, max = 245, implies that all inputs below 200
    will become 0 and all inputs above 245 will become 255, inputs between min and max are linearly
    interpolated between 0 and 255.
    :param x: batch of images, dimensions do not affect the result
    :param min: start of the clip range
    :param max: end of the clip range
    :return: processsed batch
    """
    return (255 / (max - min) * (-min + x)).clip(0, 255)

class ResNet50CnnModel():
    def __init__(self):
        modelName, model = resnet50_model()

        modelsPath = "data/models/"
        charactersDataPath = "data/resources/charsDigitsLetters"
        rawDataPath = "data/resources/rawTrainingData"

        if doTrain:
            if doReinforceTrain:
                model.load_weights(modelsPath + modelName)

            ##########################################################################
            # TRAIN STAGE 1: train on randomly generated images with black background
            if doTrainStage1:
                # prepare data augmentation configuration
                trainGenerator = TrioDatasetGenerator(
                    charactersDataPath + "/training",
                    imgDim=img_cols,
                    channels=channels,
                    getMathOutputs=True,
                    batchSize=generatorSamplesPerBatch
                )

                validationDatagenRaw = ImageDataGenerator(
                    rescale=1./255.,
                    preprocessing_function=preprocessingFunction
                )

                validationGenerator = validationDatagenRaw.flow_from_directory(
                    rawDataPath + "/validation",
                    target_size=(img_rows, img_cols),
                    batch_size=generatorSamplesPerBatch,
                    shuffle=True
                )

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=patienceEpochsStage1, verbose=1),
                ]

                # fine-tune the model
                model.fit_generator(
                    trainGenerator,
                    shuffle=True,
                    steps_per_epoch=trainingBatchCount,
                    epochs=epochsStage1,
                    validation_data=validationGenerator,
                    validation_steps=validationBatchCount,
                    max_queue_size=1,
                    callbacks=callbacks,
                )

                model.save_weights(modelsPath + modelName)

            ##########################################################################
            # TRAIN STAGE 2: train on the original images with random noise background

            if doTrainStage2:
                trainDatagenRaw = ImageDataGenerator(
                    rescale=1./255.,
                    zoom_range=0.1,
                    shear_range=0.1,
                    rotation_range=10,
                    preprocessing_function=preprocessingFunction
                )

                validationDatagenRaw = ImageDataGenerator(
                    rescale=1./255.,
                    preprocessing_function=preprocessingFunction
                )

                trainGenerator = trainDatagenRaw.flow_from_directory(
                    rawDataPath + "/training",
                    target_size=(img_rows, img_cols),
                    batch_size=generatorSamplesPerBatch,
                    shuffle=True
                )

                validationGenerator = validationDatagenRaw.flow_from_directory(
                    rawDataPath + "/validation",
                    target_size=(img_rows, img_cols),
                    batch_size=generatorSamplesPerBatch,
                    shuffle=True
                )

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=patienceEpochsStage2, verbose=1),
                ]

                # fine-tune the model
                model.fit_generator(
                    trainGenerator,
                    shuffle=True,
                    steps_per_epoch=trainingBatchCount,
                    epochs=epochsStage2,
                    validation_data=validationGenerator,
                    validation_steps=validationBatchCount,
                    max_queue_size=1,
                    callbacks=callbacks
                )

            model.save_weights(modelsPath + modelName)
        else:
            model.load_weights(modelsPath + modelName)

        self.modelName, self.model = modelName, model

    def predict(self, x):
        return self.model.predict(x)

if __name__ == '__main__':
    model = ResNet50CnnModel().model

    if doEvaluationPostTraining:
        totalSamples = 10000
        samplesPBatch = 50
        batchCount = int(totalSamples / samplesPBatch) # must divide without remainder

        if totalSamples % samplesPBatch != 0:
            print("Warning: totalSamples ({}) is not divisible by samplesPBatch ({}) without remainder.".
                  format(totalSamples, samplesPBatch))

        testDatagen = ImageDataGenerator(
            rescale=1./255.,
            preprocessing_function=preprocessingFunction,
        )

        testGenerator = testDatagen.flow_from_directory(
            "data/resources/rawTestingData",
            target_size=(img_rows, img_cols),
            batch_size=samplesPBatch,
            class_mode=None,
            shuffle=False
        )

        # print(model.evaluate_generator(testGenerator, batchCount))

        results = model.predict_generator(testGenerator, batchCount, verbose=1)
        results = [fe.mathValuesSorted[index] for index in np.argmax(results, axis=1)]

        with open('output.csv', 'w') as yFile:
            for index, value in enumerate(results):
                yFile.write("{},{}".format(index + 1, value) + "\n")
