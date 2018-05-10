
import numpy as np
import cv2
import tools

from keras.optimizers import Adadelta
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import Input
from keras.optimizers import SGD

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

import featureExtractor as fe
from data.datasetGenerator import TrioDatasetGenerator

########################################################################################################################

# train = True
train = False

generatorFlowBatchSize = 32

def getDenseModel(learningRate):
    numClasses = 36
    configuration = (numClasses*4, numClasses*8, numClasses*4)

    configurationStr = ""
    for i in configuration:
        configurationStr += i + "-"
    configurationStr = configurationStr[-1]

    img_input = Input(shape=(12, 3))
    x = img_input
    x = Flatten()(x)
    for layer in configuration:
        x = Dense(layer, activation='relu')(x)
        x = Dropout(0.5)(x)
    x = Dense(numClasses, activation='softmax')(x)

    model = Model(img_input, x, name='customCnn')

    # compile the model with a SGD/momentum optimizer
    sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    modelName = "denseModel-{}_{}-lr_{}".format(configurationStr, len(model.layers), learningRate)

    return modelName, model

def getCnnModel(input_shape, learningRate, numClasses):
    img_rows, img_cols, channels = input_shape
    img_input = Input(shape=input_shape)

    x = img_input
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(numClasses, activation='softmax')(x)

    model = Model(img_input, x, name='customCnn')

    # compile the model with a SGD/momentum optimizer
    sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    modelName = "customModel-cl_{}-in_{}x{}-l_{}-lr_{}".format(
        numClasses, img_rows, img_cols, len(model.layers), learningRate)

    return modelName, model

def trainCnn(model, trainGenerator, validationGenerator,
             trainingBatchCount, validationBatchCount,
             epochs, patienceEpochs, modelWeightsPath
             ):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patienceEpochs, verbose=1),
    ]

    # fine-tune the model
    model.fit_generator(
        trainGenerator,
        shuffle=True,
        steps_per_epoch=trainingBatchCount,
        epochs=epochs,
        validation_data=validationGenerator,
        validation_steps=validationBatchCount,
        max_queue_size=1,
        callbacks=callbacks
    )

    model.save_weights(modelWeightsPath)

class Cnn28x28CharacterClassifier:
    def __init__(self, modelsPath, removeDropout=False, train=train):
        # config
        self.dim = 28
        self.inputShape = (self.dim, self.dim, 3)
        self.numClasses = 12
        self.learningRate = 5e-4
        self.maxTrainEpochs = 200
        self.patienceEpochs = 3
        self.trainingBatchCount = 10000
        self.validationBatchCount = 1000
        # config end

        modelName, model = getCnnModel(self.inputShape, self.learningRate, self.numClasses)

        modelNameSuffix = "CharClassifier"
        modelWeightsPath = modelsPath + "/" + "{}-{}".format(modelName, modelNameSuffix)
        trainDataPath = "data/resources/chars"

        if train:
            # prepare data augmentation configuration
            trainDatagen = ImageDataGenerator(
                rotation_range=60,
                shear_range=0.1,
                zoom_range=0.1)

            validationDatagen = ImageDataGenerator(
                rotation_range=60,
                shear_range=0.1,
                zoom_range=0.1
            )

            trainGenerator = trainDatagen.flow_from_directory(
                trainDataPath + "/training",
                target_size=(self.dim, self.dim),
                batch_size=generatorFlowBatchSize
            )

            validationGenerator = validationDatagen.flow_from_directory(
                trainDataPath + "/validation",
                target_size=(self.dim, self.dim),
                batch_size=generatorFlowBatchSize
            )

            trainCnn(model, trainGenerator, validationGenerator,
                     self.trainingBatchCount, self.validationBatchCount,
                     self.maxTrainEpochs, self.patienceEpochs, modelWeightsPath)

        model.load_weights(modelWeightsPath)

        # doesnt do anything
        if removeDropout:
            for layer in model.layers:
                if type(layer) is Dropout:
                    model.layers.remove(layer)

        self.model = model
        self.modelName = modelName

    def predict(self, img):
        return self.model.predict(img)

class CnnCenterSegmentHelper:
    def __init__(self, modelsPath, removeDropout=False, train=train):
        # config
        self.inputShape = (64, 64, 1)
        self.numClasses = 6
        self.learningRate = 5e-4
        self.maxTrainEpochs = 25
        self.patienceEpochs = 2
        self.trainingBatchCount = 20000
        self.validationBatchCount = 2000
        # config end

        modelName, model = getCnnModel(self.inputShape, self.learningRate, self.numClasses)

        modelNameSuffix = "segment"
        modelWeightsPath = modelsPath + "/" + "{}-{}".format(modelName, modelNameSuffix)
        trainDataPath = "data/resources/charsDigitsLetters"

        if train:
            trainGenerator = TrioDatasetGenerator(
                trainDataPath + "/training",
                batchSize=generatorFlowBatchSize
            )

            validationGenerator = TrioDatasetGenerator(
                trainDataPath + "/validation",
                batchSize=generatorFlowBatchSize
            )

            trainCnn(model, trainGenerator, validationGenerator,
                     self.trainingBatchCount, self.validationBatchCount,
                     self.maxTrainEpochs, self.patienceEpochs, modelWeightsPath)

        model.load_weights(modelWeightsPath)

        # doesnt do anything
        if removeDropout:
            for layer in model.layers:
                if type(layer) is Dropout:
                    model.layers.remove(layer)

        self.model = model
        self.modelName = modelName
        self.outputScaler = 1. / 0.341796875 # 1 / (28 / 64 / 2 / 64 * 100) # for some reason, output is scaled by this

    def predict(self, img):
        return self.model.predict(img) * self.outputScaler

# TODO: FINISH THIS
class CnnCharacterPredictHelper:
    def __init__(self, modelsPath, removeDropout=False, train=train):
        # config
        self.inputShape = (64, 64, 1)
        self.numClasses = 36
        self.learningRate = 1e-3
        self.maxTrainEpochs = 15
        self.patienceEpochs = 2
        self.trainingBatchCount = 10000
        self.validationBatchCount = 1000
        # config end

        modelName, model = getDenseModel(self.learningRate)

        modelNameSuffix = "CharacterPredict"
        modelWeightsPath = modelsPath + "/" + "{}-{}".format(modelName, modelNameSuffix)
        trainDataPath = "data/resources/charsDigitsLetters"

        if train:
            pass

        model.load_weights(modelWeightsPath)

        # doesnt do anything
        if removeDropout:
            for layer in model.layers:
                if type(layer) is Dropout:
                    model.layers.remove(layer)

        self.model = model
        self.modelName = modelName
        self.outputScaler = 1. / 0.341796875 # 1 / (28 / 64 / 2 / 64 * 100) # for some reason, output is scaled by this

    def predict(self, img):
        return self.model.predict(img) * self.outputScaler

def runCharacterTest():
    model = Cnn28x28CharacterClassifier("data/models")

    imgsPBatch = 1
    numBatches = 50000

    testDatagen = ImageDataGenerator(
    )
    actualDatagen = ImageDataGenerator(
    )

    testGenerator = testDatagen.flow_from_directory(
        "data/imgs/predictTrain",
        target_size=(model.dim, model.dim),
        class_mode=None,
        batch_size=imgsPBatch * 3,
        shuffle=False
    )

    actualGenerator = actualDatagen.flow_from_directory(
        "data/imgs/trainImgs/raw",
        target_size=(64, 64),
        class_mode=None,
        batch_size=imgsPBatch,
        shuffle=False
    )

    correct = 0
    count = 0
    with open("data/" + fe.trainYFileName, 'r') as yFile: # for validation

        for i in range(numBatches):

            batch = testGenerator.next()
            batchActual = actualGenerator.next()

            predicts = model.predict(batch)

            rawBest = np.argsort(predicts, axis=1)

            combined = predicts.sum(axis=0)
            numbers = combined[:10]
            ops = combined[10:]

            bestNumbers = numbers.argsort()[-2:][::-1]

            operand1 = bestNumbers[0]
            operand2 = bestNumbers[1]
            op = np.argmax(ops) + 10

            if op == 10:
                op = "+"
                value = operand1 + operand2
            else:
                op = "*"
                value = operand1 * operand2

            expectedOutput = int(next(yFile).strip())

            good = value == expectedOutput
            if good:
                correct += 1

            count += 1

            # print(good, " -> Predicted:", operand1, op, operand2, "=", value, "; expected:", expectedOutput)
            # cv2.imshow("char1", tools.resize(batch[0], 10, 10))
            # cv2.imshow("char2", tools.resize(batch[1], 10, 10))
            # cv2.imshow("char3", tools.resize(batch[2], 10, 10))
            # cv2.imshow("charActual", tools.resize(batchActual[0]))
            # cv2.waitKey()

            if i % 100 == 0:
                print("\rFinished processing {}/{}".format(i, numBatches), end='')

    print("Accuracy: ", correct/count)

def runSegmentationCnnTest():
    model = CnnCenterSegmentHelper("data/models")

    testGenerator = TrioDatasetGenerator(
        "data/resources/charsDigitsLetters/validation",
        batchSize=generatorFlowBatchSize
    )

    batches = 100
    for b in range(batches):
        batch = testGenerator.next()
        for i in range(batch[0].shape[0]):
            _img = batch[0][i]
            yt = batch[1][i].reshape((3, 2))

            yp = model.predict(np.expand_dims(_img, axis=0)).reshape((3,2))

            dim = 64
            img = np.zeros((dim, dim, 3))

            img[..., 0] = _img[..., 0]

            img[int(yt[0, 0] * dim), int(yt[0, 1] * dim), 1] = 1
            img[int(yt[1, 0] * dim), int(yt[1, 1] * dim), 1] = 1
            img[int(yt[2, 0] * dim), int(yt[2, 1] * dim), 1] = 1

            img[int(yp[0, 0] * dim), int(yp[0, 1] * dim), 2] = 1
            img[int(yp[1, 0] * dim), int(yp[1, 1] * dim), 2] = 1
            img[int(yp[2, 0] * dim), int(yp[2, 1] * dim), 2] = 1

            # print(yt)
            # print(yp)
            # print(yp / yt)
            cv2.imshow("img", tools.resize(img, 10, 10))
            cv2.waitKey()

if __name__=="__main__":
    runSegmentationCnnTest()



