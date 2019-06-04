__author__ = "Laurence Elliott - 16600748"

import os, math
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import preprocessing, layers, optimizers
from tensorflow.python.keras.utils import Sequence, to_categorical
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


benignHists = ["benignHistVecs/" + fileName \
                for fileName in os.listdir("benignHistVecs")]
benignHists += benignHists[:745]

malwareHists = ["malwareHistVecs/" + fileName \
                for fileName in os.listdir("malwareHistVecs")]
malwareHists += malwareHists[:16]

ransomHists = ["ransomHistVecs/" + fileName \
                for fileName in os.listdir("ransomHistVecs")]
ransomHists = ransomHists[:10000]

nBenignSamples = len(benignHists)
nMalwareSamples = len(malwareHists)
nRansomSamples = len(ransomHists)



# x is a list of paths to training samples
x = np.array(benignHists + malwareHists + ransomHists)
# y is a list of samples' associated class labels with one-hot encoding
y = np.ones(nBenignSamples+nMalwareSamples+nRansomSamples)
y[0:nBenignSamples] = 0
y[nBenignSamples:nBenignSamples+nMalwareSamples] = 1
y[nBenignSamples+nMalwareSamples:nBenignSamples+nMalwareSamples+nRansomSamples] = 2
# represent labels with one-hot encoding
y = to_categorical(y, num_classes=3)

# Dataset is indexed by same shuffled and split indexing as the other model
trainInds = np.load("trainInds.npy")
valInds = np.load("valInds.npy")
testInds = np.load("testInds.npy")
X_train = x[trainInds]
X_val = x[valInds]
X_test = x[testInds]

y_train = y[trainInds]
y_val = y[valInds]
y_test = y[testInds]


class histSequence(Sequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = shuffle(x, y)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            np.load(file_name)
            for file_name in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        pass


class histSequenceVal(histSequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size


batch_size = 1000
sequenceGenerator = histSequence(X_train, y_train, batch_size)
validationSeqGen = histSequenceVal(X_val, y_val, batch_size)
print(validationSeqGen.__getitem__(0))

# Defining the ML model
model = Sequential()

model.add(layers.InputLayer(input_shape=(50,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(3, activation='softmax'))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(optimizer="rmsprop",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

filePath="weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filePath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbackList = [tensorboard, checkpoint]

# Training the model
model.fit_generator(generator=sequenceGenerator,
                    epochs=1000,
                    steps_per_epoch=len(sequenceGenerator),
                    verbose=1,
                    validation_data=validationSeqGen,
                    validation_steps=len(validationSeqGen),
                    workers=8,
                    use_multiprocessing=True,
                    callbacks=callbackList)
