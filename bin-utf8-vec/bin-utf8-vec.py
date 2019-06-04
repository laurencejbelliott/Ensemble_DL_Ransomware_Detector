__author__ = "Laurence Elliott - 16600748"

import string, os, json, math, random
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import preprocessing, layers, optimizers
from tensorflow.python.keras.utils import Sequence, to_categorical
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# https://stackoverflow.com/questions/17195924/python-equivalent-of-unix-strings-utility
# Solution to Python based 'strings' alternative from SO

def strings(filename, min=4):
    with open(filename, errors="ignore", encoding="utf-8") as f:
        result = ""
        for c in f.read():
            if c in string.printable:
                result += c
                continue
            if len(result) >= min:
                yield result
            result = ""
        if len(result) >= min:  # catch result at EOF
            yield result


vocabSizes = []
wordSequenceLens = []
benignSequences = []
malwareSequences = []
nBenignSamples = 10000
nMalwareSamples = 10000
nRansomSamples = 10000


with open("maxVocabSize.txt","r") as f:
    maxVocabSize = int(f.read())

maxSequenceLen = 10000

with open("maxHashWordID.txt","r") as f:
    maxHashWordID = int(f.read())


benignCorpus = ["finalBenignCorpus/" + fileName \
                for fileName in os.listdir("finalBenignCorpus")]

malwareCorpus = ["finalMalwareCorpus/" + fileName \
                for fileName in os.listdir("finalMalwareCorpus")]
malwareCorpus += malwareCorpus[:5]

ransomCorpus = ["finalRansomCorpus/" + fileName \
                for fileName in os.listdir("finalRansomCorpus")]
ransomCorpus += ransomCorpus[:2]

# x is a list of paths to training samples
x = np.array(benignCorpus + malwareCorpus + ransomCorpus)
# y is a list of samples' associated class labels with one-hot encoding
y = np.ones(nBenignSamples+nMalwareSamples+nRansomSamples)
y[0:nBenignSamples] = 0
y[nBenignSamples:nBenignSamples+nMalwareSamples] = 1
y[nBenignSamples+nMalwareSamples:nBenignSamples+nMalwareSamples+nRansomSamples] = 2
# represent labels with one-hot encoding
y = to_categorical(y,num_classes=3)


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


class hashCorpusSequence(Sequence):

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
            # np.load(file_name)
            np.rint(((np.load(file_name) - np.min(np.load(file_name))) /
            (np.max(np.load(file_name)) - np.min(np.load(file_name)))) * 255).astype(int)
            for file_name in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        pass

class hashCorpusSequenceVal(hashCorpusSequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size


batch_size = 150
sequenceGenerator = hashCorpusSequence(X_train, y_train, batch_size)
validationSeqGen = hashCorpusSequenceVal(X_val, y_val, batch_size)


# res = int(math.sqrt(maxSequenceLen))
# classTitles = ["benign", "malware", "ransomware"]
#
# sampleCount = random.randrange(0, 2988)
# # plt.title("Sample " + str(sampleCount) + " Converted to Grayscale Image\n(" +
# #           classTitles[sequenceGenerator.__getitem__(sampleCount)[1].tolist()[0].index(1)] + ")")
# plt.imshow(sequenceGenerator.__getitem__(sampleCount)[0][0].reshape(res, res), cmap='gray')


# Defining the ML model
model = Sequential()

model.add(layers.InputLayer(input_shape=(100, 100, 1)))
model.add(layers.SpatialDropout2D(rate=0.2))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.SpatialDropout2D(rate=0.1))
model.add(layers.Conv2D(16, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.SpatialDropout2D(rate=0.1))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(optimizer="adamax",
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
