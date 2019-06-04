__author__ = "Laurence Elliott - 16600748"

import os
import numpy as np

benignSequenceFiles = np.core.defchararray.add(np.array(["benignSequences/"]),
                                               np.array(os.listdir("benignSequences")))

malwareSequenceFiles = np.core.defchararray.add(np.array(["malwareSequences/"]),
                                               np.array(os.listdir("malwareSequences")))

ransomSequenceFiles = np.core.defchararray.add(np.array(["ransomSequences/"]),
                                               np.array(os.listdir("ransomSequences")))


vocab = set()


for sampleN in range(0, 10000):
    with open(benignSequenceFiles[sampleN]) as f:
        vocab = vocab.union(set(f.readlines()))
        if sampleN % 100 == 0:
            print(sampleN)

for sampleN in range(0, 10000):
    with open(malwareSequenceFiles[sampleN]) as f:
        vocab = vocab.union(set(f.readlines()))
        if sampleN % 100 == 0:
            print(sampleN + 10000)

for sampleN in range(0, 10000):
    with open(ransomSequenceFiles[sampleN]) as f:
        vocab = vocab.union(set(f.readlines()))
        if sampleN % 100 == 0:
            print(sampleN + 20000)

with open("finalVocabSize.txt", "w") as f:
    f.write(str(len(vocab)))