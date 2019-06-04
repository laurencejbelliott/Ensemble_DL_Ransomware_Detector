__author__= "Laurence Elliott - 16600748"

import os
import numpy as np
from keras import preprocessing

docTotal = 30000
maxSequenceLen = 10000

with open("finalVocabSize.txt", "r") as f:
    maxVocabSize = int(f.readline())


def hashWordSequences(sequencePath, outputPath, maxSeqLen, vocabSize, docT, nSamples):
    docCount = 0
    if sequencePath[-1] != "/": sequencePath += "/"
    if outputPath[-1] != "/": outputPath += "/"

    seqFiles = [sequencePath + os.listdir(sequencePath)[i] for i in range(0, nSamples)]
    for seqFile in seqFiles:
        with open(seqFile, "r") as f:
            try:
                sequence = np.char.replace(np.array(f.readlines()), "\n", "")
                text = " ".join(sequence)
                hashWordIDs = preprocessing.text.hashing_trick(text, round(maxVocabSize * 1.5), hash_function='md5')
                docLen = len(hashWordIDs)
                if docLen < maxSequenceLen:
                    hashWordIDs += [0 for i in range(0, maxSequenceLen-docLen)]
                hashWordIDs = np.array(hashWordIDs).reshape(100, 100, 1)
                np.save(outputPath + str(docCount) + ".npy", hashWordIDs)
                if docCount % 100 == 0:
                    print(str(int((docCount / nSamples) * 100)) + "%")
                docCount += 1
            except Exception as e:
                print(e)


# print("Max vocab size (for hashing trick):", maxVocabSize, "\nMax sequence length (for zero padding):", maxSequenceLen)
#
# print("Hashing benign word sequences...")
# hashWordSequences(sequencePath="benignSequences",
#                   outputPath="finalBenignCorpus",
#                   maxSeqLen=maxSequenceLen,
#                   vocabSize=maxVocabSize,
#                   docT=docTotal,
#                   nSamples=10000)

print("Hashing malware word sequences...")
hashWordSequences(sequencePath="malwareSequences",
                  outputPath="finalMalwareCorpus",
                  maxSeqLen=maxSequenceLen,
                  vocabSize=maxVocabSize,
                  docT=docTotal,
                  nSamples=10000)

print("Hashing ransomware word sequences...")
hashWordSequences(sequencePath="ransomSequences",
                  outputPath="finalRansomCorpus",
                  maxSeqLen=maxSequenceLen,
                  vocabSize=maxVocabSize,
                  docT=docTotal,
                  nSamples=10000)
