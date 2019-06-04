__author__ = "Laurence Elliott - 16600748"

import os, math
import numpy as np

# sampleLens = []
# count = 0
# for file in os.listdir("corpus"):
#     sample = np.load("corpus/" + file)
#     zeroArr = [0]
#     try:
#         zerosInSample = np.isin(sample, zeroArr)
#         zerosIndexes = np.where(zerosInSample)
#         zerosStart = zerosIndexes[0][0]
#         sample = sample[:zerosStart]
#         sampleLen = len(sample)
#         print(count, sampleLen)
#         sampleLens.append(len(sample))
#     except:
#         sampleLen = len(sample)
#         print(count, sampleLen)
#         sampleLens.append(len(sample))
#     count += 1
#     # sample = np.concatenate((sample[0:200], sample[::-1][0:200]))
#
# minSampleLen = np.min(sampleLens)
# print(minSampleLen)

# Min sample length is 18 bytes D:
maxSequenceLen = 10000
lenSqrt = int(math.sqrt(maxSequenceLen))
print(lenSqrt)

count = 0
for file in os.listdir("corpus"):
    sample = np.load("corpus/" + file)[:maxSequenceLen]
    sample = np.rint(((sample - np.min(sample)) /
             (np.max(sample) - np.min(sample))) * 255)\
        .astype('int').reshape(lenSqrt, lenSqrt, 1)
    np.save("corpusTrunc/" + file, sample)
    print(count)
    count += 1