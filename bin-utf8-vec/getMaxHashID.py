__author__ = "Laurence Elliott - 16600748"

import os
import numpy as np


x = np.core.defchararray.add(np.array(["corpus/"]), np.array(os.listdir("corpus")))
maxHashWordIDs = []

count = 0
for file in x:
    sampleArr = np.load(file)
    maxHashWordIDs.append(max(sampleArr))
    os.system("clear")
    print(count)
    count += 1

maxHashWordID = max(maxHashWordIDs)
print("maxHashWordID:", maxHashWordID)
with open("maxHashWordID.txt", "w") as f:
    f.write(str(maxHashWordID))
