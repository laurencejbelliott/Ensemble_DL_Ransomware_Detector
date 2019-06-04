__author__ = "Laurence Elliott - 16600748"

import string, os, math
import numpy as np
from tensorflow.python.keras import preprocessing
from joblib import Parallel, delayed

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


def wordSequencesBenign(sampleN):
    print(benignSampleFiles[sampleN])
    try:
        text = ""
        for s in strings("benignSamples/" + benignSampleFiles[sampleN]):
            text += s + "\n"
        sequence = preprocessing.text.text_to_word_sequence(text)[:10000]
        np.savetxt("benignSequences/" + str(sampleN) + ".txt", sequence, fmt="%s")
        del text, sequence
    except Exception as e:
        print(e)


def wordSequencesMalware(sampleN):
    print(malwareSampleFiles[sampleN])
    try:
        text = ""
        for s in strings("malwareSamples/" + malwareSampleFiles[sampleN]):
            text += s + "\n"
        sequence = preprocessing.text.text_to_word_sequence(text)[:10000]
        np.savetxt("malwareSequences/" + str(sampleN) + ".txt", sequence, fmt="%s")
        del text, sequence
    except Exception as e:
        print(e)


def wordSequencesRansom(sampleN):
    print(ransomSampleFiles[sampleN])
    try:
        text = ""
        for s in strings("ransomwareSamples/" + ransomSampleFiles[sampleN]):
            text += s + "\n"
        sequence = preprocessing.text.text_to_word_sequence(text)[:10000]
        np.savetxt("ransomSequences/" + str(sampleN) + ".txt", sequence, fmt="%s")
        del text, sequence
    except Exception as e:
        print(e)


# 10,000 samples will be used from each class
# Only 9,255 benign samples were gathered, so the first 745 samples are used again
benignSampleFiles = os.listdir("benignSamples") + os.listdir("benignSamples")[:745]
malwareSampleFiles = os.listdir("malwareSamples")[:10000]
ransomSampleFiles = os.listdir("ransomwareSamples")[:10000]

nBenignSamples = len(benignSampleFiles)
nMalwareSamples = len(malwareSampleFiles)
nRansomSamples = len(ransomSampleFiles)

benignSequences = []
malwareSequences = []
ransomSequences = []

print(nBenignSamples, nMalwareSamples, nRansomSamples)

print("Generating word sequences for benign samples...")
Parallel(n_jobs=-1, verbose=11, backend="threading")(map(delayed(wordSequencesBenign), range(0, nBenignSamples)))

print("Generating word sequences for malware samples...")
Parallel(n_jobs=-1, verbose=11, backend="threading")(map(delayed(wordSequencesMalware), range(0, nMalwareSamples)))

print("Generating word sequences for ransomware samples...")
Parallel(n_jobs=-1, verbose=11, backend="threading")(map(delayed(wordSequencesRansom), range(0, nRansomSamples)))

