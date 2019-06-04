__author__ = "Laurence Elliott - 16600748"

from capstone import *
import pefile, os
import numpy as np
from matplotlib import pyplot as plt

benignPaths = ["../bin-utf8-vec/benignSamples/" + sample for sample in os.listdir("../bin-utf8-vec/benignSamples")]
malwarePaths = ["../bin-utf8-vec/malwareSamples/" + sample for sample in os.listdir("../bin-utf8-vec/malwareSamples")]
ransomPaths = ["../bin-utf8-vec/ransomwareSamples/" + sample for sample in os.listdir("../bin-utf8-vec/ransomwareSamples")]

nSamples = len(benignPaths) + len(malwarePaths) + len(ransomPaths)
benignOpCodeSet = set()
benignOpCodeDicts = []
benignOpCodeFreqs = {}

count = 1
for sample in benignPaths:
    try:
        pe = pefile.PE(sample, fast_load=True)
        entryPoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        data = pe.get_memory_mapped_image()[entryPoint:]
        cs = Cs(CS_ARCH_X86, CS_MODE_32)

        opcodes = []
        for i in cs.disasm(data, 0x1000):
            opcodes.append(i.mnemonic)

        opcodeDict = {}
        total = len(opcodes)

        benignOpCodeSet = set(list(benignOpCodeSet) + opcodes)
        for opcode in benignOpCodeSet:
            freq = 1
            for op in opcodes:
                if opcode == op:
                    freq += 1
            try:
                benignOpCodeFreqs[opcode] += freq
            except:
                benignOpCodeFreqs[opcode] = freq

            opcodeDict[opcode] = round((freq / total) * 100, 2)

        benignOpCodeDicts.append(opcodeDict)

        os.system("clear")
        print(str((count / nSamples) * 100) + "%")
        count += 1

    except Exception as e:
        print(e)


malwareOpCodeSet = set()
malwareOpCodeDicts = []
malwareOpCodeFreqs = {}

count = len(malwarePaths)
for sample in malwarePaths:
    try:
        pe = pefile.PE(sample, fast_load=True)
        entryPoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        data = pe.get_memory_mapped_image()[entryPoint:]
        cs = Cs(CS_ARCH_X86, CS_MODE_32)

        opcodes = []
        for i in cs.disasm(data, 0x1000):
            opcodes.append(i.mnemonic)

        opcodeDict = {}
        total = len(opcodes)

        malwareOpCodeSet = set(list(malwareOpCodeSet) + opcodes)
        for opcode in malwareOpCodeSet:
            freq = 1
            for op in opcodes:
                if opcode == op:
                    freq += 1
            try:
                malwareOpCodeFreqs[opcode] += freq
            except:
                malwareOpCodeFreqs[opcode] = freq

            opcodeDict[opcode] = round((freq / total) * 100, 2)

        malwareOpCodeDicts.append(opcodeDict)

        os.system("clear")
        print(str((count / nSamples) * 100) + "%")
        count += 1

    except Exception as e:
        print(e)

ransomOpCodeSet = set()
ransomOpCodeDicts = []
ransomOpCodeFreqs = {}

count = len(benignPaths) + len(malwarePaths)
for sample in ransomPaths:
    try:
        pe = pefile.PE(sample, fast_load=True)
        entryPoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        data = pe.get_memory_mapped_image()[entryPoint:]
        cs = Cs(CS_ARCH_X86, CS_MODE_32)

        opcodes = []
        for i in cs.disasm(data, 0x1000):
            opcodes.append(i.mnemonic)

        opcodeDict = {}
        total = len(opcodes)

        ransomOpCodeSet = set(list(ransomOpCodeSet) + opcodes)
        for opcode in ransomOpCodeSet:
            freq = 1
            for op in opcodes:
                if opcode == op:
                    freq += 1
            try:
                ransomOpCodeFreqs[opcode] += freq
            except:
                ransomOpCodeFreqs[opcode] = freq

            opcodeDict[opcode] = round((freq / total) * 100, 2)

        ransomOpCodeDicts.append(opcodeDict)

        os.system("clear")
        print(str((count / nSamples) * 100) + "%")
        count += 1

    except Exception as e:
        print(e)


opCodeFreqsSorted = np.genfromtxt("top50opcodes.csv", delimiter=",", dtype="str")[1:, 0]

count = 0
for opDict in benignOpCodeDicts:
    opFreqVec = []
    for opcode in opCodeFreqsSorted[:50]:
        try:
            opFreqVec.append(opDict[opcode])
        except Exception as e:
            if str(type(e)) == "<class 'KeyError'>":
                opFreqVec.append(0.0)

    np.save("benignHistVecs/" + str(count)+".npy", opFreqVec)
    os.system("clear")
    print(str((count / nSamples) * 100) + "%")
    count += 1


count = len(benignPaths)
for opDict in malwareOpCodeDicts:
    opFreqVec = []
    for opcode in opCodeFreqsSorted[:50]:
        try:
            opFreqVec.append(opDict[opcode])
        except Exception as e:
            if str(type(e)) == "<class 'KeyError'>":
                opFreqVec.append(0.0)

    np.save("malwareHistVecs/" + str(count)+".npy", opFreqVec)
    os.system("clear")
    print(str((count / nSamples) * 100) + "%")
    count += 1


count = len(benignPaths) + len(malwarePaths)
for opDict in ransomOpCodeDicts:
    opFreqVec = []
    for opcode in opCodeFreqsSorted[:50]:
        try:
            opFreqVec.append(opDict[opcode])
        except Exception as e:
            if str(type(e)) == "<class 'KeyError'>":
                opFreqVec.append(0.0)

    np.save("ransomHistVecs/" + str(count)+".npy", opFreqVec)
    os.system("clear")
    print(str((count / nSamples) * 100) + "%")
    count += 1


# benignVecPaths = ["benignHistVecs/" + vecPath for vecPath in os.listdir("benignHistVecs")]

# for vecPath in benignVecPaths:
#     opFreqVec = np.load(vecPath)
#     print(opFreqVec)
#     plt.figure(count)
#     plt.bar(np.arange(len(opFreqVec)), opFreqVec)
#     plt.show()
