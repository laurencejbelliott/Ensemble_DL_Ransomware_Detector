__author__ = "Laurence Elliott - 16600748"

from capstone import *
import pefile, os

# samplePaths = ["testSamples/" + sample for sample in os.listdir("testSamples")]
samplePaths = ["../bin-utf8-vec/benignSamples/" + sample for sample in os.listdir("../bin-utf8-vec/benignSamples")] + \
["../bin-utf8-vec/malwareSamples/" + sample for sample in os.listdir("../bin-utf8-vec/malwareSamples")] + \
["../bin-utf8-vec/ransomwareSamples/" + sample for sample in os.listdir("../bin-utf8-vec/ransomwareSamples")]



opcodeSet = set()
opCodeDicts = []
opCodeFreqs = {}
nSamples = len(samplePaths)

count = 1
for sample in samplePaths:
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

        opcodeSet = set(list(opcodeSet) + opcodes)
        for opcode in opcodeSet:
            freq = 1
            for op in opcodes:
                if opcode == op:
                    freq += 1
            try:
                opCodeFreqs[opcode] += freq
            except:
                opCodeFreqs[opcode] = freq

            opcodeDict[opcode] = round((freq / total) * 100, 2)


        opCodeDicts.append(opcodeDict)
        os.system("clear")
        print(str((count / nSamples) * 100) + "%")
        count += 1
    except Exception as e:
        print(e)

    # for opcode in opcodeSet:
    #     print(opcode, str(opcodeDict[opcode]) + "%")

# for opcodeDict in opCodeDicts:
#     freqSorted = sorted(opcodeDict, key=opcodeDict.get)[-1:0:-1]
#     print(opcodeDict[freqSorted[0]], opcodeDict[freqSorted[1]], opcodeDict[freqSorted[2]], freqSorted)

opCodeFreqsSorted = sorted(opCodeFreqs, key=opCodeFreqs.get)[-1:0:-1]

with open("top50opcodes.csv", "w") as f:
    f.write("opcode, frequency\n")
    for opcode in opCodeFreqsSorted[:50]:
        f.write(str(opcode) + ", " + str(opCodeFreqs[opcode]) + "\n")
        print(opcode, opCodeFreqs[opcode])

