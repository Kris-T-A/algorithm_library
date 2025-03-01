import numpy as np

indexEnd = 8000
nInputs = 513
nOutputs = 40

lowFreqLog = indexEnd / nInputs 
highFreqLog = np.log10(1+indexEnd/lowFreqLog)
linLogs = np.linspace(0, highFreqLog, nOutputs+1)
freqsLog = lowFreqLog * (10**(linLogs)-1)
cornerBinsFloat = (nInputs / indexEnd * freqsLog)
print("cornerBinsFloat: ", cornerBinsFloat)

cornerBinsFloatDiff = cornerBinsFloat[1:] - cornerBinsFloat[:-1]
nSmallBins = (cornerBinsFloatDiff <= 1).sum()
binsWeight = cornerBinsFloatDiff[:nSmallBins]
print("nSmallBins: ", nSmallBins)

binsWeight = cornerBinsFloat[:nSmallBins] - np.floor(cornerBinsFloat[:nSmallBins])
print("binsWeight: ", binsWeight)
cornerBins = np.round(cornerBinsFloat).astype(int)
print("cornerBins:", cornerBins)

indexStart = np.zeros(nOutputs)
indexStart[:nSmallBins] = np.floor(cornerBinsFloat[:nSmallBins]).astype(int)
indexStart[nSmallBins:] = cornerBins[nSmallBins:nOutputs]
print("indexStart: ", indexStart)


highFreqMel = np.log10(1 + indexEnd/700)*2595
linMels = np.linspace(0, highFreqMel, nOutputs+1)
freqsMel = 700 * (10**(linMels / 2595) - 1)
cornerBinsMel = (nInputs / indexEnd * freqsMel)
print(cornerBinsMel)
