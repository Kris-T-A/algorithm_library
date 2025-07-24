import librosa
import numpy as np
import os
import PythonAlgorithmLibrary as pal

# load audio file
y, sr = librosa.load(os.path.expanduser("~/Dropbox/sound files/appended.wav"), sr=None)

print("Audio loaded with sample rate:", sr)

# set parameters
frameSize_ms = 128
bufferSize_ms = 32
nFolds = 1
nonlinearity = 1
nSpectrograms = 3

# derive parameters
NFFT = int(frameSize_ms * sr / 1000)
nBands = int(NFFT / 2 + 1)
bufferSize = int(bufferSize_ms * sr / 1000)
nAdaptiveOutputFrames = int(2**(nSpectrograms - 1))

# create spectrogram object
spectrogram = pal.Spectrogram()
cSpectrogram = spectrogram.getCoefficients()

specAdaptive = pal.SpectrogramAdaptive()
cSpecAdaptive = specAdaptive.getCoefficients()

# set coefficients
cSpectrogram['bufferSize'] = bufferSize
cSpectrogram['nBands'] = nBands
cSpectrogram['nFolds'] = nFolds
cSpectrogram['nonlinearity'] = nonlinearity
spectrogram.setCoefficients(cSpectrogram)

cSpecAdaptive['bufferSize'] = bufferSize
cSpecAdaptive['nBands'] = nBands
cSpecAdaptive['nFolds'] = nFolds
cSpecAdaptive['nSpectrograms'] = nSpectrograms
cSpecAdaptive['nonlinearity'] = nonlinearity
cSpecAdaptive['sampleRate'] = sr
specAdaptive.setCoefficients(cSpecAdaptive)

print("Spectrogram:", spectrogram)
print("Adaptive Spectrogram:", specAdaptive)

# loop through audio in frames of size bufferSize and call spectrogram and adaptive spectrogram
# preallocate the output arrays that have the size nBands x number of frames
numFrames = int(len(y) // bufferSize)
spectrogramOutput = np.zeros((nBands, numFrames), dtype=np.float32)
adaptiveSpectrogramOutput = np.zeros((nBands, nAdaptiveOutputFrames * numFrames), dtype=np.float32)

for i in range(0, numFrames):
    frame = y[i * bufferSize:(i + 1) * bufferSize]

    # compute spectrogram
    spec = spectrogram.process(frame)
    spectrogramOutput[:, i] = 10 * np.log10(np.maximum(spec, 1e-20))
    #print("Spectrogram computed for frame", i // bufferSize)

    # compute adaptive spectrogram
    specAdaptiveResult = specAdaptive.process(frame)
    adaptiveSpectrogramOutput[:, (i) * nAdaptiveOutputFrames:(i + 1) * nAdaptiveOutputFrames] = specAdaptiveResult
    #print("Adaptive Spectrogram computed for frame", i // bufferSize)

# find max value in spectrogram outputs
maxValue = np.max(spectrogramOutput)
print("Max value in spectrogram output:", maxValue)

maxValue = np.max(adaptiveSpectrogramOutput)
print("Max value in adaptive spectrogram output:", maxValue)


# print the results using matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.imshow(spectrogramOutput, aspect='auto', origin='lower', cmap='viridis', vmin=maxValue-80, vmax=maxValue)
plt.title('Spectrogram Output')
plt.colorbar()
plt.xlim(1000, 4500//4)
plt.subplot(2, 1, 2)
plt.imshow(adaptiveSpectrogramOutput, aspect='auto', origin='lower', cmap='viridis', vmin=maxValue-80, vmax=maxValue)
plt.title('Adaptive Spectrogram Output')
plt.colorbar()
plt.tight_layout()
plt.xlim(4000, 4500)
plt.show()
