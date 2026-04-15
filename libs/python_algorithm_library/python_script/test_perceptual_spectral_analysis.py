# Test script for PerceptualSpectralAnalysis.
# Processes a test signal at three different buffer sizes and plots the resulting spectrograms.
# The test signal consists of: 0.5s silence, 1s white noise, 0.5s silence, 1s 1kHz sinusoid,
# 0.5s silence, 2s frequency sweep (100Hz-10kHz), 0.5s silence (5.5s total).

import numpy as np
import matplotlib.pyplot as plt
import PythonAlgorithmLibrary as pal

# shared coefficients for all three runs
sampleRate = 48000
frequencyMin = 20       # minimum frequency in Hz
frequencyMax = 20000    # maximum frequency in Hz
spectralTilt = False    # no spectral tilt correction
nSpectrograms = 3       # number of spectrograms (each halves the buffer size)
nFolds = 1              # number of folds: frameSize = nFolds * 2 * (nBands - 1)
nonlinearity = 1        # window nonlinearity factor
method = "Adaptive"     # use adaptive method

# generate test signal segments
silence_05 = np.zeros(int(0.5 * sampleRate), dtype=np.float32)                                         # 0.5s silence
white_noise = np.random.randn(int(1.0 * sampleRate)).astype(np.float32) * 0.5                          # 1s white noise
sinusoid_1k = np.sin(2 * np.pi * 1000 * np.arange(int(1.0 * sampleRate)) / sampleRate).astype(np.float32)  # 1s 1kHz sinusoid
t_sweep = np.arange(int(2.0 * sampleRate)) / sampleRate                                                # 2s time vector for sweep
sweep = np.sin(2 * np.pi * np.cumsum(100 + (10000 - 100) * t_sweep / 2.0) / sampleRate).astype(np.float32)  # 2s frequency sweep 100Hz-10kHz

# concatenate all segments into a single signal
signal = np.concatenate([silence_05, white_noise, silence_05, sinusoid_1k, silence_05, sweep, silence_05])


def process_signal(bufferSize):
    """Create a PerceptualSpectralAnalysis instance with the given bufferSize and process the test signal frame by frame."""
    nBands = 2 * bufferSize + 1                 # number of perceptual frequency bands
    nOutputFrames = 2 ** (nSpectrograms - 1)    # number of output frames per input buffer (4 for nSpectrograms=3)

    # create and configure the algorithm
    psa = pal.PerceptualSpectralAnalysis()
    psa.setCoefficients({
        'bufferSize': bufferSize,
        'nBands': nBands,
        'sampleRate': sampleRate,
        'frequencyMin': frequencyMin,
        'frequencyMax': frequencyMax,
        'spectralTilt': spectralTilt,
        'nSpectrograms': nSpectrograms,
        'nFolds': nFolds,
        'nonlinearity': nonlinearity,
        'method': method,
    })
    print(f"bufferSize={bufferSize}, nBands={nBands}, coefficients:", psa.getCoefficients())

    # process signal frame by frame and collect output
    numFrames = len(signal) // bufferSize
    output = np.zeros((nBands, nOutputFrames * numFrames), dtype=np.float32)

    for i in range(numFrames):
        frame = signal[i * bufferSize:(i + 1) * bufferSize]
        result = psa.process(frame)
        output[:, i * nOutputFrames:(i + 1) * nOutputFrames] = result

    return output


# run at three buffer sizes, each halving the previous
bufferSizes = [2048, 1024, 512]
outputs = [process_signal(bs) for bs in bufferSizes]

# find global max across all spectrograms and set min to 50 dB below
globalMax = max(np.max(o) for o in outputs)
globalMin = globalMax - 50
print(f"Global min: {globalMin:.2f}, Global max: {globalMax:.2f}")

# plot the three spectrograms with shared color scale
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
signalDuration = len(signal) / sampleRate
for ax, output, bs in zip(axes, outputs, bufferSizes):
    nOutputFrames = 2 ** (nSpectrograms - 1)
    numFrames = len(signal) // bs
    timeExtent = numFrames * bs / sampleRate  # total time covered by processed frames
    im = ax.imshow(output, aspect='auto', origin='lower', cmap='viridis', vmin=globalMin, vmax=globalMax,
                   extent=[0, timeExtent, 0, output.shape[0]])
    ax.set_title(f'bufferSize={bs}, nBands={2 * bs + 1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Perceptual Band')
    fig.colorbar(im, ax=ax, label='Magnitude')

fig.suptitle(f'Global range: [{globalMin:.2f}, {globalMax:.2f}]', fontsize=12)
plt.tight_layout()
plt.show()
