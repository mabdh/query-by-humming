import numpy as np
import numpy.fft
from numpy import *
import os
import scipy
from scipy.fftpack import *
from subprocess import Popen, PIPE
import scipy.io.wavfile
import matplotlib.pyplot as plt
import pyaudio
import midi
from scipy.signal import fftconvolve, hann, decimate

def open(filename, sampleRate=44100):
        """
        Open a file (WAV or MP3), return instance of this class with data loaded in
        Note that this is a static method. This is the preferred method of constructing this object
        """
        _, ext = os.path.splitext(filename)

        if ext.endswith('mp3') or ext.endswith('m4a'):
            print("masuk")
            ffmpeg = Popen([
                "ffmpeg",
                "-i", filename,
                "-vn", "-acodec", "pcm_s16le",  # Little Endian 16 bit PCM
                "-ac", "1", "-ar", str(sampleRate),  # -ac = audio channels (1)
                "-f", "s16le", "-"],  # -f wav for WAV file
                stdin=PIPE, stdout=PIPE, stderr=open(os.devnull, "w"))

            rawData = ffmpeg.stdout

            samples = numpy.fromstring(rawData.read(), numpy.int16)
            samples = samples.astype('float32') / 32767.0

            return samples
        elif ext.endswith('wav'):
            sampleRate, samples = scipy.io.wavfile.read(filename)

            # Convert to float
            samples = samples.astype('float32') / 32767.0

            # Get left channel
            if len(samples.shape) > 1:
                samples = samples[:, 0]

            return samples
def plot_fft(signal, sampleRate):
    n = len(signal)
    p = fft(signal)

    fftPoint = ceil((n+1)/2.0)
    p = p[0:fftPoint]
    p = abs(p)

    # scale for independency with length or fs
    p = p / float(n)
    p = p**2

    # check even or odd the number point of fft
    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p)-1] = p[1:len(p)-1] * 2

    freqAxis = arange(0, fftPoint, 1.0) * (sampleRate / n)
    plt.plot(freqAxis/1000, 10*log10(p))
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.show()

def zero_crossing(signal,sampleRate):
    indices = np.where((signal[1:] >= 0) & (signal[:-1] < 0))
    indices = indices[0]
    # print(indices)
    # raising = indices[0::2]
    # print(str(len(raising)))
    # interpolate
    zc = [i - signal[i] / (signal[i+1] - signal[i]) for i in indices]
    return (indices,sampleRate/np.diff(zc), np.diff(zc))

def freq_to_midi(freqArray):
    return [int(round(69 + 12 * math.log(f / 440.0, 2))) for f in freqArray]

def create_midi(midiNotes):
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    i = 0
    idx = 0
    while i < len(midiNotes)/2:
        # i_n = i
        on = midi.NoteOnEvent(tick=idx, velocity=50, pitch=midiNotes[i])
        track.append(on)
        i = i + 1
        idx = idx + 1
        off = midi.NoteOffEvent(tick=idx, pitch=midiNotes[i])
        track.append(off)
        i = i + 1
        idx = idx + 1
        print(i)
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    print(pattern)
    midi.write_midifile("test.mid", pattern)
    # print(str(len(indices)) + " " + str(len(midiNotes)) + " " + str(len(duration)))

def fundamental_freq_fft_windowed(signal,sampleRate):
    n = len(signal)
    frame_ms = 100
    frame_s = frame_ms / 1000.0
    frame_samples = int(frame_s * sampleRate)
    # percent %
    overlap = 0
    overlap_samples = int(overlap * frame_samples)

    i = 0
    freqArray = []
    while(i < n):
        signal_sample = signal[i:i+frame_samples]

        n_sig_sample = len(signal_sample)

        windowed = signal_sample * hann(n_sig_sample)

        freq = rfft(windowed)    
        freq_d2 = rfft(windowed)
        freq_d3 = rfft(windowed)

        freq_d2 =  freq_d2[0::2]
        diff = len(freq) - len(freq_d2)
        freq_d2 = np.lib.pad(freq_d2, (0,diff),'constant', constant_values=(0,0))
        freq_d3 =  freq_d3[0::3]
        diff = len(freq) - len(freq_d3)
        freq_d3 = np.lib.pad(freq_d3, (0,diff),'constant', constant_values=(0,0))

        mul1 = np.multiply(freq,freq_d2)
        mul2 = np.multiply(mul1,freq_d3)
        
        freqF0 = argmax(mul2) * (sampleRate / n_sig_sample)
        # print(str(freqF0))
        if(freqF0 > 0.0):
            freqF0 = int(round(69 + 12 * math.log(freqF0 / 440.0, 2))) 
        else:
            freqF0 = 0
        freqArray.append(freqF0)
        i = i + frame_samples - overlap_samples
    freqArray[freqArray == -inf] = 0
    plt.figure(1)

    # freqAxis = arange(0, len(freq), 1.0) * (sampleRate / len(freq))
    # plt.plot(freqAxis/1000, 10*log10(freq))
    plt.plot(freqArray)
    plt.xlabel('k-sample')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    print(freqArray)
    return freqArray

def fundamental_freq_whole_file(signal,sampleRate):
    n = ceil(len(signal)/2)
    
    windowed = signal[0:n] * hann(n)
    freq = rfft(windowed)
    freq_d2 = rfft(windowed)
    freq_d3 = rfft(windowed)

    # freq = abs(freq)
    # freq_d2 = abs(freq_d2)
    # freq_d3 = abs(freq_d3)    

    freq_d2 =  freq_d2[0::2]
    diff = len(freq) - len(freq_d2)
    freq_d2 = np.lib.pad(freq_d2, (0,diff),'constant', constant_values=(0,0))
    freq_d3 =  freq_d3[0::3]
    diff = len(freq) - len(freq_d3)
    freq_d3 = np.lib.pad(freq_d3, (0,diff),'constant', constant_values=(0,0))
    
    mul1 = np.multiply(freq,freq_d2)
    mul2 = np.multiply(mul1,freq_d3)

    print(str(len(freq)))
    print(str(len(freq_d2)))
    print(str(len(freq_d3)))

    freqF0 = argmax(mul2) * (sampleRate / n)
    print("F0 = ", freqF0)


    freqAxis = arange(0, len(mul2), 1.0) * (sampleRate / len(mul2))
    plt.figure(4)
    plt.plot(freqAxis/1000, mul2)



    plt.figure(1)

    freqAxis = arange(0, len(freq), 1.0) * (sampleRate / len(freq))
    plt.plot(freqAxis/1000, 10*log10(freq))
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    

    plt.figure(2)
    freqAxis = arange(0, len(freq_d2), 1.0) * (sampleRate / len(freq_d2))
    plt.plot(freqAxis/1000, 10*log10(freq_d2))
    plt.xlabel('Frequency (kHz) Downsampled 2')
    plt.ylabel('Power (dB)')


    plt.figure(3)
    freqAxis = arange(0, len(freq_d3), 1.0) * (sampleRate / len(freq_d3))
    plt.plot(freqAxis/1000, 10*log10(freq_d3))
    plt.xlabel('Frequency (kHz) Downsampled 3')
    plt.ylabel('Power (dB)')
    
    plt.show()



# stream = open("wavFile/Berklee/piano_C2.wav")
stream = open("wavFile/GScale.mp3")
# partStream = stream[0:(int(len(stream)*0.3))]
# plt.plot(stream)
# plt.show()
# plot_fft(stream,44100)
sampleRate = 44100
freqArray = fundamental_freq_fft(stream,sampleRate)
# midiNotes = freq_to_midi(freqArray)
create_midi(freqArray)