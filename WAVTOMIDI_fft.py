import numpy as np
import numpy.fft
from numpy import *
import os
import scipy
from scipy.fftpack import *
from subprocess import Popen, PIPE
import scipy.io.wavfile
import matplotlib.pyplot as plt
# import pyaudio
import midi, pylab
from scipy.signal import fftconvolve, hann, decimate
from scipy import *
from scipy import signal
import operator

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

def freq_to_midi(freqArray):
    return [int(round(69 + 12 * math.log(f / 440.0, 2))) for f in freqArray]

def create_midi(midiNotes,tick_ms):
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    i = 0
    idx = 0
    while i < len(midiNotes):
        
        on = midi.NoteOnEvent(tick=int(tick_ms[idx][0]), velocity=100, pitch=midiNotes[i])
        track.append(on)
        
        off = midi.NoteOffEvent(tick=int(tick_ms[idx][1]), pitch=midiNotes[i])
        track.append(off)
        i = i + 1
        idx = idx + 1
        print(i)
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    print(pattern)
    midi.write_midifile("test.mid", pattern)
    # print(str(len(indices)) + " " + str(len(midiNotes)) + " " + str(len(duration)))

def getEnergy(streamData):
	sumRes = np.sum(abs(streamData)**2.0)
	return sumRes/len(streamData)

def fundamental_freq_fft(signal,sampleRate):
    n = len(signal)
    print(n)

    # To quantify energy
    frame_ms = 1
    frame_s = frame_ms / 1000.0
    frame_samples = int(frame_s * sampleRate)
    # frame_samples = 1024
    # percent %
    overlap = 0
    overlap_samples = int(overlap * frame_samples)
    downsampleorder = 100
    i = 0
    freqArray = []

    # print(frame_samples)

    #calculate energy
    energyStream = []
    while(i < n):
        signal_sample = signal[i:i+frame_samples]
        n_sig_sample = len(signal_sample)
        Nfft = n_sig_sample
        windowed = signal_sample * hann(n_sig_sample, Nfft)
        en = getEnergy(windowed)
        energyStream.append(en)
        i = i + frame_samples - overlap_samples

    #threshold
    meanEnergy = np.sum(energyStream)/len(energyStream)
    energyTh = meanEnergy/4

    for i in range(len(energyStream)):
        if energyStream[i] < energyTh:
            energyStream[i] = 0.0
    # print(energyTh)

    #segmenting
    energyStream = np.cumsum(energyStream)
    energyStream = energyStream[0::downsampleorder]
    energyStream = np.gradient(np.gradient(energyStream))

    #get indices with zero crossing
    indices = []
    indices = np.where(((energyStream[1:] >= 0) & (energyStream[:-1] < 0)) | ((energyStream[1:] < 0) & (energyStream[:-1] >= 0)))[0]
    if(len(indices)%2!=0):
        indices = indices[0:len(indices)-1]
    
    indices = np.split(indices,int(len(indices)/2)) 
    indices = [idx * (1.0-overlap) for idx in indices]
    print(n/frame_samples)
    print(len(energyStream))
    print(indices)
    # for i in range(1,len(energyStream)-1):
        # if ((energyStream[i-1] < 0) and (energyStream[i+1] > 0)) or ((energyStream[i-1] > 0) and (energyStream[i+1] < 0)):
        #   indices.append(i)

    # plt.figure(1)
    # plt.plot(energyStream)
    # plt.figure(2)
    # plt.plot(signal)
    # plt.figure(3)
    # plt.step(np.arange(len(indices)),indices)
    # plt.show()
    # print(energyStream)
    
    multiplier = downsampleorder * frame_samples
    tick_indices=[]
    for idx in indices:
        idx = idx * multiplier
        istart = idx[0]
        frame_samples = idx[1] - idx[0] 

        if istart+frame_samples >= n:
            frame_samples = n - istart
        print("debug " + str(n) + " " + str(istart) + " " + str(frame_samples))
        
        tickidx = np.array([istart,istart+frame_samples])
        tickidx = (tickidx/sampleRate) * 100 #convert to ms
        tickidx = tickidx.astype(int)

        tick_indices.append(tickidx)

        signal_sample = signal[istart:istart+frame_samples]
        n_sig_sample = len(signal_sample)

        # plt.figure(5)
        # plt.plot(signal_sample)
        # plt.show()

        Nfft = n_sig_sample
        windowed = signal_sample * hann(n_sig_sample, Nfft)
        # print("debug " + str(len(windowed)))
        freq = rfft(windowed, Nfft)    
        freq_d2 = rfft(windowed, Nfft)
        freq_d3 = rfft(windowed, Nfft)

        freq_d2 =  freq_d2[0::2]
        diff = len(freq) - len(freq_d2)
        freq_d2 = np.lib.pad(freq_d2, (0,diff),'constant', constant_values=(0,0))
        freq_d3 =  freq_d3[0::3]
        diff = len(freq) - len(freq_d3)
        freq_d3 = np.lib.pad(freq_d3, (0,diff),'constant', constant_values=(0,0))


        freq[:20] = 0
        freq_d2[:20] = 0
        freq_d3[:20] = 0

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
        ax1.plot(freq)
        ax1.set_title('Sharing both axes')
        ax2.plot(freq_d2)
        ax3.plot(freq_d3)

        mul1 = np.multiply(freq,freq_d2)
        mul2 = np.multiply(mul1,freq_d3)

        freqF0 = argmax(mul2) * (sampleRate / n_sig_sample)
        
        if(freqF0 > 0.0):
            freqF0 = int(round(69 + 12 * math.log(freqF0 / 440.0, 2)))
        else:
            freqF0 = 0
        print(freqF0)
        freqArray.append(freqF0)
        # i = i + frame_samples - overlap_samples
    plt.figure(1)

    # freqAxis = arange(0, len(freq), 1.0) * (sampleRate / len(freq))
    # plt.plot(freqAxis/1000, 10*log10(freq))
    # plt.plot(freqArray)
    # plt.xlabel('k-sample')
    # plt.ylabel('Frequency (Hz)')
    
    # plt.figure(2)
    # plt.plot(signal)
    # plt.show()
    print(tick_indices)
    print(freqArray)
    return freqArray, tick_indices

def show_spectrogram(x, fs):
	plt.figure(12)
	Pxx, freqs, bins, im = plt.specgram(x, NFFT=1024, Fs=fs, noverlap=900,
                                cmap=plt.cm.gist_earth)
	plt.show()


# stream = open("wavFile/Berklee/piano_C2.wav")
# stream = open("wavFile/gnote.mp3")
stream = open("wavFile/")
stream = open("wavFile/gundulpacul.wav")

# partStream = stream[0:(int(len(stream)*0.3))]
# plt.plot(stream)
# plt.show()
# plot_fft(stream,44100)
sampleRate = 44100
# sampleRate = 8000
# show_spectrogram(stream,sampleRate)

freqArray, tick_ms = fundamental_freq_fft(stream,sampleRate)
# midiNotes = freq_to_midi(freqArray)
create_midi(freqArray,tick_ms)
# pattern = midi.read_midifile("test.mid")
# print(pattern)