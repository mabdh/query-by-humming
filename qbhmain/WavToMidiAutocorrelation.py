# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 00:12:24 2016

@author: maureen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from subprocess import Popen, PIPE
import midi
from pylab import *
from scipy.signal import butter, lfilter, freqz

class WavToMidiAutocorrelation:
    sampleRate = 44100
    
    def open(self, filename, sampleRate=44100):
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
    
                samples = np.fromstring(rawData.read(), np.int16)
                samples = samples.astype('float32') / 32767.0
    
                return samples, sampleRate
            elif ext.endswith('wav'):
                sampleRate, samples = scipy.io.wavfile.read(filename)
    
                # Convert to float
                samples = samples.astype('float32') / 32767.0
    
                # Get left channel
                if len(samples.shape) > 1:
                    samples = samples[:, 0]
                return samples, sampleRate
    
    def freq_to_midi(self,freqArray):
        midi = []
        for f in freqArray:
            if(f>0):
                midi.append(int(round(69 + 12 * np.log2(f / 440.0))))
            else:
                midi.append(0)
        return midi
    
    
    
    
    def cut2Frames(self, x, noframe, nooverlap):
        i=0
        y=[]
        while(i+noframe<len(x)):
            data = x[int(i):int(i+noframe)]
            y.append(data)
            i += noframe - nooverlap
        return y 
    
    def windowing(self, x, window):
        y=[]
        for e in x:
            y.append(e * window)
        return y
    
    
    def acf(self, x):
        result = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] \
            for i in range(1, len(x))])
        return result[:int(result.size/2)]
        
    def preemphasis(self,x):
        y=np.empty(len(x))
        alpha = 0.95
        y[0]=x[0]
        for n in range(1,len(x)):
            y[n] = x[n] - alpha*x[n-1]
        return y
        
    def clipping(self, x):
        y=np.empty(len(x))
        maxA = np.max(x)
        cL = (30/100) * maxA
        for i in range(0,len(x)):
            a = x[i]
            if(np.abs(a)<=cL):
                y[i] = 0
            elif(a > cL):
                y[i] = a - cL
            elif(a<-cL):
                y[i] = a + cL
        return y
        
    def clippingProcess(self, x):
        y = []
        for a in x:
            y.append(self.clipping(a))
        return y
        
    def create_midi(self,midiNotes,tick_ms,filename,hummflag):
            pattern = midi.Pattern()
            track = midi.Track()
            pattern.append(track)
    
            i = 0
            idx = 0
            while i < len(midiNotes):
                
                on = midi.NoteOnEvent(tick=int(tick_ms[idx][0]), velocity=100, pitch=int(midiNotes[i]))
                track.append(on)
                
                off = midi.NoteOffEvent(tick=int(tick_ms[idx][1]), pitch=int(midiNotes[i]))
                track.append(off)
                i = i + 1
                idx = idx + 1
            eot = midi.EndOfTrackEvent(tick=1)
            track.append(eot)
            # print(pattern)
            if hummflag==0:
                midi.write_midifile("midiFile/" + filename + ".mid", pattern)
            else:
                midi.write_midifile("hummFile/" + filename + ".mid", pattern)
                
        
    def energyCalculation(self, x):
        e = []
        for i in range(0,len(x)):
            r = np.sum(np.power(x[i],2))/len(x[i])
            e.append(r)
        return e
        
    def onsetDetection(self, x, hflag):
        energy = self.energyCalculation(x)
        energyTh = np.mean(energy)/2
        energyStream = np.asarray(energy)
        low_values_indices = energyStream < energyTh
        energyStream[low_values_indices] = 0
        energyStream = np.gradient(np.gradient(energyStream)) 
        a = 5
        b = 1
        if(hflag==0):
            cLambda = 1.0
            cAlpha = 1.0
            
        else:
            cLambda = 0.5
            cAlpha = 0.5


        delta=[]
        for i in range(0,len(energyStream)):
            d = energyStream[i-a:i+b+1]
            dn = cLambda*np.median(d) + cAlpha * np.mean(d)
            delta.append(dn)
        peakCandidates = energyStream - delta
        absPeakCandidates = np.abs(peakCandidates)
        absPeakCandidates=np.nan_to_num(absPeakCandidates)
        a=np.empty(len(absPeakCandidates))
        a.fill(1)
        if(hflag==0):
            silentThreshold = 0.025*np.median(absPeakCandidates)
            idxA  = np.where(absPeakCandidates<=silentThreshold)
        else:
            silentThreshold = 0.025*np.median(absPeakCandidates)
            idxA  = np.where(absPeakCandidates<=silentThreshold)
            
        a[idxA] = -1
        s3= np.sign(a)
        zcEnergyIdx = np.where(np.diff(s3))[0] 
        if(len(zcEnergyIdx)%2==0):
            onset=zcEnergyIdx[0:][::2]
            offset=zcEnergyIdx[1:][::2]
        else:
            onset=zcEnergyIdx[0:][::2]
            offset=zcEnergyIdx[1:][::2]
            offset=offset+zcEnergyIdx[-1]
        for i in range(0,len(onset)):
            onset[i] = onset[i]+np.argmax(absPeakCandidates[onset[i]:offset[i]+1])
        
#        b=np.empty(len(absPeakCandidates))
#        b.fill(0)
#        b[onset] = 1   
#        b[offset] = 1       
#        plt.figure()
#        plt.subplot(2, 1, 1)
#        plt.plot(absPeakCandidates)
#        plt.subplot(2, 1, 2)
#        plt.plot(b)
        return onset, offset
        
    def pitchDetection(self, clippedFrames,minP, maxP,autoCorrelationThres, sr):
        f0 = []    
    
        for i in range(0,len(clippedFrames)):
    
            segment = clippedFrames[i]
            result=self.acf(segment)        
            idx=np.argmax(result[minP-1:maxP+1])
    
            oriIdx = minP+idx-1
            q = result[oriIdx]
            if(result[oriIdx]<result[oriIdx-1] and result[oriIdx] < result[oriIdx+1]) or (q<autoCorrelationThres):
                fundamentalFreq=0
                f0.append(0)
            else:
                shiftedIdx = oriIdx
                if(oriIdx+1 < len(result)):
                    mid  = result[oriIdx]
                    left = result[oriIdx-1]
                    right = result[oriIdx+1]
    
                    if(mid*2 - left - right >0):
                        shift = 0.5*(right-left) + (mid*2 - left - right)
                        shiftedIdx = oriIdx + shift
                fundamentalFreq = sr/shiftedIdx
                f0.append(fundamentalFreq)
        return f0
        
    def midiPostProcessing(self, midiResult, onset, offset, sr, noframe, nooverlap):
        midi = np.empty(len(onset))
        ticks = []
        midi.fill(0) 
        for i in range(0,len(onset)):
            startIdx = onset[i]
            endIdx = offset[i]
            median = np.median(midiResult[startIdx:endIdx+1])
            midi[i] = int(median)
            startIdx = (noframe * startIdx) - ( nooverlap * (startIdx-1)) - noframe -1
            endIdx = (noframe * endIdx) - ( nooverlap * (endIdx-1)) -1
            tickidx = np.array([startIdx, endIdx])
            tickidx = (tickidx/sr) * 100 #convert to ms
            tickidx = tickidx.astype(int)
            ticks.append(tickidx)
        return midi, ticks
        
            
        
    def pitchTracking_autocorrelation(self, wavData,sr, hflag):
        #Setting
        autoCorrelationThres = 0.4
        minF = 27.5 #minFreq 27.5 Hz
        maxF = 1000.0 #maxFreq 1000.0 Hz
        minP = int(sr/maxF-1)
        maxP = int(sr/minF+1)
    
        frame_length = 30 #30 ms
        frame_overlap = 20 #20 ms
        noframe = round(frame_length  * sr / 1000)
        nooverlap = round(frame_overlap  * sr / 1000)
        window = np.hamming(noframe)
    
        wavData= self.preemphasis(wavData)
        frames =self.cut2Frames(wavData, noframe, nooverlap) 
        clippedFrames = self.clippingProcess(frames)
        windowedFrames = self.windowing(clippedFrames,window)
        onset,offset = self.onsetDetection(windowedFrames,hflag)
        f0 = self.pitchDetection(windowedFrames,minP, maxP,autoCorrelationThres,sr)      
        midiResult = self.freq_to_midi(f0)
        finalMidiResult, ticks= self.midiPostProcessing(midiResult, onset, offset, sr,noframe, nooverlap)
#        plt.figure() 
#        plt.subplot(2, 1, 1)
#        plt.plot(wavData)
#        plt.subplot(2, 1, 2)
#        plt.plot(finalMidiResult)
        return finalMidiResult, ticks
        
    def __init__(self):
        # self.open_db(dbname)
        self.sampleRate = 44100
    
    def convert(self, filename, hflag):
        if hflag==0:
            stream,sr = self.open("wavFile/" + filename + ".wav")
        else:
            stream,sr = self.open("hummFile/" + filename + ".wav")
        self.sampleRate = sr
        freqArray, tick_ms = self.pitchTracking_autocorrelation(stream,sr,hflag)
        self.create_midi(freqArray,tick_ms,filename,hflag)
        print(filename + " was converted")