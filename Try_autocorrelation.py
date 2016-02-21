# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:47:38 2016

@author: maureen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from subprocess import Popen, PIPE


def openWAV(filename, sampleRate=44100):
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

def freq_to_midi(freqArray):
    return [int(round(69 + 12 * np.log2(f / 440.0))) for f in freqArray]


def dataDummy():
    sr = 44100
    minF = 27.5
    maxF = 4186.0
    minP = int(sr/maxF-1)
    maxP = int(sr/minF+1)
    
    A440 = 440.0
    f = A440 * np.power(2.0,-9.0/12.0)
    f2 = A440
    p = sr/f
    p2 = sr/f2
    n = 2*maxP
    n2 = 2*maxP
    q = 1
    x = np.empty(n)
    for k in range(0,n):
        x[k] = 0
        x[k] += 1.0*np.sin(2*np.pi*1*k/p)
        x[k] += 0.6*np.sin(2*np.pi*2*k/p)
        x[k] += 0.3*np.sin(2*np.pi*3*k/p)
    x2 = np.empty(n2)
    for k in range(0,n2):
        x2[k] = 0
        x2[k] += 1.0*np.sin(2*np.pi*1*k/p2)
        x2[k] += 0.6*np.sin(2*np.pi*2*k/p2)
        x2[k] += 0.3*np.sin(2*np.pi*3*k/p2)
    
    #fEst = calculatePitch(x2,n2,minP,maxP,q)
        
    #print( "Actual freq:         %8.3lf\n"% f2 )
    #print( "Estimated freq:      %8.3lf\n"% float(fEst) )
    #print( "Error (cents):       %8.3lf\n"% float(100*12*np.log(fEst/f2)/np.log(2)) )
    #print( "Periodicity quality: %8.3lf\n"% q )    
        
        
    x3 = np.concatenate((x,x2),axis=0)
    n3 = n + n2
    return x3,n3,minP,maxP,sr,q

def cut2Frames(x, noframe, nooverlap):
    i=0
    y=[]
    while(i+noframe<len(x)):
        data = x[int(i):int(i+noframe)]
        y.append(data)
        i += noframe - nooverlap
    return y 



def acf(x):
    result = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] \
        for i in range(1, len(x))])
    return result[:int(result.size/2)]
    
def preemphasis(x):
    y=np.empty(len(x))
    alpha = 0.95
    y[0]=x[0]
    for n in range(1,len(x)):
        y[n] = x[n] - alpha*x[n-1]
    return y
    
def clipping(x):
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
        
if __name__ == "__main__":
    
#Open WAV
    dataDir = "/Users/maureen/Documents/Study/RWTH_Aachen/LabMultimedia/QueryByHumming/QbH/qbhmain/hummFile/"
    wavFile = dataDir + 'C4_humm.wav'
    wavData,sr = openWAV(wavFile)
#    wavData,n,minP,maxP,sr,q = dataDummy()

#Setting
    eps = 1e-50
    autoCorrelationThres = 0.4
    minF = 27.5 #minFreq 27.5 Hz
    #maxF = 4186.0
    maxF = 1000.0 #maxFreq 1000.0 Hz
    minP = int(sr/maxF-1)
    maxP = int(sr/minF+1)

    frame_length = 30 #30 ms
    frame_overlap = 20 #20 ms
    noframe = round(frame_length  * sr / 1000)
    nooverlap = round(frame_overlap  * sr / 1000)
    window = np.hamming(noframe)



    wavData= preemphasis(wavData)
    f0 = []
    windowingResult =cut2Frames(wavData, noframe, nooverlap) 
    for i in range(0,len(windowingResult)):

        segment = windowingResult[i]
        segment = clipping(segment)
        segment = segment * window
        result=acf(segment)        
        idx=np.argmax(result[minP-1:maxP+1])

        oriIdx = minP+idx-1
        q = result[oriIdx]
        if(result[oriIdx]<result[oriIdx-1] and result[oriIdx] < result[oriIdx+1]) or (q<autoCorrelationThres):
            fundamentalFreq=eps
        else:
            mid  = result[oriIdx]
            left = result[oriIdx-1]
            right = result[oriIdx+1]
            shiftedIdx = oriIdx
            
            if(mid*2 - left - right >0):
                shift = 0.5*(right-left) + (mid*2 - left - right)
                shiftedIdx = oriIdx + shift
            fundamentalFreq = sr/shiftedIdx
            f0.append(fundamentalFreq)
      
    midiResult = freq_to_midi(f0)
    #for i in range(0,len(f0)):
    #    print(f0[i], midiResult[i] )
    plt.plot(f0)
    plt.figure()
    plt.plot(midiResult)
    plt.show()
    
    
#def autocorr(x):
#    result = np.correlate(x, x, mode='full')
#    return result[int(result.size/2):]
#
#def octaveDetection(shiftedIdx, oriIdx,minP,result):
#    k_subMulThreshold = 0.90
#    maxMul = int(round(oriIdx / minP))
#    found = False
#    mul = maxMul
#    while((mul>=1) and (not found)):
#        subsAllStrong = True
#        for k in range(1,mul):
#            subMulP = int(k*shiftedIdx/mul+0.5)
#            if ( result[subMulP] < k_subMulThreshold * result[oriIdx] ):
#                subsAllStrong = False
#        if ( subsAllStrong):
#            found = True
#            shiftedIdx = shiftedIdx / mul
#        mul-=1
#    return shiftedIdx
#    
 
    
    
    
#    result = autocorr(wavData)
#    m=np.max(result[ms2:ms20])
#    idx=np.argmax(result[ms2:ms20])
#    f0 = sr/(ms2+idx-1)
#    print(f0)
#    
#    #x, y = np.random.randn(2, 100)
#    fig = plt.figure()
#    ax1 = fig.add_subplot(211)
#    ax1.xcorr(wavData,wavData, usevlines=True, maxlags=sr/50, normed=True, lw=2)
#    ax1.grid(True)
#    ax1.axhline(0, color='black', lw=2)
#
#    ax2 = fig.add_subplot(212, sharex=ax1)
#    (lags, c, line, b) = ax2.acorr(wavData, usevlines=True, normed=True, maxlags=sr/50, lw=2)
#    ax2.grid(True)
#    ax2.axhline(0, color='black', lw=2)
#        
#    plt.show() 
#
#    result = c[int(np.floor(len(c)/2)):]
#    plt.plot(result)
#    result = autocorr(wavData)
#    m=np.max(result[ms2:ms20])
#    idx=np.argmax(result[ms2:ms20])
#    f0 = sr/(ms2+idx-1)
#    print(f0)
    
    
    
                #shiftedIdx2 = octaveDetection(shiftedIdx,oriIdx,minP,result)
#            if(shiftedIdx!=shiftedIdx2):
#                #plt.figure()
#                a = sorted(range(len(result)), key=lambda k: result[k],reverse=True)
#                print(a[0:10])            
#                #plt.plot(result[minP-1:maxP+1])
#                print(sr/shiftedIdx,sr/shiftedIdx2,q)