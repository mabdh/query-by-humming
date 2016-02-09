
# coding: utf-8

# In[21]:

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal


# In[137]:

def EstimatePeriod( x, n, minP, maxP, q ):
    nac = np.empty(maxP+2)
    x = np.asarray(x)
    for p in range(minP-1,maxP+2):
        ac = 0.0
        sumSqBeg = 0.0
        sumSqEnd = 0.0
         
        for i in range (0,n-p):
            ac += x[i]*x[i+p]
            sumSqBeg += x[i]*x[i]
            sumSqEnd += x[i+p]*x[i+p]
        nac[p] = ac / np.sqrt( sumSqBeg * sumSqEnd )
    
    bestP = np.amax([minP,np.argmax(nac)])
    
    if(nac[bestP]<nac[bestP-1] and nac[bestP]<nac[bestP+1]):
        return (0.0, 0)
    
    q = nac[bestP]

    mid = nac[bestP]
    left = nac[bestP-1]
    right = nac[bestP+1]
    shift = 0.5*(right-left) / ( 2*mid - left - right )
    pEst = bestP + shift

    k_subMulThreshold = 0.90
    maxMul = int(bestP / minP)

    found = False
    mul = maxMul
    while((not found) and mul>=1):
        subsAllStrong = True
        for k in range(1,mul):
            subMulP = int(k*pEst/mul+0.5)
            if ( nac[subMulP] < k_subMulThreshold * nac[bestP] ):
                subsAllStrong = False
        if(subsAllStrong):
            found = True
            pEst = pEst / mul
        mul -=1
    
    return pEst, q


# In[139]:

eps = 1e-50
sr = 44100
minF = 27.5
maxF = 4186.0
minP = int(sr/maxF-1)
maxP = int(sr/minF+1)

A440 = 440.0
#f = A440 * np.power(2.0,-9.0/12.0)
f = A440
p = sr/f
print(p)
n = 2*maxP
q = 1
x = np.empty(n)
for k in range(0,n):
    x[k] = 0
    x[k] += 1.0*np.sin(2*np.pi*1*k/p)
    x[k] += 0.6*np.sin(2*np.pi*2*k/p)
    x[k] += 0.3*np.sin(2*np.pi*3*k/p)

(pEst,q) = EstimatePeriod( x, n, minP, maxP, q )
pEst = np.max([pEst,eps])
fEst = eps
if ( pEst > 0 ):
    fEst = sr/pEst
    
print( "Actual freq:         %8.3lf\n"% f );
print( "Estimated freq:      %8.3lf\n"% float(fEst) );
print( "Error (cents):       %8.3lf\n"% float(100*12*np.log(fEst/f)/np.log(2)) );
print( "Periodicity quality: %8.3lf\n"% q );


# In[ ]:




# In[ ]:




# In[ ]:



