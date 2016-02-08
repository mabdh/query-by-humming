
# coding: utf-8

# In[21]:

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal


# In[71]:

def autocorr(x, t=1):
    x = np.asarray(x)
    return np.corrcoef([x[0:len(x)-t], x[t:len(x)]])
def EstimatePeriod( x, n, minP, maxP, q ):
    nac = np.empty(maxP+1)
    x = np.asarray(x)
    for p in range(minP-1,maxP+1):
        ac = 0.0
        sumSqBeg = 0.0
        sumSqEnd = 0.0
         
        for i in range (0,n-p):
            ac += x[i]*x[i+p]
            sumSqBeg += x[i]*x[i]
            sumSqEnd += x[i+p]*x[i+p]
        nac[p] = ac / np.sqrt( sumSqBeg * sumSqEnd )
    print(len(nac))
    bestP = np.amax([minP,np.argmax(nac)])
    if(nac[bestP]<nac[bestP-1] and nac[bestP]<nac[bestP+1]):
        return (0.0, 0)
    q = nac[bestP]
    mid = nac[bestP]
    left = nac[bestP-1]
    right = nac[bestP+1]
    shift = 0.5*(right-left) / ( 2*mid - left - right )
    pEst = bestP + shift
    return (pEst,bestP)


# In[72]:

eps = 1e-50
sr = 44100
minF = 27.5
maxF = 4186.0
minP = int(sr/maxF-1)
maxP = int(sr/minF+1)

A440 = 440.0
f = A440 * np.power(2.0,-9.0/12.0)
p = sr/f
n = 2*maxP
q = 1
x = np.empty(n)
for k in range(0,n):
    x[k] = 0
    x[k] += 1.0*np.sin(2*np.pi*1*k/p)
    x[k] += 0.6*np.sin(2*np.pi*2*k/p)
    x[k] += 0.3*np.sin(2*np.pi*3*k/p)
pEst = np.max(EstimatePeriod( x, n, minP, maxP, q ),eps)
fEst = eps
if ( pEst > 0 ):
    fEst = sr/pEst
    
print( "Actual freq:         %8.3lf\n"% f );
print( "Estimated freq:      %8.3lf\n"% float(sr/pEst) );
print( "Error (cents):       %8.3lf\n"% float(100*12*np.log(fEst/f)/np.log(2)) );
print( "Periodicity quality: %8.3lf\n"% q );


# In[74]:




# In[75]:




# In[ ]:



