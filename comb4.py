import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#import matplotlib.mlab as mlab

def mainfn():
    print "In comb4.py"

    #Read the file
    f = wavfile.read('/home/akshay/anaconda/ModulesPython/MUSICGEVD/sep_0.wav')
    rdata1 = f[1]
    fs = f[0]
    print "Sampling frequency: " + str(fs)
    print "shape of rdata1: ", str(rdata1.shape)
    #rdata1 = np.matrix(rdata1)
    #print "shape of rdata1: ", str(rdata1.shape)
    g = wavfile.read('/home/akshay/anaconda/ModulesPython/MUSICGEVD/sep_1.wav')
    rdata2 = g[1]
    #rdata2 = np.matrix(rdata2)
    h = wavfile.read('/home/akshay/anaconda/ModulesPython/MUSICGEVD/sep_2.wav')
    rdata3 = h[1]
    #rdata3 = np.matrix(rdata3)
    i = wavfile.read('/home/akshay/anaconda/ModulesPython/MUSICGEVD/sep_3.wav')
    rdata4 = i[1]
    #rdata4 = np.matrix(rdata4)
    
    rdata = np.array([[rdata1],[rdata2],[rdata3],[rdata4]])
    rdatat = rdata[:,0,:]
    print "Shape of rdatat: ", str(rdatat.shape)
    return(rdatat)
    
 
if __name__ == '__main__':
    rdata = mainfn()
