import numpy as np

def interpolation_matrix(ni,ne):
    Vi = np.fft.ifft(np.eye(ni))*ni
    xe = 2*np.pi*np.arange(ne)/ne 
    Ve = np.array([np.exp(1j*k*xe) for k in np.fft.fftfreq(ni)*ni]).T
    I = np.real(np.fft.fft(Ve.T,axis=0).T)/ni
    return I

def finterp1(fi,x):
    ni = len(fi)
    fhat = np.fft.fft(fi) / ni
    kv = np.fft.fftfreq(ni) * ni
    return np.real( sum( fhat[i]*np.exp(1j*k*x) for i,k in enumerate(kv) ) )
 
