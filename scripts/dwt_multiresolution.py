import pywt
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy



def dwt_multiresolution(data, wtype='sym2', nlevels=5, output='coeff'):
    # Init
    w   =   pywt.Wavelet(wtype)
    a   =   deepcopy(data)
    ca  =   []
    cd  =   []
    if output=='coeff':
        for i in range(nlevels):
            (a, d)  =   pywt.dwt(a, w, mode='per', axis=0)
            cd.append(d)
        cd.append(a)
        out =   cd
    else:
        for i in range(nlevels):
            (a, d)  =   pywt.dwt(a, w, mode='per', axis=0)
            cd.append(d)
            cd.append(a)

        # iWavelet transform: wavelet coefficients
        out =   []
        for i, coeff in enumerate(cd):
            coeff_list  =   [None, coeff.T] + [None] * i
            out.append(pywt.waverec(coeff_list, w)[:len(data)])
        # iWavelet transform: scaling coefficients
        coeff_list      =   [ca[-1].T, None] + [None] * (nlevels - 1)
        out.append(pywt.waverec(coeff_list, w)[:len(data)])
    return out


def test_dwt_multiresolution():
    sig = np.random.random(1000)
    dwt = dwt_multiresolution(sig, nlevels=10)
    plt.figure();
    plt.plot(sig, label='original signal')
    plt.plot( np.sum(dwt, axis=0), '--r', label='reconstructed')
    #[plt.plot(y, label='psi_{}'.format(x)) for x,y in enumerate(dwt[:-1])]
    #plt.plot(dwt[-1], label='phi_{}'.format(10))
    plt.legend()


# LAUNCHER
if __name__=='__main__':
    test_dwt_multiresolution()
