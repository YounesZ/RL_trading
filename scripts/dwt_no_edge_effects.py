import pywt
import numpy as np
import matplotlib.pyplot as plt
from scripts.dwt_multiresolution import dwt_multiresolution


def dwt_no_edge_effects(data, padding_type='mirror', pad_size=0.5, wtype='sym2', nlevels=5):
    # --- PAD SIGNAL
    dataL   =   len(data)
    if type(pad_size) is float:
        # relative window size
        pad_size    =   int(pad_size * dataL)
    if padding_type is 'mirror':
        data_p      =   (data[:pad_size][::-1], data[-pad_size:][::-1])
    elif padding_type is 'zeros':
        data_p      =   (np.zeros(pad_size), np.zeros(pad_size))
    data_p  =   np.vstack( (data_p[0], data, data_p[1]) )

    # --- COMPUTE DWT
    dwt     =   dwt_multiresolution(data_p, wtype, nlevels)

    # --- UN-PAD SIGNAL
    padsz   =   np.divide(dataL, np.power(2, range(2, nlevels + 2)))
    padsz   =   np.insert(padsz, -1, padsz[-1])
    dwt     =   np.vstack( [x[int(y):-int(y)] for x,y in zip(dwt, padsz)] )
    assert len(dwt)==len(data)
    return dwt


def test_dwt_no_edge_effects():
    # generate random signal
    sig     =   np.random.random(1000)
    # Decompose
    dwt_m   =   dwt_no_edge_effects(sig, 'mirror')
    sig_m   =   np.sum(dwt_m, axis=0)
    dwt_z   =   dwt_no_edge_effects(sig, 'zeros')
    sig_z   =   np.sum(dwt_z, axis=0)
    # Plot
    plt.figure()
    plt.plot(sig, label='original signal')
    plt.plot(sig_m, '--r', label='Mirror padding')
    plt.plot(sig_z, '--g', label='Zero padding')
    plt.legend()


if __name__=='__main__':
    test_dwt_no_edge_effects()
