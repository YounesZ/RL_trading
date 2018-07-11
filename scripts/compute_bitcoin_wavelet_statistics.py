import numpy as np
import matplotlib.pyplot as plt

from os import path
from src.sampler import BTCsampler
from src.emulator import Market



def main():
    """
    This function computes the wavelet transform over non-overlapping time windows across the bitcoin dataset to identify
    coefficients that can be shrinked and removed from the state space
    """

    # Database: set options and create
    Sampler     =   BTCsampler
    db_type     = 	'BTCsampler';
    db          =   'db_bitcoin.pickle'
    fld         =   path.join('..', 'data', db_type, db)
    wavChan     =   4
    window_state=   32
    sampler     =   Sampler(True, fld=fld, variables=['Close'], wavelet_channels=wavChan,
                            window_training_episode=window_state, window_testing_episode=window_state)

    # Wavelet transform
    # --- With time difference
    envTD       =   Market(sampler, window_state, 0, time_difference=True, wavelet_channels=wavChan)
    stateTD     =   []
    # --- Without time difference
    envRAW      =   Market(sampler, window_state, 0, time_difference=False, wavelet_channels=wavChan)
    stateRAW    =   []
    # --- Loop and compute
    for il in range(400):
        stTD,_  =   envTD.reset()
        stRW,_  =   envRAW.reset()
        stateTD +=  [stTD]
        stateRAW+=  [stRW]

    # Compute descriptive stats
    stateTD     =   np.array(stateTD)
    stateRAW    =   np.array(stateRAW)

    # --- Mean
    F   =   plt.figure()
    Ax1 =   F.add_subplot(121)
    Ax1.plot( np.mean(stateRAW, axis=0) )
    Ax1.set_xlabel('Wavelet coefficient ID')
    Ax1.set_ylabel('Average magnitude')
    Ax1.set_title('WT on RAW signal')
    Ax2 = F.add_subplot(122)
    Ax2.plot(np.mean(stateTD, axis=0))
    Ax2.set_xlabel('Wavelet coefficient ID')
    Ax2.set_ylabel('Average magnitude')
    Ax2.set_title('WT on T-D signal')

    # --- STD
    F = plt.figure()
    Ax1 = F.add_subplot(121)
    Ax1.plot(np.std(stateRAW, axis=0))
    Ax1.set_xlabel('Wavelet coefficient ID')
    Ax1.set_ylabel('Standard deviation')
    Ax1.set_title('WT on RAW signal')
    Ax2 = F.add_subplot(122)
    Ax2.plot(np.std(stateTD, axis=0))
    Ax2.set_xlabel('Wavelet coefficient ID')
    Ax2.set_ylabel('Standard deviation')
    Ax2.set_title('WT on T-D signal')


if __name__ == '__main__':
    main()
