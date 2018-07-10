# Import modules
import matplotlib.pyplot as plt
import numpy as np
from scripts.dwt_windowed import do_transform


def py_closest(data, value):
    return np.argmin(np.abs(data - value))


def py_cumvar_n(data):
    return np.cumsum(data**2) / np.sum(data**2)



## RANDOM INPUT
# Generate test signal
sig = np.random.random([2**16])
plt.figure();
plt.plot(sig)

# Apply DWT
dwt =   do_transform(sig)
cols=   dwt.columns
FIG =   plt.figure()
ax1 =   FIG.add_subplot(611); ax1.plot(dwt[cols[0]])
ax2 =   FIG.add_subplot(612); ax2.plot(dwt[cols[1]])
ax3 =   FIG.add_subplot(613); ax3.plot(dwt[cols[2]])
ax4 =   FIG.add_subplot(614); ax4.plot(dwt[cols[3]])
ax5 =   FIG.add_subplot(615); ax5.plot(dwt[cols[4]])
ax6 =   FIG.add_subplot(616); ax6.plot(dwt[cols[5]])



## TRAILING PULSE
# Generate test signal
sig = np.zeros([2**16])
sig[-1] = 1
plt.figure();
plt.plot(sig)

# Apply DWT
nlv =   10
dwt =   do_transform(sig, wtype='sym2', nlevels=nlv)
cols=   dwt.columns
FIG =   plt.figure()
AX  =   []
for il in range(nlv):
    if nlv<6:
        AX  +=  [FIG.add_subplot(nlv+1, 1, il+1)];
        AX[il].plot(dwt[cols[il]]);
        AX[il].set_xlim([2**16-120, 2**16])
        AX[il].set_xticks([])
    else:
        AX  +=  [FIG.add_subplot(5, 2, il+1)];
        AX[il].plot(dwt[cols[il]]);
        AX[il].set_xlim([2**16-240, 2 ** 16]);
        AX[il].set_title(cols[il])
        AX[il].set_xticks([])
AX[-2].set_xticks([2**16-240, 2**16-120, 2**16])
AX[-2].set_xticklabels(['-4', '-2', '0'])
AX[-2].set_xlabel('Time (h)')
AX[-1].set_xticks([2**16-240, 2**16-120, 2**16])
AX[-1].set_xticklabels(['-4', '-2', '0'])
AX[-1].set_xlabel('Time (h)')


# Check variance threshold
thresh  =   [0.05, 0.1, 0.15, 0.2]
scale   =   list( range(10) )
TABLE   =   np.zeros([10, len(thresh)])
for ix, it in enumerate(thresh):
    for isc in scale:
        TABLE[isc, ix]  =   2**16 - py_closest( py_cumvar_n(dwt[cols[isc]]), it )

p1  =   plt.bar(range(10), TABLE[:,0], 0.35)
p2  =   plt.bar(range(10), TABLE[:,1]-TABLE[:,0], 0.35, bottom=TABLE[:,0])
p3  =   plt.bar(range(10), TABLE[:,2]-TABLE[:,1], 0.35, bottom=TABLE[:,1])
p4  =   plt.bar(range(10), TABLE[:,3]-TABLE[:,2], 0.35, bottom=TABLE[:,2])

plt.ylabel('edge effect duration (min)')
plt.title('Effects of wavelet scale and variance threshold')
plt.xticks(range(10), cols[:-1])
plt.legend( (p1[0], p2[0], p3[0], p4[0]), ('5% threshold', '10%', '15%', '20%') )
plt.show()
plt.plot([-0.5, 9.5], [30, 30], '--r')
plt.plot([-0.5, 9.5], [60, 60], '--r')
plt.xlim([-0.5, 9.5])
plt.text(1, 32, '30 minutes', color='r')
plt.text(2, 62, '60 minutes', color='r')