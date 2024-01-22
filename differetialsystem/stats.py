import matplotlib.pyplot as plt
import numpy as np

arr_800_300_000_ = np.array([317.3, 169.39, 121.05, 98.77])
arr_800_250_000_ = np.array([270.89, 139.96, 99.04, 79.45])
arr_700_200_000_ = np.array([185.5, 101.12, 69.19, 56.03])

def calc_Sn(array):
    output = np.zeros_like(array) + array[0]
    output = output / array
    return output

def calc_En(array):
    output = array.copy()
    output = output / np.arange(1, len(array)+1)
    return output


Sn_800_300_000_ = calc_Sn(arr_800_300_000_)
Sn_800_250_000_ = calc_Sn(arr_800_250_000_)
Sn_700_200_000_ = calc_Sn(arr_700_200_000_)

En_800_300_000_ = calc_En(arr_800_300_000_)
En_800_250_000_ = calc_En(arr_800_250_000_)
En_700_200_000_ = calc_En(arr_700_200_000_)


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))
# axs[0].set_xlabel('n')
# axs[0].set_ylabel('second')
# axs[0].plot(np.arange(1, len(arr_800_300_000_)+1), arr_800_300_000_, color='red', label='N=800, M=300000')
# axs[0].plot(np.arange(1, len(arr_800_250_000_)+1), arr_800_250_000_, color='green', label='N=800, M=250000')
# axs[0].plot(np.arange(1, len(arr_700_200_000_)+1), arr_700_200_000_, color='blue', label='N=700, M=200000')

axs[0].set_xlabel('n')
axs[0].set_ylabel('Sn')
# axs[1].plot(np.arange(1, len(Sn_800_300_000_)+1), 
#             (np.zeros_like(Sn_800_300_000_) + Sn_800_300_000_) / np.arange(1, len(Sn_800_300_000_)+1),
#             color='orange', label='ideal')
axs[0].plot(np.arange(1, len(Sn_800_300_000_)+1), np.arange(1, len(Sn_800_300_000_)+1), 
            color='orange', label='ideal')
axs[0].plot(np.arange(1, len(Sn_800_300_000_)+1), Sn_800_300_000_, color='red', label='N=800, M=300000')
axs[0].plot(np.arange(1, len(Sn_800_250_000_)+1), Sn_800_250_000_, color='green', label='N=800, M=250000')
axs[0].plot(np.arange(1, len(Sn_700_200_000_)+1), Sn_700_200_000_, color='blue', label='N=700, M=200000')
axs[0].legend()

axs[1].set_xlabel('n')
axs[1].set_ylabel('En')
# axs[1].plot(np.arange(1, len(Sn_800_300_000_)+1), np.arange(1, len(Sn_800_300_000_)+1), 
#             color='orange', label='ideal')
axs[1].plot(np.arange(1, len(En_800_300_000_)+1), 
            (np.zeros_like(En_800_300_000_) + En_800_300_000_) / np.arange(1, len(En_800_300_000_)+1),
            color='orange', label='ideal')
axs[1].plot(np.arange(1, len(En_800_300_000_)+1), En_800_300_000_, color='red', label='N=800, M=300000')
axs[1].plot(np.arange(1, len(En_800_250_000_)+1), En_800_250_000_, color='green', label='N=800, M=250000')
axs[1].plot(np.arange(1, len(En_700_200_000_)+1), En_700_200_000_, color='blue', label='N=700, M=200000')
axs[1].legend()

plt.show()