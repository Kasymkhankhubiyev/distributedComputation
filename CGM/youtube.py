import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()


def conjugate_gradient_method(A_part: np.ndarray, b_part: np.ndarray, 
                              x_part: np.ndarray, N: int, N_part: int, 
                              rcounts_N: np.ndarray, displs_N: np.ndarray) -> np.ndarray:
    
    pass

if rank == 0:
    f1 = open('in.dat', 'r')
    N = np.array(f1.readline(), dtype=np.int32)
    M = np.array(f1.readline(), dtype=np.int32)
    f1.close()
else:
    N = np.array(0, dtype=np.int32)

comm.Bcast([N, 1, MPI.DOUBLE], root=0)

def auxiallary_arrays(M: int, numprocs: int) -> 'tuple[np.ndarray, np.ndarray]':
    ave, res = divmod(M, numprocs-1)
    rcounts = np.zeros(numprocs, dtype=np.int32)
    displs = np.zeros(numprocs, dtype=np.int32)

    for k in range(1, numprocs):
        if k < 1 + res:
            rcounts[k]