import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()


if rank == 0:
    f1 = open('in.dat', 'r')
    N = np.array(f1.readline(), dtype=np.int32)
    M = np.array(f1.readline(), dtype=np.int32)
    f1.close()
else:
    N = np.array(0, dtype=np.int32)

comm.Bcast([N, 1, MPI.DOUBLE], root=0)

if rank == 0:
    ave, res = divmod(M, numprocs-1)