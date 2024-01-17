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
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)
    rcounts[0], displs[0] = 0, 0
    
    for k in range(1, numprocs):
        if k < res + 1:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k-1] + rcounts[k-1]
else:
    rcounts, displs = None, None

M_part = np.array(0, dtype=np.int32)

comm.Scatter([rcounts, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

if rank == 0:
    f2 = open('AData.dat', 'r')
    for k in range(1, numprocs):
        A_part = np.empty((rcounts[k], N), dtype=np.float64)
        for j in range(rcounts[k]):
            for i in range(N):
                A_part[j,i] = np.float64(f2.readline())
        comm.Send([A_part, rcounts[k]*N, MPI.DOUBLE], dest=k, tag=0)
    f2.close()
    A_part = np.empty((M_part, N), dtype=np.float64)
else:
    A_part = np.empty((M_part, N), dtype=np.float64)
    comm.Recv([A_part, M_part*N, MPI.DOUBLE], source=0, tag=0, status=None)

if rank == 0:
    x = np.empty(M, dtype=np.int32)
    f3 = open('xData.dat', 'r')
    for j in range(M):
        x[j] = np.float64(f3.readline())
    f3.close()
else:
    x = None

x_part = np.empty(M_part, dtype=np.float64)
comm.Scatterv([x, rcounts, displs, MPI.DOUBLE],
              [x_part, M_part, MPI.DOUBLE], root=0)

b_temp = np.array(np.dot(A_part.T, x_part), dtype=np.float64)

if rank==0:
    b = np.empty(N, dtype=np.float64)
else:
    b = None

comm.Reduce([b_temp, N, MPI.DOUBLE],
            [b, N, MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:
    print(b)