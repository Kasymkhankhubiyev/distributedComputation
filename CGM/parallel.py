import numpy as np
import sys
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def сonjugate_gradient_method(A, b, x, N) :
    s = 1
    p = np.zeros(N)
    while s <= N :
        if s == 1 :
            r = np.dot(A.T, np.dot(A,x) - b)
        else :
            r = r - q/np.dot(p, q)
        p = p + r/np.dot(r, r)
        q = np.dot(A.T, np.dot(A, p))
        x = x - p/np.dot(p,q)
        s = s + 1
    return x

if rank == 0:
    # Считываем из файла число строк M и число столбцов N
    f1 = open('in.dat', 'r')
    N = np.array(int(f1.readline()), dtype=np.int32)
    M = np.array(int(f1.readline()), dtype=np.int32)
    f1.close()
else:
    N = np.array(0, dtype=np.int32)
    M = np.array(0, dtype=np.int32)

comm.Bcast(M, root=0)

if rank == 0:
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)

    # rcounts[0] = 0
    displs[0] = 0

    ave, res = divmod(M, numprocs)

    for k in range(0, numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k-1] + rcounts[k-1]

    print(rcounts, displs)
else:
    rcounts, displs = None, None

M_part = np.array(1, dtype=np.int32)

comm.Scatter([rcounts, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

if rank == 0:
    # Считываем из файла матрицу A
    f2 = open('AData.dat', 'r')
    A_part = np.empty((M_part, N), dtype=np.float64)
    for j in range(M_part) :
        for i in range(N) :
            A_part[j,i] = float(f2.readline())

    for k in range(1, numprocs):
        A_part_temp = np.empty((rcounts[k] ,N), dtype=np.float64)
        for j in range(rcounts[k]) :
            for i in range(N) :
                A_part_temp[j,i] = float(f2.readline())
        comm.Send(A_part_temp, dest=k)
    f2.close()
    del A_part_temp
else:
    A_part = np.empty((M_part, N), dtype=np.float64); 
    comm.Recv(A_part, source=0)

# Считываем из файла вектор b
if rank == 0:
    b = np.empty(M, dtype=np.float64)
    f3 = open('bData.dat', 'r')
    for i in range(M):
        b[i] = float(f3.readline())
    f3.close()

comm.Bcast(b, root=0)

x = np.zeros(N)

x = сonjugate_gradient_method(A_part, b, x, N)

fig = plt.figure()
ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
ax.set_xlabel('i'); ax.set_ylabel('x[i]')
ax.plot(np.arange(N), x, '-r', lw=3)

plt.show()