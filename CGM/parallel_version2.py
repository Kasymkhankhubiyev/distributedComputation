import numpy as np
import sys
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def conjugate_gradient_method(A_part, b_part, x, N) :
    s = 1
    p = np.zeros(N, dtype=np.float64)
    while s <= N :
        if s == 1 :
            # paralleling the following code block:
            # r = dot(A_part.T, dot(A_part,x) - b_part)

            r_temp = np.dot(A_part.T, np.dot(A_part, x) - b_part)
            r = np.empty(N, dtype=np.float64)
            comm.Allreduce(r_temp, r, op=MPI.SUM)
        else :
            r = r - q / np.dot(p, q)
        p = p + r / np.dot(r, r)

        # paralleling the following code block
        # q = dot(A_part.T, dot(A_part, p))
        q_temp = np.dot(A_part.T, np.dot(A_part, p))
        q = np.empty(N, dtype=np.float64)
        comm.Allreduce(q_temp, q, op=MPI.SUM)

        x = x - p/np.dot(p,q)
        s = s + 1
    return x

# Считываем из файла число строк M и число столбцов N
if rank == 0:
    f1 = open('in.dat', 'r')
    N = np.array(int(f1.readline()), dtype=np.int32)
    M = np.array(int(f1.readline()), dtype=np.int32)
    f1.close()
else:
    N = np.array(0, dtype=np.int32)
    M = None

comm.Bcast(N, root=0)

def auxiallary_arrays(M: int, numprocs: int):
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)

    displs[0] = 0

    ave, res = divmod(M, numprocs)

    for k in range(numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k >= 1:  # here was the difference
            displs[k] = displs[k-1] + rcounts[k-1]

    return rcounts, displs


if rank == 0:
    rcounts_M, displs_M = auxiallary_arrays(M, numprocs)
    rcounts_N, displs_N = auxiallary_arrays(N, numprocs)
else:
    rcounts_M, displs_M = None, None
    rcounts_N, displs_N = np.empty(numprocs, dtype=np.int32), None

M_part = np.array(0, dtype=np.int32)
N_part = np.array(0, dtype=np.int32)

comm.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)
comm.Scatter([rcounts_N, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)
comm.Bcast(rcounts_N, root=0)

if rank == 0:
    # Считываем из файла матрицу A
    f2 = open('AData.dat', 'r')
    A_part = np.empty((M_part, N), dtype=np.float64)
    for j in range(M_part):
        for i in range(N):
            A_part[j,i] = float(f2.readline())

    for k in range(1, numprocs):
        A_part_temp = np.empty((rcounts_M[k] ,N), dtype=np.float64)
        for j in range(rcounts_M[k]):
            for i in range(N):
                A_part_temp[j,i] = float(f2.readline())
        comm.Send(A_part_temp, dest=k)
        del(A_part_temp)
    f2.close()
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
else:
    b = None

b_part = np.empty(M_part, dtype=np.float64)

comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], 
              [b_part, M_part, MPI.DOUBLE], root=0)

x = np.zeros(N, dtype=np.float64)

# x = conjugate_gradient_method(A_part, b, x, N, M, 
#                               M_part, N_part, comm,
#                               rcounts_M, rcounts_N,
#                               displs_M, displs_N)

x = conjugate_gradient_method(A_part, b_part, x, N)

if rank == 0:
    fig = plt.figure()
    ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(np.arange(N), x, '-r', lw=3)

    plt.show()