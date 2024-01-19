"""
    USE the following command to run more processes
    than kernels available

    `mpiexec --map-by :oversubscribe -n 21 python MatrixMultiplication.py`
"""

import numpy as np
import sys

from matplotlib.pyplot import style, figure, axes, show
from mpi4py import MPI


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

num_rows = num_cols = np.int32(np.sqrt(numprocs))
# num_rows = num_cols = 3

def conjugate_gradient_method(A, b, x, N) :
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
    rcounts_M, displs_M = auxiallary_arrays(M, num_rows)
    rcounts_N, displs_N = auxiallary_arrays(N, num_cols)
else:
    rcounts_M, displs_M = None, None
    rcounts_N, displs_N = np.empty(num_cols, dtype=np.int32), None

M_part = np.array(0, dtype=np.int32)
N_part = np.array(0, dtype=np.int32)

# create new commutators for each row and column
# processes with the same value of `rank % num_cols`
# includes into a one group - subcommutator
comm_col = comm.Split(rank % num_cols, rank)
comm_row = comm.Split(rank // num_rows, rank)

# now we want to scatter data over a group of 
# processes of the same column
# each single communicator has info about all 
# the proccesses include into the group
# before running the code we need to send
# rcounts data to all of root processes of subcommunicators
# but at the start only the first communicator with
# a global rank equal 0 knows about rcounts
if rank in range(0, numprocs, num_cols):
    comm_col.Scatter([rcounts_M, 1, MPI.INT], 
                     [M_part, 1, MPI.INT], root=0)
    
# broadcasting data over rows
comm_row.Bcast(M_part, root=0)

if rank in range(num_cols):
    comm_row.Scatter([rcounts_N, 1, MPI.INT], 
                     [N_part, 1, MPI.INT], root=0)
    
# broadcasting data over rows
comm_col.Bcast(N_part, root=0)

print(f'for rank = {rank}: M_part = {M_part}, N_part = {N_part}')


sys.exit()
	
f2 = open('AData.dat', 'r')
for j in range(M) :
    for i in range(N) :
        A[j,i] = float(f2.readline())
f2.close()
		
f3 = open('bData.dat', 'r')
for j in range(M):
    b[j] = float(f3.readline())
f3.close()

x = np.zeros(N)

x = conjugate_gradient_method(A, b, x, N)

fig = figure()
ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
ax.set_xlabel('i'); ax.set_ylabel('x[i]')
ax.plot(np.arange(N), x, '-r', lw=3)

show()
