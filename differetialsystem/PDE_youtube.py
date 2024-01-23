import numpy as np
from matplotlib.pyplot import style, figure, axes, show
from mpi4py import MPI
# import time
import sys


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
# we want a linear topology
comm_cart = comm.Create_cart(dims=(numprocs,), periods=(False), reorder=True)
rank_cart = comm_cart.Get_rank()


# constants
N = 800; M = 300000
# N = 200; M = 20000

a = 0.; b = 1.
t_0 = 0.; T = 6.0

eps = 10**(-1.5)

h = (b - a)/N; 
x = np.linspace(a, b, N+1)
tau = (T - t_0)/M; t = np.linspace(t_0, T, M+1)


def u_init(x) :
    u_init = np.sin(3*np.pi*(x - 1/6))
    return u_init


def u_left(t) :
    u_left = -1.
    return u_left


def u_right(t) :
    u_right = 1.
    return u_right


if rank_cart == 0:
    start_time = MPI.Wtime()

a = 0.
b=1.
t_0 = 0.
T = 6.0
eps = 10**(-1.5)

N = 800
M = 300000

h = (b - a) / N
x = np.linspace(a, b, N+1)
tau = (T - t_0) / M
t = np.linspace(t_0, T, M+1)

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

if rank_cart == 0:
    rcounts, displs = auxiallary_arrays(N+1, numprocs)
else:
    rcounts, displs = None, None

N_part = np.array(0, dtype=np.int32)

comm_cart.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)


if rank_cart == 0:
    rcounts_from_0 = np.empty(numprocs, dtype=np.int32)
    displs_from_0 = np.empty(numprocs, dtype=np.int32)

    rcounts_from_0[0] = rcounts[0] + 1
    displs_from_0[0] = 0

    for k in range(1, numprocs-1):
        rcounts_from_0[k] = rcounts[k]+2
        displs_from_0[k] = displs[k] - 1

    rcounts_from_0[numprocs-1] = rcounts[numprocs-1] + 1
    displs_from_0[numprocs-1] = displs[numprocs-1] - 1
else:
    rcounts_from_0 = None
    displs_from_0 = None

N_part_aux = np.array(0, dtype=np.int32)

comm.Scatter([rcounts_from_0, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0)

if rank_cart == 0:
    u = np.empty((M+1, N+1), dtype=np.float64)
    for n in range(N+1):
        u[0, n] = u_init(x[n])
else:
    u = np.empty((M+1, 0), dtype=np.float64)


u_part = np.empty(N_part, dtype=np.float64)
u_part_aux = np.empty(N_part_aux, dtype=np.float64)


for m in range(M):

    comm_cart.Scatterv([u[m], rcounts_from_0, displs_from_0, MPI.DOUBLE],
                       [u_part_aux, N_part_aux, MPI.DOUBLE], root=0)
    
    for n in range(1, N_part_aux-1):
        u_part[n-1] = 0
    