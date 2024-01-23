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
displ_aux = np.array(0, dtype=np.int32)

comm.Scatter([rcounts_from_0, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0)
comm.Scatter([displs_from_0, 1, MPI.INT], [displ_aux, 1, MPI.INT], root=0)

u_part_aux = np.empty((M+1, N_part_aux), dtype=np.float64)


for n in range(N_part_aux):
    u_part_aux[0, n] = u_init(x[displ_aux+n])
if rank_cart == 0:
    for m in range(1, M+1):
        u_part_aux[m, 0] = u_left(t[m])
if rank_cart == numprocs-1:
    for m in range(1, M+1):
        u_part_aux[m, N_part_aux-1] = u_right(t[m])


for m in range(M):

    for n in range(1, N_part_aux-1):
        left_part = u_part_aux[m, n] + eps * tau * (u_part_aux[m,n+1] - 2 * u_part_aux[m,n] + u_part_aux[m, n-1]) / h ** 2
        right_part = tau * u_part_aux[m, n] * (u_part_aux[m, n+1] - u_part_aux[m, n-1]) / (2 * h) + tau * u_part_aux[m, n] ** 3
        u_part_aux[m+1, n] = left_part + right_part

    if rank_cart == 0:
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, N_part_aux-2:], 1, MPI.DOUBLE], dest=1, sendtag=0,
                           recvbuf=[u_part_aux[m+1, N_part_aux-1:], 1, MPI.DOUBLE], source=1, recvtag=MPI.ANY_TAG,
                           status=None)
    elif rank_cart == numprocs-1:
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, 1:], 1, MPI.DOUBLE], dest=numprocs-2, sendtag=0,
                           recvbuf=[u_part_aux[m+1, 0:], 1, MPI.DOUBLE], source=numprocs-2, recvtag=MPI.ANY_TAG,
                           status=None)
    else:
        # # обмен слева
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, 1:], 1, MPI.DOUBLE], dest=rank_cart-1, sendtag=0,
                           recvbuf=[u_part_aux[m+1, 0:], 1, MPI.DOUBLE], source=rank_cart-1, recvtag=MPI.ANY_TAG,
                           status=None)
        
        # обмен справа
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m+1, N_part_aux-2:], 1, MPI.DOUBLE], dest=rank_cart+1, sendtag=0,
                           recvbuf=[u_part_aux[m+1, N_part_aux-1:], 1, MPI.DOUBLE], source=rank_cart+1, recvtag=MPI.ANY_TAG,
                           status=None)
        

if rank_cart == 0:
    u_T = np.empty(N+1, dtype=np.float64)
else:
    u_T = None

if rank_cart == 0:
    comm_cart.Gatherv([u_part_aux[M, 0:N_part_aux], N_part_aux, MPI.DOUBLE],
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)
if rank_cart in range(1, numprocs-1):
    comm_cart.Gatherv([u_part_aux[M, 1:N_part_aux-1], N_part_aux, MPI.DOUBLE],
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)
if rank_cart == numprocs - 1:
    comm_cart.Gatherv([u_part_aux[M, 1:N_part_aux], N_part_aux, MPI.DOUBLE],
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)