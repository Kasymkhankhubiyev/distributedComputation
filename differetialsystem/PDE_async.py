import numpy as np
from matplotlib.pyplot import style, figure, axes, show
from mpi4py import MPI
import time
import sys


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
# we want a linear topology
comm_cart = comm.Create_cart(dims=(numprocs,), periods=(False), reorder=True)
rank_cart = comm_cart.Get_rank()


# constants
# N = 800; M = 300000
N = 200; M = 20000

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

rcounts, displs = auxiallary_arrays(N+1, numprocs)

N_part = rcounts[rank_cart]

if rank_cart == 0:
    N_part_aux = np.array(N_part + 1, dtype=np.int32)
    displ_aux = np.array(displs[rank_cart], dtype=np.int32)
elif rank_cart == numprocs-1:
    N_part_aux = np.array(N_part + 1, dtype=np.int32)
    displ_aux = np.array(displs[rank_cart] - 1,dtype=np.int32)
else:
    N_part_aux = np.array(N_part + 2, dtype=np.int32)
    displ_aux = np.array(displs[rank_cart] - 1, dtype=np.int32)

if rank_cart == 0:
    rcounts_aux = np.empty(numprocs, dtype=np.int32)
    displs_aux = np.empty(numprocs, dtype=np.int32)
else:
    rcounts_aux, displs_aux = None, None

comm_cart.Gather(N_part_aux, rcounts_aux, root=0)
comm_cart.Gather(displ_aux, displs_aux, root=0)

u_part_aux = np.empty((M+1, N_part_aux), dtype=np.float64)

if rank_cart == 0:
    start_time = MPI.Wtime() # time.time()

for n in range(N_part_aux) :
    # print(u_part_aux[0, n])
    u_part_aux[0, n] = u_init(x[n + displ_aux])
    # print(u_init(x[n + displ_aux]))

    
if rank_cart == 0:
    for m in range(1, M + 1) :
        u_part_aux[m, 0] = u_left(t[m])
elif rank_cart == numprocs - 1:
    for m in range(1, M+1):
        u_part_aux[m, N_part_aux-1] = u_right(t[m])
    
requests = [MPI.Request() for i in range(4)]

for m in range(M):

    # TODO: улучшение - выставляем сразу все асинхронные отправления и приемы,
    # выполняем серединный подсчет, а в конце с ожиданием запускаем подсчет крайних
    if rank_cart == 0:
        
        requests[0] = comm_cart.Isend([u_part_aux[m, N_part_aux-2:], 1, MPI.DOUBLE], dest=1, tag=0)
        requests[1] = comm_cart.Irecv([u_part_aux[m, N_part_aux-1:], 1, MPI.DOUBLE], source=1, tag=MPI.ANY_TAG)

    elif rank_cart == numprocs-1:
        
        requests[0] = comm_cart.Isend([u_part_aux[m, 1:], 1, MPI.DOUBLE], dest=numprocs-2, tag=0)
        requests[1] = comm_cart.Irecv([u_part_aux[m, 0:], 1, MPI.DOUBLE], source=numprocs-2, tag=MPI.ANY_TAG)

    else:
        
        requests[0] = comm_cart.Isend([u_part_aux[m, 1:], 1, MPI.DOUBLE], dest=rank_cart-1, tag=0)
        requests[1] = comm_cart.Irecv([u_part_aux[m, 0:], 1, MPI.DOUBLE], source=rank_cart-1, tag=MPI.ANY_TAG)

        requests[2] = comm_cart.Isend([u_part_aux[m, N_part_aux-2:], 1, MPI.DOUBLE], dest=rank_cart+1, tag=0)
        requests[3] = comm_cart.Irecv([u_part_aux[m, N_part_aux-1:], 1, MPI.DOUBLE], source=rank_cart+1, tag=MPI.ANY_TAG)

    for n in range(2, N_part_aux - 2):

        left_part = u_part_aux[m, n] + eps * tau * (u_part_aux[m,n+1] - 2 * u_part_aux[m,n] + u_part_aux[m, n-1]) / h ** 2
        right_part = tau * u_part_aux[m, n] * (u_part_aux[m, n+1] - u_part_aux[m, n-1]) / (2 * h) + tau * u_part_aux[m, n] ** 3
        u_part_aux[m+1, n] = left_part + right_part

    MPI.Request.Waitall(requests)

    for n in [1, N_part_aux - 2]:

        left_part = u_part_aux[m, n] + eps * tau * (u_part_aux[m,n+1] - 2 * u_part_aux[m,n] + u_part_aux[m, n-1]) / h ** 2
        right_part = tau * u_part_aux[m, n] * (u_part_aux[m, n+1] - u_part_aux[m, n-1]) / (2 * h) + tau * u_part_aux[m, n] ** 3
        u_part_aux[m+1, n] = left_part + right_part


if rank_cart == 0:
    end_time = MPI.Wtime()# time.time()

    print(f'Elapsed time for {numprocs} processes is {end_time-start_time:.4f} sec')

if rank_cart == 0:
    u = np.empty(N+1, dtype=np.float64)
else:
    u = None

comm_cart.Gatherv([u_part_aux[m, :], N_part_aux, MPI.DOUBLE],
                  [u, rcounts_aux, displs_aux, MPI.DOUBLE], root=0)

if rank_cart == 0:
    fig = figure()
    ax = axes(xlim=(a,b), ylim=(-2.0, 2.0))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x, u, color='r', ls='-', lw=2)
    show()
