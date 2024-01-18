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
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k-1] + rcounts[k-1]

    return rcounts, displs

if rank == 0:
    rcounts_M, displs_M = auxiallary_arrays(M, numprocs)
    rcounts_N, displs_N = auxiallary_arrays(N, numprocs)
else:
    rcounts_M, displs_M = None, None
    rcounts_N = np.empty(numprocs, dtype=np.int32)
    displs_N = np.empty(numprocs, dtype=np.int32)

M_part = np.array(0, dtype=np.int32)
N_part = np.array(0, dtype=np.int32)

comm.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

comm.Bcast([rcounts_N, numprocs, MPI.INT], root=0)
comm.Bcast([displs_N, numprocs, MPI.INT], root=0)


if rank == 0:
    f2 = open("AData.dat", 'r')
    for k in range(1, numprocs):
        A_part = np.empty((rcounts_M[k], N), dtype=np.float64)
        for j in range(rcounts_M[k]):
            for i in range(N):
                A_part[j,i] = np.float64(f2.readline())
        comm.Send([A_part, rcounts_M[k]*N, MPI.DOUBLE], dest=k, tag=0)
        f2.close()
        A_part = np.empty((M_part, N), dtype=np.float64)
    else:
        A_part = np.empty((M_part, N), dtype=np.float64)
        comm.Recv([A_part, M_part*N, MPI.DOUBLE], source=0, tag=0, status=None)

if rank == 0:
    b = np.empty(M_part, dtype=np.float64)
    f3 = open('bData.dat', 'r')
    for j in range(M):
        b[j] = np.float64(f3.readline())
    f3.close()
else:
    b = None

b_part = np.empty(0, dtype=np.float64)

comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
              [b_part, M_part, MPI.DOUBLE], root=0)

if rank == 0:
    x = np.zeros(N, dtype=np.float64)
else:
    x = None

x_part = np.empty(rcounts_N[rank], dtype=np.float64)

comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
              [x_part, rcounts_N[rank], MPI.DOUBLE], root=0)

x_part = conjugate_gradient_method(A_part, b_part, x_part,
                                   N, rcounts_N[rank], rcounts_N, displs_N)

comm.Gatherv([x_part, rcounts_N[rank], MPI.DOUBLE],
             [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank == 0:
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i')
    ax.set_ylabel('x[i]')