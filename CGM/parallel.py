import numpy as np
import sys
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def conjugate_gradient_method(A_part, b, x, N, M, 
                              M_part, N_part, comm,
                              rcounts_M, rcounts_N, 
                              displs_M, displs_N):
    s = 1
    p_part = np.zeros(N_part, dtype=np.float64)
    while s <= N:
        if s == 1:
            # paralleling the following code:
            # r = np.dot(A_part.T, np.dot(A_part,x) - b)

            # каждый процесс считает свою часть матричного умножения
            Ax_part = np.dot(A_part, x)

            if rank == 0:
                Ax = np.empty(M, dtype=np.float64)
            else:
                Ax = None
            
            # собираем все результирующие вектора в один Ax
            comm.Gatherv([Ax_part, M_part, MPI.DOUBLE],
                         [Ax, rcounts_M, displs_M, MPI.DOUBLE], 
                         root=0)
            
            if rank == 0:
                Ax = Ax - b
            
            # раздаем всем процессам соответствующий кусочек нового вектора
            comm.Scatterv([Ax, rcounts_M, displs_M, MPI.DOUBLE],
                          [Ax_part, M_part, MPI.DOUBLE], 
                          root=0)
            
            r_temp = np.dot(A_part.T, Ax_part)
            r_part = np.empty(N_part, dtype=np.float64)

            # агрегируем сумму результирующих векторов с каждого процесса 
            # и разссылаем всем процессам обратно соответствующий кусок 
            # итогового вектора
            comm.Reduce_scatter([r_temp, N, MPI.DOUBLE],
                                [r_part, N_part, MPI.DOUBLE],
                                rcounts_N, MPI.SUM)
        else:
            # paralleling the following code:
            # r = r - q/np.dot(p, q)

            r_part = r_part - q_part / ScalP

        # p = p + r/np.dot(r, r)
        ScalP_temp = np.array(np.dot(r_part, r_part), dtype=np.float64)
        ScalP = np.array(0, dtype=np.float64)
        comm.Allreduce(ScalP_temp, ScalP, MPI.SUM)

        p_part = p_part + r_part / ScalP

        # paralleling the followinf code:
        #q = np.dot(A_part.T, np.dot(A_part, p))
        p = np.empty(N, dtype=np.float64)
        # собираем кусочки вектора со всех процессов
        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                     [p, rcounts_N, displs_N, MPI.DOUBLE])
        
        Ap_part = np.dot(A_part, p)
        q_temp = np.dot(A_part.T, Ap_part)
        q_part = np.empty(N_part, dtype=np.float64)
        
        # Аггрегируем результат вычисления и раздаем соответсвующие
        # кусочи процессам
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE],
                            [q_part, N_part, MPI.DOUBLE],
                            rcounts_N, MPI.SUM)

        # paralleling the followinf code:
        #x = x - p/np.dot(p,q)
        ScalP_temp = np.array(np.dot(p_part, q_part), dtype=np.float64)
        ScalP = np.array(0, dtype=np.float64)
        comm.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)

        if rank == 0:
            x = x - p / ScalP

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

x = np.zeros(N, dtype=np.float64)

x = conjugate_gradient_method(A_part, b, x, N, M, 
                              M_part, N_part, comm,
                              rcounts_M, rcounts_N,
                              displs_M, displs_N)

if rank == 0:
    fig = plt.figure()
    ax = plt.axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(np.arange(N), x, '-r', lw=3)

    plt.show()