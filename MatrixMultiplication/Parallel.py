# from numpy import empty, int32, array
import numpy as np
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

"""
    Будем работать по схеме master&slave
    Т.е. будет управляющий процесс и исполняющий
"""

if rank == 0:
    # Считываем из файла число строк M и число столбцов N
    f1 = open('in.dat', 'r')
    N = np.array(int(f1.readline()), dtype=np.int32)
    M = np.array(int(f1.readline()), dtype=np.int32)
    f1.close()
else:
    N = np.array(0, dtype=np.int32)
    M = np.array(0, dtype=np.int32)


"""

    Send and Recieve functions are distributed in time, so formally some
    processes are busy, but in real they can be just waiting for source

    But there are some more efficient algorothms to send data - cascade or treeview (древовидный)

    Broadcasting -  широкоевещание 
"""

if rank == 0:
    comm.Bcast(M, root=0)
    for i in range(1, numprocs):
        # упращенный синтаксис
        comm.Send(N, dest=i)

        # """
        #     Полный синтаксис:
        #         N - массив переменных
        #         1 - кол-во элемнетов в массиве на отправку
        #         MPI.INT - размер данных
        # """
        # comm.Send([N, 1, MPI.INT], dest=i, tag=0)
else:
    comm.Bcast(M, root=0)
    # упращенный синтаксис
    comm.Recv(N, source=0)
    # comm.Recv(M, source=0)

    # """
    #     Полный синтаксис:
    #         N - массив переменных
    #         1 - кол-во элемнетов в массиве на отправку
    #         MPI.INT - размер данных
    # """
    # comm.Recv([N, 1, MPI.INT], source=0, tag=0)
    

# print(f'For rank={rank}: N={N} , M={M}')
    
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

# print(f'rank {rank}, M: {M_part}')

# # Выделяем под матрицы A,x и b соответствующее место в памяти
# A = np.empty((M,N)); x = np.empty(N); b = np.empty(M)

print(f'for rank: {rank} M_part: {M_part}, N: {N}')
	
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
    # A_part = np.empty((0, N), dtype=np.float64)
else:
    A_part = np.empty((M_part, N), dtype=np.float64); 
    comm.Recv(A_part, source=0)

# sys.exit()
		
# Считываем из файла вектор x
x = np.empty(N, dtype=np.float64)
if rank == 0:
    f3 = open('xData.dat', 'r')
    for i in range(N) :
        x[i] = float(f3.readline())
    f3.close()

comm.Bcast(x, root=0)

# if rank != 0:
    # b_part = np.dot(A_part, x)

b_part = np.dot(A_part, x)

# if rank == 2:
#     print(b_part)

# if rank == 0:
#     b = np.empty(N, dtype=np.float64)
#     # b_part = np.empty(M // (numprocs - 1), dtype=np.float64)

#     status = MPI.Status()
    
#     # данные получаются по порядку
#     for k in range(1, numprocs):
#         # comm.Recv(b_part, source=k)
#         # comm.Recv(b[(k-1)* M // (numprocs - 1): k* M // (numprocs - 1)], source=k)

#         # # данные получаем по мере решения блоков
#         # # comm.Recv(b[(status.Get_source()-1)* M // (numprocs - 1): status.Get_source()* M // (numprocs - 1)], source=MPI.ANY_SOURCE, status=status)
#         # comm.Recv(b_part, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
#         # # + we need to know a recieve status to know a sender index
#         # b[(status.Get_source()-1) * M // (numprocs - 1): status.Get_source() * M // (numprocs - 1)] = b_part


#         # зондирование - получаем информацию о входящем сигнале и заполняем статус
#         comm.Probe(source=MPI.ANY_SOURCE, status=status)
#         comm.Recv(b[(status.Get_source()-1) * M // (numprocs - 1): status.Get_source() * M // (numprocs - 1)], 
#                   source=status.Get_source(), status=status)

#         # use Gather or Gatherv
#     # b = np.empty(N, dtype=np.float64)
#     # b_part = np.empty(M // (numprocs - 1), dtype=np.float64)
#     # b = comm.Gather(b_part, b, root=0)
# else:
#     b = None
#     comm.Send(b_part, dest=0) #, tag=rank)
        

# Каскадный алгоритм - время работы log_2(N), N - число процессов
# Gather function - a function of collective interaction,
# it is runnable only if all processes run the command
# Gather uses only data with the same size
# Gatherv allows to use data with different sizes

# b = comm.Gather(b_part, b, root=0)
# comm.Gather([sendbuf, scount, stype], [recbuf, rcount, rtype], root)

sys.exit()

# for k in range(1, numprocs):
#     rcounts[k] = M // (numprocs - 1)
#     displs[k] = displs[k-1] + rcounts[k-1]

if rank == 0:
    b = np.empty(N, dtype=np.float64)
else:
    b = None

comm.Gatherv([b_part, rcounts[rank], MPI.DOUBLE], 
             [b, rcounts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    print(b)
	
# # Основная вычислительная часть программы
# # Умножаем матрцу A на вектор x
# # ------------------------------------------------------------
# for j in range(M) :
#     b[j] = 0.
#     for i in range(N) :
#         b[j] = b[j] + A[j,i]*x[i]
# #------------------------------------------------------------
	
# # Сохраняем результат вычислений в файл
# f4 = open('Results.dat', 'w')
# for j in range(M) :
#     f4.write(str(b[j])+'\n')
# f4.close()

# print(b)