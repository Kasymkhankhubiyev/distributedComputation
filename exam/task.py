import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()

num_rows = num_cols = np.int32(np.sqrt(numprocs))
comm_cart = comm.Create_cart(dims=(num_rows, num_cols), periods=(False, False), reorder=True)
rank_cart = comm_cart.Get_rank()

# группируем
comm_col = comm_cart.Split(rank_cart % num_cols, rank_cart)
comm_row = comm_cart.Split(rank_cart // num_rows, rank_cart)

if rank_cart == 0:
    M = np.array(10, np.int32)
    N = np.array(10, np.int32)
else:
    N = np.array(0 , np.int32)

# рассылаем по группе процессов по строке
comm_col.Bcast(N, root=0)

def auxialiry_array(M, numprocs) :
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)
    ave, res = divmod(M, numprocs)
    displs[0] = 0
    for k in range(numprocs) :
        if k < res :
            rcounts[k] = ave + 1
        else :
            rcounts[k] = ave
        if k >= 1:
            displs[k] = displs[k-1] + rcounts[k-1]
    return rcounts, displs

# if rank_cart == 0:
#     rcounts_M, displs_M = auxialiry_array(M, numprocs)
    
#     comm_col.Scatterv()


if rank_cart == 0:

    rcounts_M, displs_M = auxialiry_array(M, numprocs // num_cols)
    rcounts_N, displs_N = auxialiry_array(N, numprocs // num_rows)

    displs_M = np.append(displs_M, [M])
    displs_N = np.append(displs_N, [N])

    M_part, N_part = np.array(0, dtype=np.int32), np.array(0, dtype=np.int32)

    comm_col.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

    comm_row.Scatter([rcounts_N, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)

    comm_row.Bcast(M_part, root=0)
    comm_col.Bcast(N_part, root=0)

    matrix = np.arange(1, M*N+1, dtype=np.int32).reshape((M, N))

    sub_matrixes = np.array(0, dtype=np.int32)
    lengthes = []
    displs = np.zeros(numprocs, dtype=np.int32)

    for i in range(num_rows):
        for j in range(num_cols):
            sub_matrix = matrix[displs_M[i] : displs_M[i+1], 
                                displs_N[j] : displs_N[j+1]]
            
            sub_matrixes = np.append(sub_matrixes, sub_matrix.flatten())
            lengthes.append(len(sub_matrix.flatten()))

    for i in range(1, len(lengthes)):
        displs[i] = displs[i-1] + lengthes[i-1]


    sub_matrix = np.empty((M_part * N_part), dtype=np.int32)

    comm_cart.Scatterv([sub_matrixes[1:].flatten(), 
                        np.array(lengthes, dtype=np.int32), 
                        displs, 
                        MPI.INT],
                       [sub_matrix, M_part * N_part, MPI.INT], root=0)
    
    print(f'for process with rank {rank_cart} matrix is:\n {sub_matrix.reshape(M_part, N_part)}\n')

else:
    
    M_part, N_part = np.array(0, dtype=np.int32), np.array(0, dtype=np.int32)

    if rank_cart % num_cols == 0:
        comm_col.Scatter([None, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

    if rank_cart // num_rows == 0:
        comm_row.Scatter([None, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)

    comm_row.Bcast(M_part, root=0)
    comm_col.Bcast(N_part, root=0)

    sub_matrix = np.empty((M_part * N_part), dtype=np.int32)

    comm_cart.Scatterv([None, None, None, MPI.INT],
                       [sub_matrix, M_part * N_part, MPI.INT], root=0)
    
    print(f'for process with rank {rank_cart} matrix is:\n {sub_matrix.reshape(M_part, N_part)}\n')


        
    