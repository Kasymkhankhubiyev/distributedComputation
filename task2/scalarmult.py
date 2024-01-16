import numpy as np
import sys
from mpi4py import MPI


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()


if rank == 0:
    N = 20
    a = np.arange(N, dtype=np.float64) + 1
    # print(f'type of a is {a.dtype}')
    print(f'scalar a x a = {np.dot(a, a)}')
else:
    a = None


if rank == 0:
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)

    # rcounts[0] = 0
    displs[0] = 0

    ave, res = divmod(N, numprocs)

    for k in range(0, numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k-1] + rcounts[k-1]

    print(rcounts, displs)
else:
    rcounts, displs = None, None

N_part = np.array(1, dtype=np.int32)

comm.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)

a_part = np.empty(N_part, dtype=np.float64)

comm.Scatterv([a, rcounts, displs, MPI.DOUBLE], [a_part, N_part, MPI.DOUBLE], root=0)

scalP_temp = np.array(np.dot(a_part, a_part), dtype=np.float64)

print(f'for rank: {rank} dot_prod: {scalP_temp}')

# Reduce, Allreduce - в первом результат на корневом процессе, 
# а во втором с корневого бродкастом на все процесссы раскидываем
# TODO: replace the following code block with a Reduce command
# if rank == 0:
#     output = scalP_temp.copy()
#     for k in range(1, numprocs):
#         comm.Recv(scalP_temp, source=MPI.ANY_SOURCE)
#         output += scalP_temp
    
#     print(f'total dot product is {output}')
# else:
#     comm.Send(scalP_temp, dest=0)

if rank == 0:
    output = scalP_temp.copy()
else:
    output = None

comm.Reduce(scalP_temp, output, MPI.SUM, root=0)

if rank == 0:
    print(output)
