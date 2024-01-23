from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

a = np.array(rank, dtype=np.int32)
b = np.array(100, dtype=np.int32)

requests = [MPI.Request() for i in range(2)]

if rank == 0:
    requests[0] = comm.Isend([a, 1, MPI.INT], dest=numprocs-1, tag=0)
    requests[1] = comm.Irecv([b, 1, MPI.INT], source=1, tag=0)
elif rank == numprocs - 1:
    requests[0] = comm.Isend([a, 1, MPI.INT], dest=numprocs-2, tag=0)
    requests[1] = comm.Irecv([b, 1, MPI.INT], source=0, tag=0)
else:
    requests[0] = comm.Isend([a, 1, MPI.INT], dest=rank-1, tag=0)
    requests[1] = comm.Irecv([b, 1, MPI.INT], source=rank+1, tag=0)

    # MPI.Request.Wait(requests[0], status=None)
    a += 10


MPI.Request.Wait(requests[1], status=None)

print(f'Process number {rank} got number {b}')