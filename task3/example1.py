import numpy as np
from mpi4py import MPI

# get info about all processes
comm = MPI.COMM_WORLD
# get the number of working processes
numprocs = comm.Get_size()

# get the group of the process
group = comm.Get_group()
# get the rank of the process
rank = comm.Get_rank() 

# create the processes group that contains partial number of processes
new_group = group.Range_incl([(2, numprocs-1, 1)])
# get a new rank in a new group of processes
newrank = new_group.Get_rank()

print(f'My rank {rank} in the 1st group and {newrank} in the 2nd group')