from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

# get group of processes
group = comm.Get_group()

# create a new group
new_group = group.Range_incl([(2, numprocs-1, 1)])
# create a new communicator
newcomm = comm.Create(new_group)

new_rank = newcomm.Get_rank()

print(f'My rank is {rank} in the 1st group and {new_rank} in the 2nd group')