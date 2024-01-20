from mpi4py import MPI

"""
    Run the programm as ``mpiexec -n 4 python example.py``
"""

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

num_row = 2
num_col = 4

comm_cart = comm.Create_cart(dims=[num_row, num_col],
                             periods=(True, True),
                             reorder=True)

rank_cart = comm_cart.Get_rank()

coords = comm_cart.Get_coords(rank_cart)


print(f'Process with rank {rank} has a rank_cart = {rank_cart} and coordinates = {coords}')