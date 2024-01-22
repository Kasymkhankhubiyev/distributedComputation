from mpi4py import MPI
import numpy as np

"""
    Run the programm as ``mpiexec -n 8 python example1.py``
"""

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

num_row = 2
num_col = 4

# создаем Декартову топологию - процессы находятся в узлах
# декартовой сетки
comm_cart = comm.Create_cart(dims=[num_row, num_col],
                             periods=(True, True),
                             reorder=True)

rank_cart = comm_cart.Get_rank()

# соседи вдоль возрастания номера строки
neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)

# соседи вдоль возрастания номера столбца
neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)

a = np.array([(rank_cart % num_col + 1 + i)*2**(rank_cart // num_col)
              for i in range(2)], dtype=np.int32)


summ_1 = a.copy()
summ_2 = a.copy()
for n in range(num_row - 1):
    comm_cart.Sendrecv_replace([a, 2, MPI.INT], 
                               dest=neighbour_right, sendtag=0,
                               source=neighbour_left, recvtag=MPI.ANY_TAG,
                               status=None)
    
    summ_1 += a

print(f'for the proccess with a rank {rank_cart} summ = {summ_1}')

for m in range(num_row - 1):
    comm_cart.Sendrecv_replace([a, 2, MPI.INT], 
                               dest=neighbour_down, sendtag=0,
                               source=neighbour_up, recvtag=MPI.ANY_TAG,
                               status=None)
    
    summ_2 += a

print(f'for the process with a rank {rank_cart} column sum = {summ_2}')