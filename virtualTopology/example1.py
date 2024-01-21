from mpi4py import MPI

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

print(f'Process {rank_cart} has up neighbour={neighbour_up} and down neighbour={neighbour_down}')
print(f'Process {rank_cart} has left neighbour={neighbour_left} and right neighbour={neighbour_right}\n')