from mpi4py import MPI

"""
    Run the programm as ``mpiexec -n 4 python TestProgramPy.py``
"""

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

print(f'Hello from process {rank} out of {numprocs}')