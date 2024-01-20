"""
    USE the following command to run more processes
    than kernels available

    `mpiexec --map-by :oversubscribe -n 21 python MatrixMultiplication.py`

    In this programm we wanna upgrate our code to make the programm run faster
    using virtual topology
"""

import numpy as np
import sys

from matplotlib.pyplot import style, figure, axes, show
from mpi4py import MPI


comm = MPI.COMM_WORLD
numprocs = comm.Get_size()

num_rows = num_cols = np.int32(np.sqrt(numprocs))

# create cartesian topology
# we enetered dimensions, added periods for both directions
# setting periods = (True, True) and turnd on reordering
comm_cart = comm.Create_cart(dims=(num_rows, num_cols),
                             periods=(True, True),
                             reorder=True)

# get a new reordred rank in new coordinated
rank_cart = comm_cart.Get_rank()

def conjugate_gradient_method(A_part, b_part, x_part, N) :
    s = 1
    p_part = np.zeros(N_part, dtype=np.float64)
    while s <= N :
        if s == 1 :
            # paralleling the following code line:
            # r = np.dot(A.T, np.dot(A,x) - b)
            Ax_part_temp = np.dot(A_part, x_part)
            Ax_part = np.empty(M_part, np.float64)
            # Data sending is transfering parallel inside 
            # every commutator over a row
            comm_row.Allreduce(Ax_part_temp, Ax_part, op=MPI.SUM)

            Ax_part = Ax_part - b_part

            # now each process in a row has a corresponding part
            # of a column Ax_part
            r_part_temp = np.dot(A_part.T, Ax_part)
            r_part = np.empty(N_part, dtype=np.float64)
            comm_col.Allreduce(r_part_temp, r_part, op=MPI.SUM)
        else:
            # paralleling the following code line:
            # r = r - q/np.dot(p, q)
            # ScalP_temp = np.array(np.dot(p_part, q_part), dtype=np.float64)
            # ScalP = np.array(0, dtype=np.float64)

            # comm_row.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)
            r_part = r_part - q_part / ScalP

        # paralleling the following code line:
        # p = p + r/np.dot(r, r)
        ScalP_temp = np.array(np.dot(r_part, r_part), dtype=np.float64)
        ScalP = np.array(0, dtype=np.float64)

        comm_row.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)
        p_part = p_part + r_part / ScalP

        # paralleling the following code line:
        # q = np.dot(A.T, np.dot(A, p))
        Ap_part_temp = np.dot(A_part, p_part)
        Ap_part = np.empty(M_part, np.float64)
        # Data sending is transfering parallel inside 
        # every commutator over a row
        comm_row.Allreduce(Ap_part_temp, Ap_part, op=MPI.SUM)

        # now each process in a row has a corresponding part
        # of a column Ap_part
        q_part_temp = np.dot(A_part.T, Ap_part)
        q_part = np.empty(N_part, dtype=np.float64)
        # sum up all q parts and trasfer it over column processes
        comm_col.Allreduce(q_part_temp, q_part, op=MPI.SUM)

        # paralleling the following code line:
        # x = x - p/np.dot(p,q)
        ScalP_temp = np.array(np.dot(p_part, q_part), dtype=np.float64)
        ScalP = np.array(0, dtype=np.float64)
        comm_row.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)
        x_part = x_part - p_part / ScalP

        s = s + 1
    return x_part

N = np.array(200, dtype=np.int32)
M = np.array(300, dtype=np.int32)

def auxiallary_arrays(M: int, numprocs: int):
    rcounts = np.empty(numprocs, dtype=np.int32)
    displs = np.empty(numprocs, dtype=np.int32)

    displs[0] = 0

    ave, res = divmod(M, numprocs)

    for k in range(numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k >= 1:  # here was the difference
            displs[k] = displs[k-1] + rcounts[k-1]

    return rcounts, displs


if rank_cart == 0:
    rcounts_M, displs_M = auxiallary_arrays(M, num_rows)
    rcounts_N, displs_N = auxiallary_arrays(N, num_cols)
else:
    rcounts_M, displs_M = None, None
    rcounts_N, displs_N = np.empty(num_cols, dtype=np.int32), None

M_part = np.array(0, dtype=np.int32)
N_part = np.array(0, dtype=np.int32)

# create new commutators for each row and column
# processes with the same value of `rank % num_cols`
# includes into a one group - subcommutator
comm_col = comm_cart.Split(rank_cart % num_cols, rank_cart)
comm_row = comm_cart.Split(rank_cart // num_rows, rank_cart)

# now we want to scatter data over a group of 
# processes of the same column
# each single communicator has info about all 
# the proccesses include into the group
# before running the code we need to send
# rcounts data to all of root processes of subcommunicators
# but at the start only the first communicator with
# a global rank equal 0 knows about rcounts
if rank_cart in range(0, numprocs, num_cols):
    comm_col.Scatter([rcounts_M, 1, MPI.INT], 
                     [M_part, 1, MPI.INT], root=0)
    
# broadcasting data over rows
comm_row.Bcast(M_part, root=0)

if rank_cart in range(num_cols):
    comm_row.Scatter([rcounts_N, 1, MPI.INT], 
                     [N_part, 1, MPI.INT], root=0)
    
# broadcasting data over rows
comm_col.Bcast(N_part, root=0)

# print(f'for rank = {rank}: M_part = {M_part}, N_part = {N_part}')

A_part = np.random.random_sample((M_part, N_part))

if rank_cart == 0:
    x_model = np.array([np.sin(2*np.pi*i/(N-1)) for i in range(N)], 
                       dtype=np.float64)
else:
    x_model = None
    
x_part = np.empty(N_part, dtype=np.float64)

# now we need to scatter a vector x over processes
# included into one row -> x_part
# after that we spread same x_parts over processes
# included into one column
if rank_cart in range(num_cols):
    comm_row.Scatterv([x_model, rcounts_N, displs_N, MPI.DOUBLE],
                      [x_part, N_part, MPI.DOUBLE], root=0)
    
comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)

# in every process get its own b as a dot product
b_part_temp = np.dot(A_part, x_part)
b_part = np.empty(M_part, np.float64)

# and sum up all the dot products inside a corresponding 
# row group of processes
comm_row.Allreduce(b_part_temp, b_part, op=MPI.SUM)

x_part = np.zeros(N_part, dtype=np.float64)

x_part = conjugate_gradient_method(A_part, b_part, x_part, N)

if rank_cart == 0:
    x = np.empty(N, dtype=np.float64)
else:
    x = None

if rank_cart in range(num_cols):
    comm_row.Gatherv([x_part, N_part, MPI.DOUBLE],
                     [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank_cart == 0:
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(np.arange(N), x, '-r', lw=3)

    show()
