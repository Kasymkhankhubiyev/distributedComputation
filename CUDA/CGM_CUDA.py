from mpi4py import MPI
from numpy import zeros, empty, arange, int32, float64, array, dot
import cupy as cp

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def сonjugate_gradient_method(A_part, b_part, x, N) :

    # send data from CPU to graphic memory
    A_part_d = cp.asarray(A_part)
    x_d = cp.asarray(x)
    b_part_d = cp.asarray(b_part)

    s = 1
    p_d = cp.zeros(N, dtype=float64)
    while s <= N :
        if s == 1 :
            # r = dot(A.T, dot(A,x) - b)
            r_temp = cp.dot(A_part_d.T, cp.dot(A_part_d,x_d) - b_part_d).get()
            r = zeros(N, dtype=float64)
            comm.Allreduce(r_temp, r, op=MPI.SUM)
            r_d = cp.asarray(r)
        else :
            r_d = r_d - q_d / cp.dot(p_d, q_d)
        p_d = p_d + r_d / cp.dot(r_d, r_d)
        q_temp = cp.dot(A_part_d.T, cp.dot(A_part_d, p_d)).get()
        q = zeros(N, dtype=float64)
        comm.Allreduce(q_temp, q, op=MPI.SUM)
        q_d = cp.asarray(q)
        x_d = x_d - p_d / cp.dot(p_d ,q_d)
        s = s + 1
    return x_d.get()

if rank == 0 :
    N = array(2000, dtype=int32)
    M = array(5000, dtype=int32)
else :
    N = array(0, dtype=int32)
    
comm.Bcast(N, root=0)

def auxialiry_array(M, numprocs) :
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
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

if rank == 0 :
    rcounts, displs = auxialiry_array(M, numprocs)
else :
    rcounts = None; displs = None
          
M_part = array(0, dtype=int32) 

comm.Scatter([rcounts, 1, MPI.INT],
             [M_part, 1, MPI.INT], root=0)
         
from numpy import random
A_part = random.random_sample((M_part, N))    

if rank == 0 :
    from numpy import sin, pi
    x_model = array([sin(2*pi*i/(N-1)) for i in range(N)],
                    dtype=float64)
else :
    x_model = empty(N, dtype=float64)
                      
comm.Bcast(x_model, root=0)     

b_part = dot(A_part, x_model)

x = zeros(N)

if rank == 0 :
    start_time = MPI.Wtime()

x = сonjugate_gradient_method(A_part, b_part, x, N)

if rank == 0 :
    end_time = MPI.Wtime()
    print(f'Elapsed time for {numprocs} processes is {end_time-start_time:.4f} sec')


if rank==0 :
    from matplotlib.pyplot import figure, axes, show
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(arange(N), x, '-r', lw=3)
    show()