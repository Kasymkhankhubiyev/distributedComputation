from mpi4py import MPI
from numpy import zeros, empty, arange, int32, float64, array, dot
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def сonjugate_gradient_method(A_part, b, x, N, M,
                              M_part, N_part, 
                              rcounts_M, displs_M,
                              rcounts_N, displs_N,
                              comm) :
    s = 1
    p_part = zeros(N_part, dtype=float64)
    while s <= N :
        if s == 1 :
            # r = dot(A.T, dot(A,x) - b)
            Ax_part = dot(A_part, x)
            if rank==0 :
                Ax = empty(M, dtype=float64)
            else :
                Ax = None
            comm.Gatherv([Ax_part, M_part, MPI.DOUBLE],
                         [Ax, rcounts_M, displs_M, MPI.DOUBLE], 
                         root=0)
            if rank== 0 :
                Ax = Ax - b
            comm.Scatterv([Ax, rcounts_M, displs_M, MPI.DOUBLE],
                          [Ax_part, M_part, MPI.DOUBLE], 
                          root=0)
            r_temp = dot(A_part.T, Ax_part)
            r_part = empty(N_part, dtype=float64)
            
            comm.Reduce_scatter([r_temp, N, MPI.DOUBLE],
                                [r_part, N_part, MPI.DOUBLE], 
                                rcounts_N, op=MPI.SUM)
        else :
            # r = r - q/dot(p, q)
            r_part = r_part - q_part/ScalP
            
        # p = p + r/dot(r, r)
        
        ScalP_temp = array(dot(r_part, r_part), dtype=float64)
        ScalP = array(0, dtype=float64)
        comm.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)
        
        p_part = p_part  + r_part/ScalP
        
        # q = dot(A.T, dot(A, p))
        
        p = empty(N, dtype=float64)
        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p, rcounts_N, displs_N, MPI.DOUBLE])
        Ap_part = dot(A_part, p)
        q_temp = dot(A_part.T, Ap_part)
        q_part = empty(N_part, dtype=float64)
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE],
                            [q_part, N_part, MPI.DOUBLE], 
                            recvcounts=rcounts_N, op=MPI.SUM)
        
        # x = x - p/dot(p,q)
        
        ScalP_temp = array(dot(p_part, q_part), dtype=float64)
        ScalP = array(0, dtype=float64)
        comm.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)
        
        if rank== 0 :
            x = x - p/ScalP
        
        s = s + 1
    return x

# Считываем из файла число строк M и число столбцов N
if rank == 0 :
    f1 = open('in.dat', 'r')
    N = array(int(f1.readline()), dtype=int32)
    M = array(int(f1.readline()), dtype=int32)
    f1.close()
else :
    N = array(0, dtype=int32)
    M = None
    
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
    rcounts_M, displs_M = auxialiry_array(M, numprocs)
    rcounts_N, displs_N = auxialiry_array(N, numprocs)
else :
    rcounts_M = None; displs_M = None
    rcounts_N = empty(numprocs, dtype=int32); displs_N = None
          
M_part = array(0, dtype=int32) 
N_part = array(0, dtype=int32) 

comm.Scatter([rcounts_M, 1, MPI.INT],
             [M_part, 1, MPI.INT], root=0)

comm.Scatter([rcounts_N, 1, MPI.INT],
             [N_part, 1, MPI.INT], root=0)
             
comm.Bcast(rcounts_N, root=0)             
             
if rank == 0 :
    f2 = open('AData.dat', 'r')
    A_part = empty((M_part,N), dtype=float64)
    for j in range(M_part) :
        for i in range(N) :
            A_part[j,i] = float(f2.readline())
    for k in range(1,numprocs) :
        A_part_temp = empty((rcounts_M[k],N), dtype=float64)
        for j in range(rcounts_M[k]) :
            for i in range(N) :
                A_part_temp[j,i] = float(f2.readline())
        comm.Send(A_part_temp, dest=k)
        del(A_part_temp)
    f2.close()
else :
    A_part = empty((M_part,N), dtype=float64)
    comm.Recv(A_part, source=0)             

if rank == 0 :
    b = empty(M, dtype=float64)
    f3 = open('bData.dat', 'r')
    for i in range(M) :
        b[i] = float(f3.readline())
    f3.close()
else : 
    b = None

x = zeros(N)

x = сonjugate_gradient_method(A_part, b, x, N, M,
                              M_part, N_part, 
                              rcounts_M, displs_M,
                              rcounts_N, displs_N,
                              comm)

if rank==0 :
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(arange(N), x, '-r', lw=3)

    show()