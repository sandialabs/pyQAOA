import numpy as np
from numba import njit, prange
from multiprocessing import current_process

def mpnjit(*args,**kwargs):
    """
    Only apply Numba parallelism to a function if it is running on the main process (serial)
    """
    kwargs["parallel"] = current_process().name=="MainProcess"
    return njit(*args,**kwargs)


@mpnjit
def additive_assign(n,x,y):
    for k in prange(1<<n):
        y[k] += x[k]


@mpnjit
def scale(n,alpha,v):
    for k in prange(1<<n):
        v[k] *= alpha

@mpnjit
def axpy(n,alpha,x,y):
    for k in prange(1<<n):
        y[k] += alpha*x[k]

@mpnjit
def inner_product(n,u,v):
    result = 0
    for k in prange(1<<n):
        result += u[k] * v[k]
    return result

@mpnjit
def conj_inner_product(n,u,v):
    result = 0
    for k in prange(1<<n):
        result += np.conj(u[k]) * v[k]
    return result

@mpnjit
def hadamard_mult(n,d,v,Dv):
    for j in prange(len(d)):
        Dv[j] = d[j]*v[j]

@mpnjit
def hadamard_div(n,d,v,Dv):
    for j in prange(1<<n):
        Dv[j] = v[j]/d[j]

@mpnjit
def hadamard_conj_mult(n,d,v,Dv):
    for j in prange(1<<n):
        Dv[j] = numpy.conj(d[j])*v[j]

@mpnjit
def hadamard_conj_div(n,d,v,Dv):
    for j in prange(1<<n):
        Dv[j] = v[j]/numpy.conj(d[j])

@mpnjit
def cexp_hadamard_mult(n,d,theta,v,Uv):
    for k in prange(1<<n):
        Uv[k] = np.exp(1j*theta*d[k])*v[k]

@mpnjit
def cexp_hadamard_div(n,d,theta,v,Uv):
    for k in prange(1<<n):
        Uv[k] = np.exp(-1j*theta*d[k])*v[k]


@mpnjit
def projection_1d(n,v,x,y):
    s = 0
    for j in prange(1<<n):
        s += v[j]*x[j]
    for j in prange(1<<n):
        y[j] = s*v[j]

@mpnjit
def projection_exp_1d(n,alpha,v,x,y):
    s = 0
    for j in prange(1<<n):
        s += v[j]*x[j]
    for j in prange(1<<n):
        y[j] = x[j] - alpha*s*v[j]

@mpnjit
def negate_real(n,z):
    for k in prange(1<<n):
        zk = z[k]
        z[k] = -zk.real + 1j * zk.imag

@mpnjit
def apply_kron2( A, n, k, x, y ):
    ldim = 1<<k
    rdim = 1<<(n-k-1)
    for j in prange(ldim):
        j1 = 2*j*rdim
        j2 = j1+rdim
        for l in prange(rdim):
            y[j1] = A[0,0]*x[j1] + A[0,1]*x[j2]
            y[j2] = A[1,0]*x[j1] + A[1,1]*x[j2]
            j1 += 1
            j2 += 1


@mpnjit
def diag_inner_product(n,u,d,v):
    result = 0
    for k in prange(1<<n):
        result += u[k] * d[k] * v[k]
    return result

@mpnjit
def diag_conj_inner_product(n,u,d,v):
    result = 0
    for k in prange(1<<n):
        result += np.conj(u[k]) * d[k] * v[k]
    return result

@njit
def zspin(n,k,i):
    return 1 - 2 * ( (k>>(n-i-1)) & 1 )

@mpnjit
def ising_dense_h(n,h_v,c):
    n = len(h_v)
    for k in prange(1<<n):
        for i,h in enumerate(h_v):
            c[k] += h * zspin(n,k,i)

#@mpnjit
def ising_sparse_h(n,h_rv,c):
    p = len(h_rv)
    for k in range(1<<n):
        for i, h in h_rv:
            c[k] += h * zspin(n,k,i)

@mpnjit
def ising_dense_J(n,J,c):
    for k in prange(1<<n):
        for i in range(n):
            zi = zspin(n,k,i)
            for j in range(i+1,n):
                c[k] += J[i,j] * zi * zspin(n,k,j)

@mpnjit
def ising_sparse_J(n,J_rc,c):
    for k in prange(1<<n):
        for i,j in J_rc:
            c[k] += zspin(n,k,i) * zspin(n,k,j)

@mpnjit
def ising_sparse_weighted_J(n,J_rcv,c):
    for k in prange(1<<n):
        for i,j,J in J_rcv:
            c[k] += J * zspin(n,k,i) * zspin(n,k,j)


@mpnjit
def sum_sigma_x_mult(n,v,Dv):
    for j in prange(1<<n):
        Dv[j] = 0
        for k in prange(n):
            Dv[j] += v[j^(1<<k)]

@mpnjit
def sum_sigma_x_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for k in prange(n):
            lresult += v[j^(1<<k)]
        result += u[j]*lresult
    return result

@mpnjit
def sum_sigma_x_conj_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for k in prange(n):
            lresult += v[j^(1<<k)]
        result += np.conj(u[j])*lresult
    return result

@mpnjit
def sum_sigma_y_mult(n,v,Dv):
    for j in prange(1<<n):
        Dv[j] = 0
        for l in prange(n):
            k = j ^ (1<<l)
            s = 1j if j>k else -1j
            Dv[j] += s*v[k]

@mpnjit
def sum_sigma_y_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for l in prange(n):
            k = j ^ (1<<l)
            s = 1j if j>k else -1j
            lresult += s*v[j^(1<<k)]
        result += u[j]*lresult
    return result

@mpnjit
def sum_sigma_y_conj_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for l in prange(n):
            k = j ^ (1<<l)
            s = 1j if j>k else -1j
            lresult += s*v[j^(1<<k)]
        result += np.conj(u[j])*lresult
    return result

















