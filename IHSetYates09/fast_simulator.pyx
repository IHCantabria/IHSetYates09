# cython: boundscheck=False
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple yates09c(np.ndarray[double, ndim=1] E, double dt, double a, double b, double cacr, double cero, double Yini):
    """
    Yates et al. 2009 model
    """
    cdef int n = len(E)
    cdef np.ndarray[double, ndim=1] Seq = np.empty(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] Y = np.empty(n, dtype=np.float64)
    cdef int i

    Seq[:] = (E - b) / a
    Y[0] = Yini

    for i in range(n - 1):
        if Y[i] < Seq[i + 1]:
            Y[i + 1] = ((Y[i] - Seq[i + 1]) * np.exp(-1 * a * cacr * (E[i + 1] ** 0.5) * dt)) + Seq[i + 1]
        else:
            Y[i + 1] = ((Y[i] - Seq[i + 1]) * np.exp(-1 * a * cero * (E[i + 1] ** 0.5) * dt)) + Seq[i + 1]

    return Y, Seq

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] model_simulation(np.ndarray[double, ndim=1] E, double dt, double Yini, np.ndarray[int, ndim=1] idx_obs, np.ndarray[double, ndim=1] params):
    """
    Executa o modelo com parâmetros dados transformados
    """
    cdef double a = params[0]
    cdef double b = params[1]
    cdef double cacr = params[2]
    cdef double cero = params[3]
    cdef np.ndarray[double, ndim=1] Ymd, _

    Ymd, _ = yates09c(E, dt, -np.exp(a), np.exp(b), -np.exp(cacr), -np.exp(cero), Yini)
    return Ymd[idx_obs]