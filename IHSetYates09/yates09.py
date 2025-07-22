import numpy as np
from numba import njit
import math

# @jit
# def yates09(E, dt, a, b, cacr, cero, Yini):
#     """
#     Yates et al. 2009 model
#     """
#     Seq = (E - b) / a
#     Y = np.zeros_like(E)
#     Y[0] = Yini
#     for i in range(0, len(E)-1):
#         if Y[i] < Seq[i+1]:
#             Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cacr *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1]
#         else:
#             Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cero *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1]

#     return Y, Seq

@njit(fastmath=True, cache=True)
def yates09(E, dt, a, b, cacr, cero, Yini):
    n = E.shape[0]
    # Precomputados
    Seq = (E - b) / a
    sqrtE = np.sqrt(E)              # raíz cuadrada de cada E[i]
    Y = np.empty(n, dtype=E.dtype)
    Y[0] = Yini

    # Constantes locales
    a_cacr = a * cacr
    a_cero = a * cero

    for i in range(n - 1):
        seq_n = Seq[i + 1]
        delta = Y[i] - seq_n
        # Evitamos ramificaciones usando expresiones matemáticas directas
        s = sqrtE[i + 1]
        e1 = math.exp(-a_cacr * s * dt[i])
        e2 = math.exp(-a_cero * s * dt[i])
        # cond = 1.0 si entra en cacr, 0.0 si entra en cero
        cond = 1.0 if Y[i] < seq_n else 0.0
        Y[i + 1] = delta * (cond * e1 + (1.0 - cond) * e2) + seq_n

    return Y, Seq


def yates09_njit(E, dt, a, b, cacr, cero, Yini):
    """
    Yates et al. 2009 model
    """
    Seq = (E - b) / a
    Y = np.zeros_like(E)
    Y[0] = Yini
    for i in range(0, len(E)-1):
        if Y[i] < Seq[i+1]:
            Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cacr *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1]
        else:
            Y[i+1] = ((Y[i]-Seq[i+1])*np.exp(-1 * a *cero *(E[i+1] ** 0.5)*dt[i]))+Seq[i+1]

    return Y, Seq