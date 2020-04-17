"""
Order of functions:
1. [kf_fx] - State transition matrix for system dynamics equation
2. [kf_hx] - Measurement/Observation matrix for output dynamics equation
3. [kf_dfx] - Jacobian of system transition matrix
4. [kf_dhx] - Jacobian of output observation matrix
5. [kf_obs] - State observability of discretized linear system
6. [c2d] - changes matrices/scalars into discrete form - uses ZOH
7. [rk38] - runge-kutta 3/8ths rule, can be expanded for multi-state ODEs
"""
import numpy as np
from math import *
import scipy.linalg as sl


def kf_fx(t, x, u):
    F = np.zeros((4, 4))
    return F@x + u  # u is inputted as B@u in this case


def kf_hx(t, x, u):
    z = np.array([[atan(x[2] / x[0]) * (1 + x[3])],
                  [atan(x[1] / sqrt(x[0] ** 2 + x[2] ** 2))],
                  [sqrt(sum(x[0:3] ** 2))]])
    return z.astype(float)


# Jacobian = [df_1/dx1 df_1/dx2
#             df_2/dx1 df_2/dx2]
# input variable derivative is ignored
def kf_dfx(t, x, u):
    dFx = np.zeros((4, 4))
    return dFx


def kf_dhx(t, x, u):
    dHx = [[-x[2]/(x[0]**2+x[2]**2)*(1+x[3]), 0, 1/(x[0]+x[2]**2/x[0])*(1+x[3]), atan(x[2]/x[0])],
           [-x[0]*x[1]/(sqrt(x[0]**2+x[2]**2)*(sum(x[0:3]**2))), sqrt(x[0]**2+x[2]**2)/sum(x[0:3]**2), -x[2]*x[1]/(sqrt(x[0]**2+x[2]**2)*(sum(x[0:3]**2))), 0],
           [x[0]/sqrt(sum(x[0:3]**2)), x[1]/sqrt(sum(x[0:3]**2)), x[2]/sqrt(sum(x[0:3]**2)), 0]]
    return np.array(dHx).astype(float)


def kf_obs(h, fx):  # rank of [C;CA;CA^2;...] = observability
    f = np.identity(np.size(fx,axis=0))
    Rank = np.dot(h, f)
    for i in np.linspace(1, len(fx)-1, len(fx)-1):
        f = np.dot(f, fx)
        Rank = np.append(Rank, np.dot(h,f), axis=0)

    return np.linalg.matrix_rank(Rank)  # numpy.rank is tensor rank


# Using a form of x_dot = Ax + Bu (a and b matrices from state-space)-------------
def c2d(a, b, dt):
    c = np.append(a, b, axis=1)
    c = np.append(c, np.zeros([c.shape[1]-c.shape[0], c.shape[1]]), axis=0)
    d = sl.expm(c*dt)
    a = d[0:a.shape[0], 0:a.shape[1]]
    b = d[0:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1]]
    return np.array(a), np.array(b)


def rk38(fn, x, u, t):
    a = t[0]
    b = t[1]
    t = a
    n = 2  # number of steps
    h = (b - a) / n  # step size
    for j in range(n):
        k1 = h * fn(t, x, u)
        k2 = h * fn(t + h / 3, x + k1/3, u)
        k3 = h * fn(t + 2*h/3, x - k1/3 + k2, u)
        k4 = h * fn(t + h, x + k1 - k2 + k3, u)
        x = x + (k1 + 3 * k2 + 3 * k3 + k4) / 8
    return x.astype(float)
# https://math.stackexchange.com/questions/721076/help-with-using-the-runge-kutta-4th-order-method-on-a-system-of-2-first-order-od
# ^ In case of interacting ODEs
