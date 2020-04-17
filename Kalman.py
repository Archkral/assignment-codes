import numpy as np
from math import *
from Part2.kalman_statefunctions import *


def kalman_IEKF(dt, epsilon, maxIter, x_00, Ex_00, P_00, u_k, z_k, B, G, R, Q):
    # Setup
    x = x_00
    x_k1k1 = Ex_00
    P_k1k1 = P_00

    # Iterator
    ti = 0
    tf = dt
    N = len(u_k)
    z_storage = np.zeros([np.size(u_k, axis=0), np.size(u_k, axis=1)])
    x_storage = np.zeros([np.size(u_k, axis=0), np.size(x_00, axis=0)])  # sizes: u_k[0] = 10001, u_k[1] = 3, x_00[1] =4
    stdx_storage = np.zeros([np.size(u_k, axis=0), np.size(x_00, axis=0)])
    iteration_counter = []

    # Check observability of state
    uk = np.array([u_k[0]]).T
    dFx = kf_dfx(ti, x, B @ uk)
    x_kk1 = rk38(kf_fx, x_k1k1, B @ uk, [ti, tf])
    dHx = kf_dhx(tf, x_kk1, B @ uk)
    rankHF = kf_obs(dHx, dFx)
    if rankHF < len(x):
        print('The current state is not observable; rank of Observability Matrix is (', rankHF, '), but it should be (',
              len(x), ').\n')

    for k in range(N):
        zm = np.array([z_k[k]]).T
        uk = np.array([u_k[k]]).T
        # Note: _k1k1 variables from this point until their next update are in fact _kk
        # Calculate Phi/Gamma for state equation
        dFx = kf_dfx(ti, x, B @ uk)  # should it be linearized around nominal values?
        [Phi, Gamma] = c2d(dFx, G, dt)
        # Phi is identity in the assignment, so it can technically be skipped :/
        x_kk1 = rk38(kf_fx, x_k1k1, B @ uk, [ti, tf])
        z_kk1 = kf_hx(tf, x_kk1, B @ uk)

        P_kk1 = Phi @ P_k1k1 @ Phi.T + Gamma @ Q @ Gamma.T
        # iterative part
        eta1 = x_kk1
        its = 0
        for its in range(maxIter):
            its += 1

            dHx = kf_dhx(tf, eta1, B @ uk)
            K = P_kk1 @ dHx.T @ np.linalg.inv(dHx @ P_kk1 @ dHx.T + R)
            z_k1k1 = kf_hx(tf, eta1, B @ uk)  # predicted output
            x_k1k1 = x_kk1 + K @ (zm - z_k1k1 - dHx @ (x_kk1 - eta1))
            err = np.linalg.norm((x_k1k1 - eta1)) / np.linalg.norm(eta1)
            eta1 = x_k1k1
            if err < epsilon:
                # Break the loop if error < maximum allowed error
                break

        #    if k==1000:
        #        print('Were iterations performed [0 = no, >0 = yes]:',sum(iteration_counter) - k)
        #        breakpoint()

        iteration_counter.append(its)

        P_k1k1 = (np.identity(len(x)) - K @ dHx) @ P_kk1 @ (np.identity(len(x)) - K @ dHx).T + K @ R @ K.T
        ti = tf
        tf = tf + dt
        x_storage[k, :] = x_k1k1[:, 0]
        z_storage[k, :] = z_k1k1[:, 0]
        for i in range(len(x_k1k1)):
            stdx_storage[k, i] = P_k1k1[i, i]

    return x_storage, z_storage, stdx_storage, iteration_counter
