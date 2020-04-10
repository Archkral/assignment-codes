import numpy as np
import scipy.io as sio
from math import *
from Part2.Equations import *
import matplotlib.pyplot as mplt

mat = sio.loadmat('../Assignment input code and data/F16traindata_CMabV_2017.mat')
Cm = np.array(mat['Cm'])
z_k = np.array(mat['Z_k'])  # 3x10001 measured output array - true angles but contaminated with white noise/bias
u_k = np.array(mat['U_k'])  # 3x10001 input array - U_k = np.array([u_dot, v_dot, w_dot])

# h(x,u,t) = [alpha_true*(1+C_alpha), beta_true, V_true]
# alpha_true = atan(w / u)
# beta_true = atan(v / sqrt(u ^ 2 + w ^ 2))
# V_true = sqrt(sum(x_k[1:3] ** 2))
# z = [alpha_true * (1 + C_alpha), beta_true, V_true] + noise


# Q2.3
alpha = sum(z_k[:, 0]) / len(z_k)
avar = sum((z_k[:, 0]-alpha)**2) / (len(z_k)-1)
beta = sum(z_k[:, 1]) / len(z_k)
bvar = sum((z_k[:, 1]-beta)**2) / (len(z_k)-1)
V = sum(z_k[:, 2]) / len(z_k)
Vvar = sum((z_k[:, 2]-V)**2) / (len(z_k)-1)
ux = V*cos(alpha)*cos(beta)  # V*cos(alpha)*cos(beta)
vx = V*sin(beta)  # V*sin(beta)
wx = V*sin(alpha)*cos(beta)  # V*sin(alpha)*cos(beta)
x_00 = np.array([[ux, vx, wx, 0]]).T  # Initial state estimate
Ex_00 = np.array([[150, 0, -20, 0.33]]).T  # optimal state estimate

P_00 = np.diag([avar, bvar, Vvar, 1])  # initial cov matrix estimate

Q = np.diag([1e-3, 1e-3, 1e-3, 0]) ** 2
R = np.diag([0.01, 0.0058, 0.112]) ** 2
B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]]).astype(float)
G = np.diag([1, 1, 1, 0]).astype(float)

# Setup
x = x_00
x_k1k1 = Ex_00
P_k1k1 = P_00

# Iterator
ti = 0
dt = 0.01
tf = dt
epsilon = 1e-10  # only used for IEKF
maxIter = 100  # only used for IEKF
N = len(u_k)
z_storage = np.zeros([np.size(u_k, axis=0), np.size(u_k, axis=1)])
x_storage = np.zeros([np.size(u_k, axis=0), np.size(x_00, axis=0)])  # u_k[0] = 10001, u_k[1] = 3, x_00[1] = 4
stdx_storage = np.zeros([np.size(u_k, axis=0), np.size(x_00, axis=0)])  # u_k[0] = 10001, u_k[1] = 3, x_00[1] = 4
iteration_counter = []

# Check observability of state
uk = np.array([u_k[0]]).T
dFx = kf_dfx(ti, x, B@uk)
x_kk1 = rk38(kf_fx, x_k1k1, B@uk, [ti, tf])
dHx = kf_dhx(tf, x_kk1, B@uk)
rankHF = kf_obs(dHx, dFx)
if rankHF < len(x):
    print('The current state is not observable; rank of Observability Matrix is (', rankHF,'), but it should be (', len(x), ').\n')

for k in range(N):
    zm = np.array([z_k[k]]).T
    uk = np.array([u_k[k]]).T
    # Note: _k1k1 variables from this point until their next update are in fact _kk
    # Calculate Phi/Gamma for state equation
    dFx = kf_dfx(ti, x_k1k1, B@uk)  # should it be linearized around nominal values?
    [Phi, Gamma] = c2d(dFx, G, dt)
    # Phi is basically identity in this case
    x_kk1 = rk38(kf_fx, x_k1k1, B@uk, [ti, tf])
    z_kk1 = kf_hx(tf, x_kk1, B@uk)

    P_kk1 = Phi @ P_k1k1 @ Phi.T + Gamma @ Q @ Gamma.T
    # iterative part
    eta1 = x_kk1
    its = 0
    for its in range(maxIter):
        its += 1

        dHx = kf_dhx(tf, eta1, B@uk)
        K = P_kk1 @ dHx.T @ np.linalg.inv(dHx @ P_kk1 @ dHx.T + R)
        z_k1k1 = kf_hx(tf, eta1, B@uk)  # predicted output
        x_k1k1 = x_kk1 + K @ (zm - z_k1k1 - dHx@(x_kk1 - eta1))
        err = np.linalg.norm((x_k1k1-eta1))/np.linalg.norm(eta1)
        eta1 = x_k1k1
        if err < epsilon:
            # Break the loop if error < maximum allowed error
            break

#    if k==1000:
#        print('Were iterations performed [0 = no, >0 = yes]:',sum(iteration_counter) - k)
#        breakpoint()

    iteration_counter.append(its)

    P_k1k1 = (np.identity(len(x)) - K @ dHx) @ P_kk1 @ (np.identity(len(x)) - K@dHx).T + K @ R @ K.T
    ti = tf
    tf = tf + dt
    x_storage[k,:] = x_k1k1[:,0]
    z_storage[k,:] = z_k1k1[:,0]
    for i in range(len(x_k1k1)):
        stdx_storage[k, i] = P_k1k1[i, i]
print(x_k1k1[:,0],'\n')
print(x_storage[k,:])
breakpoint
mplt.plot(range(N),x_storage[:,0],'r',range(N),x_storage[:,2],'g',range(N),x_storage[:,1],'b',range(N),x_storage[:,3],'y')
mplt.title('state variables')
mplt.show()
mplt.plot(range(N),z_k[:,2],'r',range(N),z_storage[:,2],'b')
mplt.title('Velocity [m/s] (blue = filtered)')
mplt.show()
mplt.plot(range(N),z_k[:,1],'r',range(N),z_storage[:,1],'b')
mplt.title('Side-slip angle [rad] (blue = filtered)')
mplt.show()
mplt.plot(range(N),z_k[:,0],'r',range(N),z_storage[:,0],'b')
mplt.title('Angle of attack [rad] (blue = filtered)')
mplt.show()
mplt.plot(range(N),iteration_counter,'b')
mplt.title('iterations per meme')
mplt.show()
