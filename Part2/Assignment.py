import numpy as np
import scipy.io as sio
from math import *

from numpy.core._multiarray_umath import ndarray

from Part2.Equations import *

mat = sio.loadmat('../Assignment input code and data/F16traindata_CMabV_2017.mat')
Cm = np.array(mat['Cm'])
z_k = np.array(mat['Z_k'])  # 3x10001 measured output array - true angles but contaminated with white noise/bias
u_k = np.array(mat['U_k'])  # 3x10001 input array - U_k = np.array([u_dot, v_dot, w_dot])

# h(x,u,t) = [alpha_true*(1+C_alpha), beta_true, V_true]
# alpha_true = atan(w / u)
# beta_true = atan(v / sqrt(u ^ 2 + w ^ 2))
# V_true = sqrt(sum(x_k[1:3] ** 2))
# z = [alpha_true * (1 + C_alpha), beta_true, V_true]


# Q2.3
Ex_00 = np.array([[150, 0, 40, 0.4]]).T  # optimal state estimate
alpha = sum(z_k[:, 0]) / len(z_k)
astd = sqrt(sum((z_k[:, 0]-alpha)**2) / (len(z_k)-1))
beta = sum(z_k[:, 1]) / len(z_k)
bstd = sqrt(sum((z_k[:, 1]-beta)**2) / (len(z_k)-1))
V = sum(z_k[:, 2]) / len(z_k)
Vstd = sqrt(sum((z_k[:, 2]-V)**2) / (len(z_k)-1))
ux = V*cos(alpha)*cos(beta)  # V*cos(alpha)*cos(beta)
vx = V*sin(beta)  # V*sin(beta)
wx = V*sin(alpha)*cos(beta)  # V*sin(alpha)*cos(beta)
x_00 = np.array([[ux, vx, wx, 0]]).T  # Initial state estimate

P_00 = np.diag([astd, bstd, Vstd, 1]) ** 2  # initial cov matrix estimate

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
epsilon = 1e-9  # only used for IEKF
maxIter = 100  # only used for IEKF
N = len(u_k)
z_pred = np.zeros([np.size(u_k, axis=0), np.size(u_k, axis=1)])
xx_k1k1 = np.zeros([np.size(u_k, axis=0), np.size(x_00, axis=0)])  # u_k[0] = 10001, u_k[1] = 3, x_00[1] = 4
stdx_corr = xx_k1k1
iteration_counter = []

# Check observability of state
uk = np.array([u_k[0]]).T
dFx = kf_dfx(ti, x, uk)
Psi = c2d(dFx, B, dt)[1]
x_kk1 = rk38(kf_fx, x_k1k1, uk, [ti, tf]) + Psi@uk
dHx = kf_dhx(tf, x_kk1, uk)
rankHF = kf_obs(dHx, dFx)
if rankHF < len(x):
    print('The current state is not observable; rank of Observability Matrix is (', rankHF,'), but it should be (', len(x), ').\n')

for k in range(N):

    uk = np.array([u_k[k]]).T
    # Note: _k1k1 variables from this point until their next update are in fact _kk
    # Calculate Phi/Gamma/Psi for state equation
    dFx = kf_dfx(ti, x_k1k1, uk)  # should it be linearized around nominal values?
    Psi = c2d(dFx, B, dt)[1]
    [Phi, Gamma] = c2d(dFx, G, dt)
    # Phi is basically identity in this case
    x_kk1 = rk38(kf_fx, x_k1k1, uk, [ti, tf]) + Psi @ uk
    z_kk1 = kf_hx(tf, x_kk1, uk)
    z_pred[k] = z_kk1.T
    P_kk1 = Phi @ P_k1k1 @ Phi.T + Gamma @ Q @ Gamma.T

    # iterative part
    eta1 = x_kk1
    err = 2*epsilon
    its = 0
    for its in range(maxIter):
        its += 1
        err = 0

        dHx = kf_dhx(tf, eta1, uk)
        K = P_kk1 @ dHx.T @ np.linalg.inv(dHx @ P_kk1 @ dHx.T + R)
        z_k1k1 = kf_hx(tf, eta1, uk)  # predicted output
        x_k1k1 = x_kk1 + K @ (z_kk1 - z_k1k1 - dHx@(x_kk1 - eta1))
        for i in range(len(x_k1k1)):
            err += fabs((x_k1k1-eta1)[i,0]**2)/fabs(eta1[i,0]**2)
        eta1 = x_k1k1
        if err < epsilon:
            # Break the loop if error < maximum allowed error
            break

    if k==1000:
        print('Were iterations performed [0 = no, >0 = yes]:',sum(iteration_counter) - k)
        breakpoint()

    iteration_counter.append(its)

    P_k1k1 = (np.identity(len(x)) - K @ dHx) @ P_kk1 @ (np.identity(len(x)) - K@dHx).T + K @ R @ K.T
    ti = tf
    tf = tf + dt
    xx_k1k1[k] = x_k1k1[:,0]
    for i in range(len(x_k1k1)):
        stdx_corr[k, i] = P_k1k1[i, i]

"""
"""
"""
    XX_k1k1(:,k) = x_k_1k_1;
%     PP_k1k1(k,:) = P_k_1k_1;
    STDx_cor(:,k) = stdx_cor;
end

% calculate state estimation error (in real life this is unknown!)
EstErr = (XX_k1k1-X_k);

fprintf('IEKF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr)), N, time2);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

plotID = 1000;
figure(plotID);
set(plotID, 'Position', [1 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k(1,:), 'b');
plot(X_k(2,:), 'b--');
plot(Z_k(1,:), 'k');
% plot(Z_k(2,:), 'k--');
title('True state (blue) and Measured state (black)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateMeasurement');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1001;
figure(plotID);
set(plotID, 'Position', [1 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k(1,:), 'b');
plot(X_k(2,:), 'b--');
plot(XX_k1k1(1,:), 'r');
plot(XX_k1k1(2,:), 'r--');
title('True state (blue) and Estimated state (red)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateEstimates');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end



plotID = 2001;
figure(plotID);
set(plotID, 'Position', [800 100 600 600], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
subplot(2, 1, 1);
plot(EstErr(1,:), 'b');
title('State 1 estimation error');
subplot(2, 1, 2);
plot(EstErr(2,:), 'b');
title('State 2 estimation error');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimateErrors');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 2002;
figure(plotID);
set(plotID, 'Position', [800 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
subplot(2, 1, 1);
plot(EstErr(1,:), 'b');
axis([0 50 min(EstErr(1,:)) max(EstErr(1,:))]);
title('State 1 estimation error (Zoomed in)');
subplot(2, 1, 2);
plot(EstErr(2,:), 'b');
axis([0 50 min(EstErr(2,:)) max(EstErr(2,:))]);
title('State 2 estimation error (Zoomed in)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimateErrorsZoom');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


% plotID = 2003;
% figure(plotID);
% set(plotID, 'Position', [1000 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% hold on;
% plot(EstErr, 'b');
% plot(STDx_cor, 'r');
% plot(-STDx_cor, 'g');
% legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
% title('State estimation error with STD of Innovation');
% if (printfigs == 1)
%     fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
%     savefname = strcat(figpath, fpath);
%     print(plotID, '-dpng', '-r300', savefname);
% end
%
%
% plotID = 2004;
% figure(plotID);
% set(plotID, 'Position', [1000 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
% hold on;
% plot(EstErr, 'b');
% plot(STDx_cor, 'r');
% plot(-STDx_cor, 'g');
% axis([0 50 min(EstErr) max(EstErr)]);
% title('State estimation error');
% legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
% if (printfigs == 1)
%     fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
%     savefname = strcat(figpath, fpath);
%     print(plotID, '-dpng', '-r300', savefname);
% end

plotID = 3001;
figure(plotID);
set(plotID, 'Position', [1 700 600 300], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(IEKFitcount, 'b');
title('IEKF iterations at each sample');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end
    """