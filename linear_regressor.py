# Multiple linear regression fit with an ordinary least squares error model
# returns a function y = ax + b with a best fit

# y = X@beta + err;
# y = vector of all output measurements aka observations
# X = matrix of all state measurements aka predictor variables
# X looks like: [1 x11 x12 ... x1n]
#               | ... ... ... ... |
#               [1 xn1 xn2 ... xnn]
# beta = vector of parameter estimates (b0 = mean, b1 = slope for x1, etc.)
# err = vector of errors to be minimized

#        if np.size(x, axis=0) != np.size(y, axis=0):
#            print("lin_reg_multiple: All x need to be same length vector as y \n")
#            print("vector is an np array of the form [[2],[3],[4]] (double-bracketed, yes) \n")
#            breakpoint()

import numpy as np


def combine(order, *args):
    x_matrix = np.zeros([np.size(args[0], axis=0), 1]) + 1  # mean
    fro = [0]*len(args)  # sets start of combine turn
    k = 0  # counts number of times the arguments to be combined have switched
    its = 0
    # Repeats "x" times for the "x"th degree polynomial expansion (x = order)
    for i in range(order):
        too = np.size(x_matrix, axis=1)
        for x in args:
            for index in range(fro[k], too):
                x_matrix = np.append(x_matrix, x*np.array([x_matrix[:, index]]).T, axis=1)
                its += 1
            k += 1
            fro.append(its)
    return x_matrix


def lin_reg_multiple(y, x_matrix):
    beta = np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y
    err = y - x_matrix @ beta
    cov_ols = (x_matrix.T @ x_matrix) * (err.T @ err) / (len(y) - np.size(x_matrix, axis=1))
    return beta, err, cov_ols


"""
def lin_reg_multiple_weighted(y, noise, *args):
    x_matrix = np.zeros([np.size(y, axis=0), 1]) + 1  # to find the mean
    for x in args:
        x_matrix = np.append(x_matrix, x, axis=1)

#    if noise is a list then:
#        W = np.linalg.inv(np.diag(noise))
#    elif noise is already a diagonal matrix:
#        W = np.linalg.inv(noise)
    beta = np.linalg.inv(x_matrix.T @ W @ x_matrix) @ x_matrix.T @ W @ y
    err = y - x_matrix@beta
    cov_ols = (x_matrix.T @ x_matrix) * (err.T @ err) / (len(y) - np.size(x_matrix, axis=1))
    return beta, err, cov_wls
"""
"""
def lin_reg_multiple_generalized(y, *args):
    [beta, err, cov_ols] = lin_reg_multiple(y, *args)
    
#    sigma = np.linalg.inv(err@err.T) # <- except not really
    beta_gls = np.linalg.inv(x_matrix.T @ sigma @ x_matrix) @ x_matrix.T @ sigma @ y
    err_gls = y - x_matrix @ beta_gls
    return beta_gls, err_gls
"""

# http://mezeylab.cb.bscb.cornell.edu/labmembers/documents/supplement%205%20-%20multiple%20regression.pdf
# http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
# parameter estimation slides, in particular slide 88
