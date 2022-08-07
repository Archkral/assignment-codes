from Part3.NN_RBF import NNPart1_alg, NNPart1_rand
import numpy as np


########################################################################################################################
# Part 3.2 (Levenberg-Marquardt for all 3 "weights" at once)
class NNPart2:
    def Jacobian(self, inputs, y):
        # inputs SHOULD INCLUDE biases.
        # Derivative of the error wrt the weights
        # Error in this case is the sum of squared errors.
        self.yHat = self.forward_rbf(inputs)  # UxO size (U = data entries, O = outputs)

        e = (y - self.yHat)  # UxO
        dEdyHat = -(y - self.yHat)  # UxO
        dEdvk = dEdyHat  # UxO
        # Should be summed separately for multiple outputs in a loop to create an array of O*H width (LW size).
        dEdLW = dEdvk * self.yj  # UxO*UxH -> Ux(O*H) -> UxH (only because 1 output in this case)
        dEdyj = dEdvk @ self.LW  # UxO@OxH -> UxH
        dEdvj = dEdyj * self.act_func_dz(self.vj)  # UxH * UxH -> term by term multiple

        Nin = self.Opts.inputlayersize  # I inputs
        L_end = np.size(inputs, axis=0)  # U data entries
        Nhidden = self.Opts.hiddenlayersize  # H hidden neurons

        dEdIW = np.zeros([L_end, Nin * Nhidden])  # IxH weights, U data entries
        dEdcenters = np.zeros([L_end, Nin * Nhidden])
        for j in range(Nin):  # Order is correct.
            xc = np.array([inputs[:, j]]).T - np.array([self.centers[:, j]]) * np.ones([L_end, 1])
            dvjdIW = -(xc ** 2)
            dvjdc = - self.IW[:, j] * -2 * xc
            dEdIW[:, Nhidden * j:Nhidden * (j + 1)] = dvjdIW * dEdvj
            dEdcenters[:, Nhidden * j:Nhidden * (j + 1)] = dvjdc * dEdvj

        # Set up Jacobian
        J = dEdLW
        J = np.append(J, dEdIW, axis=1)
        J = np.append(J, dEdcenters, axis=1)
        self.J = J
        # deeper neural net possible by expanding jacobian with another layer of derivations

    def CostFuncPrime(self, inputs, y):
        self.Jacobian(inputs, y)
        dW = np.linalg.inv((self.J.T @ self.J) + self.Opts.mu * np.identity(np.size(self.J, axis=1))) @ (self.J.T @ (y - self.yHat))

        # Change dW into input and output weight form
        dW = np.ravel(dW)  # change it into a 1D array form

        # unravel weight array - change it into a 2D form
        k = self.Opts.outputlayersize * self.Opts.hiddenlayersize
        self.dLW = np.reshape(dW[0:k], [self.Opts.outputlayersize, self.Opts.hiddenlayersize])
        k2 = (self.Opts.outputlayersize + self.Opts.inputlayersize) * self.Opts.hiddenlayersize
        self.dIW = np.reshape(dW[k:k2], [self.Opts.hiddenlayersize, self.Opts.inputlayersize], 'F')
        self.dcenters = np.reshape(dW[k2:], [self.Opts.hiddenlayersize, self.Opts.inputlayersize], 'F')

    def AdaptiveTraining(self, inputs, y):
        e1 = self.costFunction(inputs, y)
        self.CostFuncPrime(inputs, y)
        self.IW = self.IW + self.dIW
        self.LW = self.LW + self.dLW
        self.centers = self.centers + self.dcenters
        e2 = self.costFunction(inputs, y)
        if self.Opts.mu_delta > 1:
            if e2 < e1:  # Successful step == E_t+1 < E_t
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.IW = self.IW - self.dIW
                self.LW = self.LW - self.dLW
                self.centers = self.centers - self.dcenters
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta  # increase mu
                return e1
        else:
            if e2 < e1:  # Successful step == E_t+1 < E_t
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.IW = self.IW - self.dIW
                self.LW = self.LW - self.dLW
                self.centers = self.centers - self.dcenters
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta  # increase mu
                return e1
        # Finishes update step either way

    """ Simple gradient descent back-prop """
    def CostFuncPrimeSimpleBackprop(self, inputs, y):
        self.Jacobian(inputs, y)
        dW = -self.Opts.mu * self.J.sum(0)
        # unravel weight array - change it into a 2D form
        k = self.Opts.outputlayersize * self.Opts.hiddenlayersize
        self.dLW = np.reshape(dW[0:k], [self.Opts.outputlayersize, self.Opts.hiddenlayersize])
        k2 = (self.Opts.outputlayersize + self.Opts.inputlayersize) * self.Opts.hiddenlayersize
        self.dIW = np.reshape(dW[k:k2], [self.Opts.hiddenlayersize, self.Opts.inputlayersize], 'F')
        self.dcenters = np.reshape(dW[k2:], [self.Opts.hiddenlayersize, self.Opts.inputlayersize], 'F')

    def AdaptiveBackprop(self, inputs, y):
        e1 = self.costFunction(inputs, y)
        self.CostFuncPrimeSimpleBackprop(inputs, y)
        self.IW = self.IW + self.dIW
        self.LW = self.LW + self.dLW
        self.centers = self.centers + self.dcenters
        e2 = self.costFunction(inputs, y)
        if self.Opts.mu_delta < 1:  # "<" instead of ">"
            if e2 < e1:  # Successful step == E_t+1 < E_t
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.IW = self.IW - self.dIW
                self.LW = self.LW - self.dLW
                self.centers = self.centers - self.dcenters
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta  # increase mu
                return e1
        else:
            if e2 < e1:  # Successful step == E_t+1 < E_t
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.IW = self.IW - self.dIW
                self.LW = self.LW - self.dLW
                self.centers = self.centers - self.dcenters
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta  # increase mu
                return e1
        # Finishes update step either way


class NNPart2_alg(NNPart2, NNPart1_alg):
    pass


class NNPart2_rand(NNPart2, NNPart1_rand):
    pass
