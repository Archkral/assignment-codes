from Part4.NN_FF import NNPart1_alg, NNPart1_rand
import numpy as np


# Part 4.2 (Levenberg-Marquardt for all 3 "weights" at once)
class NNPart2:
    def Jacobian(self, inputs, y):
        # Constants with shorter names, for legibility:
        n_in = self.Opts.inputlayersize  # I inputs
        l_end = np.size(inputs, axis=0)  # U data entries
        n_hidden = self.Opts.hiddenlayersize  # H hidden neurons
        n_out = self.Opts.outputlayersize  # O outputs

        # inputs SHOULD INCLUDE biases.
        # Derivative of the error wrt the weights
        # Error in this case is the sum of squared errors.
        self.yHat = self.forward_nn(inputs)

        e = (y - self.yHat)
        dEdy = -e  # UxO, derivative of the error wrt the output.
        dEdvk = dEdy  # UxO, derivative of the error wrt the activation function

        # dEdbias2 = dEdvk  # UxO derivative of the error wrt the output bias
        dEdLW = np.zeros([l_end, n_out * n_hidden])  # Ux(O*H), derivative of the error wrt the output weights
        for i in range(n_out):
            dEdLW[:, i * n_hidden:(i + 1) * n_hidden] = np.array([dEdvk[:, i]]).T * self.yj

        dEdyj = dEdvk @ self.LW  # UxH, derivative of the error wrt the output of the hidden network
        dEdvj = dEdyj * self.act_func_dv(self.vj)  # UxH, derivative of the error wrt the input of the hidden network

        dvjdIW = inputs  # UxI, derivative of the input of the hidden network wrt the input weights
        dEdIW = np.zeros([l_end, n_hidden * n_in])  # Ux(I*H), derivative of the error wrt the input weights
        for i in range(n_in):
            dEdIW[:, i * n_hidden:(i + 1) * n_hidden] = dEdvj * np.array([dvjdIW[:, i]]).T
        # dEdbias1 = dEdvj

        # self.dLW = -self.Opts.mu * dEdLW
        # self.dbias2 = -self.Opts.mu * dEdvk
        # self.dIW = -self.Opts.mu * dEdIW
        # self.dbias1 = -self.Opts.mu * dEdvj

        # Set up Jacobian
        J = dEdLW
        J = np.append(J, dEdvk, axis=1)  # bias2
        J = np.append(J, dEdIW, axis=1)
        self.J = np.append(J, dEdvj, axis=1)  # bias1

        # # Gradient descent test (does everthing work so far? Yes)
        # dW = -np.sum(J,axis=0) * self.Opts.mu

    def LM_CostFuncPrime(self, inputs, y):
        self.Jacobian(inputs, y)
        # LM method:  http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
        dW = -np.linalg.inv(self.J.T @ self.J + self.Opts.mu * np.identity(self.J.shape[1])) @ self.J.T @ (y - self.yHat)

        # Change dW into input and output weight form
        dW = np.ravel(dW)  # change it into a 1D array form out of a "nominally" 2D array

        # unravel weight array - reformat results into proper 2D form
        O = self.Opts.outputlayersize
        H = self.Opts.hiddenlayersize
        I = self.Opts.inputlayersize
        k = O * H
        self.dLW = np.reshape(dW[0:k], [O, H], 'F')
        k2 = O * H + O
        self.dbias2 = np.reshape(dW[k:k2], [O, 1], 'F')
        k3 = (O + I) * H + O
        self.dIW = np.reshape(dW[k2:k3], [H, I], 'F')
        self.dbias1 = np.reshape(dW[k3:], [H, 1], 'F')
        # deeper neural net possible by expanding jacobian with derivations


    def LM_AdaptiveTraining(self, inputs, y):
        # Inverts the mu_delta logic compared to the normal network.
        e1 = self.costFunction(inputs, y)
        self.LM_CostFuncPrime(inputs, y)
        self.LW = self.LW + self.dLW
        self.bias2 = self.bias2 + self.dbias2
        self.IW = self.IW + self.dIW
        self.bias1 = self.bias1 + self.dbias1
        e2 = self.costFunction(inputs, y)
        if self.Opts.mu_delta < 1:  # THIS PART IS INVERTED!
            if e2 < e1:  # E_t+1 < E_t
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                self.bias2 = self.bias2 - self.dbias2
                self.IW = self.IW - self.dIW
                self.bias1 = self.bias1 - self.dbias1
                return e1
        else:
            if e2 < e1:  # E_t+1 < E_t
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                self.bias2 = self.bias2 - self.dbias2
                self.IW = self.IW - self.dIW
                self.bias1 = self.bias1 - self.dbias1
                return e1
        # Finishes update step either way

    def CostFuncPrimeSimpleBackprop(self, inputs, y):
        self.Jacobian(inputs, y)
        dW = -self.Opts.mu * self.J.sum(0)
        # Change dW into input and output weight form
        dW = np.ravel(dW)  # change it into a 1D array form out of a "nominally" 2D array

        # unravel weight array - reformat results into proper 2D form
        O = self.Opts.outputlayersize
        H = self.Opts.hiddenlayersize
        I = self.Opts.inputlayersize
        k = O * H
        self.dLW = np.reshape(dW[0:k], [O, H], 'F')
        k2 = O * H + O
        self.dbias2 = np.reshape(dW[k:k2], [O, 1], 'F')
        k3 = (O + I) * H + O
        self.dIW = np.reshape(dW[k2:k3], [H, I], 'F')
        self.dbias1 = np.reshape(dW[k3:], [H, 1], 'F')
        # deeper neural net possible by expanding jacobian with derivations

    def AdaptiveBackprop(self, inputs, y):
        e1 = self.costFunction(inputs, y)
        self.CostFuncPrimeSimpleBackprop(inputs, y)
        self.LW = self.LW + self.dLW
        self.bias2 = self.bias2 + self.dbias2
        self.IW = self.IW + self.dIW
        self.bias1 = self.bias1 + self.dbias1
        e2 = self.costFunction(inputs, y)
        if self.Opts.mu_delta < 1:  # THIS PART IS INVERTED!
            if e2 < e1:  # E_t+1 < E_t
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                self.bias2 = self.bias2 - self.dbias2
                self.IW = self.IW - self.dIW
                self.bias1 = self.bias1 - self.dbias1
                return e1
        else:
            if e2 < e1:  # E_t+1 < E_t
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta  # decrease mu
                return e2
            else:
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                self.bias2 = self.bias2 - self.dbias2
                self.IW = self.IW - self.dIW
                self.bias1 = self.bias1 - self.dbias1
                return e1
        # Finishes update step either way


class NNPart2_alg(NNPart2, NNPart1_alg):
    pass


class NNPart2_rand(NNPart2, NNPart1_rand):
    pass
