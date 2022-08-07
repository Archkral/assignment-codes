from Part3.NN_RBF_LM import NNPart2_alg
import numpy as np

class NNPart3(NNPart2_alg):
    # WARNING: np.around SHOULD HAVE A DECIMAL VALUE SET FOR THE USE-CASE
    def grow_step(self, X, y):
        self.Opts.hiddenlayersize += 1  # 1 more hidden neuron
        idx = np.argmax(np.absolute(y - self.yHat))  # idx of highest error
        self.LW = np.concatenate((self.LW, [y[idx, :] - self.yHat[idx, :]]), axis=1)  # value of said error
        # Center - location of highest error
        self.centers = np.concatenate((self.centers, [X[idx, :]]), axis=0)
        self.IW = np.concatenate((self.IW, [self.IW_karlo(X)[-1, :]]), axis=0)

    def OBS_step(self, X, y):
        # Optimal brain surgeon step. Calculates least valuable weight using hessian matrix.
        # So simple, yet so good. Expansions for better performance are possible (read papers).
        # Also this is only proven for input and output weight cutting -> not for centers in RBFs.
        self.Jacobian(X, y)
        self.J = self.J[:, :-2*self.Opts.hiddenlayersize]  # exclude the centers! they are not to be trimmed!
        # Set up (ravel) weight array, and make sure to have the same order as the jacobian:
        W = self.LW.ravel()
        W = np.append(W, self.IW.ravel())
        # W = np.append(W, self.centers.ravel())
        W = np.around(W, decimals=20)

        # W_logic is used to re-set dead connections near the end.
        W_logic = (W != 0.0)
        # masked_W is to avoid re-selecting a "0" as the lowest in saliency
        masked_W = np.ma.masked_equal(W, 0.0, copy=False)

        # Inverted Hessian, and the OBS formulas
        H_inv = np.linalg.inv(self.J.T @ self.J + np.eye(self.J.shape[1])*10**-6)  # 10^-6 is to make it significant.
        q = np.argmin(0.5 * masked_W**2/H_inv.diagonal())  # the smallest "saliency"
        dW = -W[q]/H_inv[q, q] * H_inv[:, q]
        W = W + dW

        # Ensure no float rounding errors and keeping dead neuron pathways "dead"
        W = np.around(W, decimals=20)
        W = W_logic*W

        # unravel weight array
        k = self.Opts.outputlayersize * self.Opts.hiddenlayersize
        LW = np.reshape(W[0:k], [self.Opts.outputlayersize, self.Opts.hiddenlayersize])
        k2 = (self.Opts.outputlayersize + self.Opts.inputlayersize) * self.Opts.hiddenlayersize
        IW = np.reshape(W[k:k2], [self.Opts.hiddenlayersize, self.Opts.inputlayersize])
        # centers = np.reshape(W[k2:], [self.Opts.hiddenlayersize, self.Opts.inputlayersize])

        # Fully remove the neuron if its index is within the LW area:
        if q < self.Opts.hiddenlayersize:  # correct: q = 0 for 1 hidden rbf.
            logic = np.ones([1, self.Opts.hiddenlayersize], dtype=bool)  # data type == integer as its a logic array
            logic[:, q] = 0
            self.Opts.hiddenlayersize -= 1
            LW = np.array([LW[logic]])
            IW = IW[logic[0]]
            self.centers = self.centers[logic[0]]
            # centers = centers[logic[0]]
        # # The above is better than to remove the neuron if the output weight is (in)directly set to 0:
        # logic = (LW != 0)
        # if np.sum(logic) < self.Opts.hiddenlayersize:
        #     self.Opts.hiddenlayersize = np.sum(logic)
        #     LW = np.array([LW[logic]])
        #     IW = IW[logic[0]]
        #     # centers = centers[logic[0]]
        # return q, LW, IW, centers  # proposed values
        return q, LW, IW  # proposed values
