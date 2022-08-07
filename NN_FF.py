import numpy as np


class NN():
    def forward_nn(self, X):
        # propagate inputs through network
        self.vj = X @ self.IW.T + self.bias1.T
        self.yj = self.act_func(self.vj)
        self.vk = self.yj @ self.LW.T + self.bias2.T
        yHat = self.vk  # linear function in output neuron -> no change
        return yHat

    def act_func(self, v):
        return 2 / (1 + np.exp(-2 * v)) - 1

    def act_func_dv(self, v):
        return 4 * np.exp(-2 * v) / ((1 + np.exp(-2 * v)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward_nn(X)
        e = 0.5 * np.sum((y - self.yHat) ** 2, axis=0)
        return e

    # back-prop step:
    def CostFuncPrime(self, inputs, y):
        # Derivative of the error wrt the weights
        # Error in this case is the sum of squared errors.
        self.yHat = self.forward_nn(inputs)

        dEdy = -(y - self.yHat)  # 1000x1
        dEdvk = dEdy

        dvkdbias2 = dEdvk * 0 + 1
        dvkdLW = self.yj
        dEdLW = dEdvk.T @ dvkdLW  # OxU @ UxH
        dEdbias2 = dEdvk.T @ dvkdbias2  # Ox1 size

        dEdyj = dEdvk @ self.LW  # sums across k
        dEdvj = dEdyj * self.act_func_dv(self.vj)

        dvjdIW = inputs
        dvjdbias1 = dvkdbias2  # 1 wide "ones" array for all data entries
        dEdIW = dEdvj.T @ dvjdIW  # HxU @ UxI -> HxI
        dEdbias1 = dEdvj.T @ dvjdbias1  # Hx1 size

        self.dLW = -self.Opts.mu * dEdLW
        self.dbias2 = -self.Opts.mu * dEdbias2
        self.dIW = -self.Opts.mu * dEdIW
        self.dbias1 = -self.Opts.mu * dEdbias1

    def AdaptiveTraining(self, inputs, y):
        e1 = self.costFunction(inputs, y)
        self.CostFuncPrime(inputs, y)
        self.LW = self.LW + self.dLW
        self.bias2 = self.bias2 + self.dbias2
        self.IW = self.IW + self.dIW
        self.bias1 = self.bias1 + self.dbias1
        e2 = self.costFunction(inputs, y)
        if self.Opts.mu_delta < 1:
            if e2 < e1:  # E_t+1 < E_t
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta  # increase mu
                return e2
            else:
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                self.bias2 = self.bias2 - self.dbias2
                self.IW = self.IW - self.dIW
                self.bias1 = self.bias1 - self.dbias1
                return e1
        else:
            if e2 < e1:  # E_t+1 < E_t
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta  # increase mu
                return e2
            else:
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                self.bias2 = self.bias2 - self.dbias2
                self.IW = self.IW - self.dIW
                self.bias1 = self.bias1 - self.dbias1
                return e1
        # Finishes update step either way


# Initializations
"""
# IW/bias1/LW Reasoning:
# 1. Put a line wherever the value is 0? Problem: what when there is another sigmoid function altering that 0.
#   1.1. Put a line where the second derivative of the local "curve" is 0.
#       The problem remains that the entire data set has to be re-formatted. Not smart. Think simpler.
#       Go more complex and its basically an SVM, rather than an FF NN.
#       Also local sigmoid form can change the behaviours.
#   1.2. Simple greedy approach -> magnitude-based or sign-based? Also -- "proximity"
#
# 2. Upper part faces the side that is more positive than the other in a "proximity"
#   "proximity" is based on number of lines - think k-means
#   maybe think 4 lines, each in cardinal directions, and then reduce "proximity" for every other 4?
#   maybe think grid layout? Starting "growth" from the center? Then mostly just self.Opts.range can be used?
#
# 3. Which line locations take priority? Using a local exhaustive search is garbage.
# The y=0 line is at a1x1 + a2x2 + ... + b = 0. (a = IW, b = bias1)
# the higher the magnitude of IW, the steeper the "s" curve
# LW changes the asymptotic values of the "s" curve.
"""
class NNPart1_rand(NN):
    def __init__(self, Opts, inputs, outputs):
        self.Opts = Opts
        np.random.seed(self.Opts.seed)
        # Weights (IW = input weights, LW = output weights)
        self.bias2 = np.array([outputs.mean(0)]).T  # average of the data.

        # random -- sometimes best final result, but almost always worst start
        self.IW, self.bias1 = self.randIW()
        self.LW = self.randLW()

    # slightly modified Xavier init.
    def randIW(self):
        ranges = self.Opts.range[:, 1] - self.Opts.range[:, 0]
        IW = np.random.randn(self.Opts.hiddenlayersize, self.Opts.inputlayersize) * np.sqrt(1/self.Opts.inputlayersize)
        bias1 = (np.random.randn(self.Opts.hiddenlayersize, 1) - 0.5) * ranges.sum()
        IW = IW * ranges
        bias1 = bias1 - self.Opts.range.mean()
        return IW, bias1

    def randLW(self):
        LW = np.random.randn(self.Opts.outputlayersize, self.Opts.hiddenlayersize) * np.sqrt(1/self.Opts.hiddenlayersize)
        return LW



class NNPart1_alg(NN):
    def __init__(self, Opts, inputs, outputs):
        self.Opts = Opts
        self.bias2 = np.array([outputs.mean(0)]).T  # average of the data.

        # algorithm - This isn't my thesis, lets just stop here. -- sometimes best final result
        self.IW, self.bias1 = self.covIW(inputs)
        self.LW = self.lowIQLW(outputs)

        # # algorithm - semi-radial initialization
        # self.IW, self.bias1 = self.midIQ(inputs)
        # self.LW = self.lesslowIQLW(outputs)

        # # algorithm - Grid-form, but slightly better than just randomizing.
        # self.IW, self.bias1 = self.lowIQgrid()
        # self.LW = self.lowIQLW(outputs)

    def covIW(self, inputs):
        """
        Function is based on maxing the coverage of data.
        Attempting to make the neurons intersect as much as possible. TBD:
        # 1. determine areas that need/can be more focussed on (data priority?).
        # 2. maximize GOOD irregular triangulation
        # ^ 2.a. can't use unstructured grid algorithms obv.
        # 3. Improve the speed by using a better method.
        Result:
        # For now just use a pre-mathematized maximum coverage method... modified for lines... Options are:
        # 1. A greedy algorithm
        # 2. A codified "http://www.cs.technion.ac.il/~rcohen/PAPERS/GMCP.pdf"?
        # 3. Lazy greedy; "exhaustive" search #currentstyle
        """
        # Use "best fit" line for a greedy approach. "maximum likelihood estimator" style? Too complex.
        # Hinge-loss function? Hat-function?
        # No, maximize sum[weights(dist<width)], dist = f(line,pos), line = f(angle,bias)
        # 2-variable optimization in essence. Un-definable problem though; use an exhaustive approximate search.

        # Regardless, orthogonal distances = (a * xn + b * yn + c)/sqrt(a^2 + b^2)
        # 1. Create a modified data set with weights
        weights = np.ones([inputs.shape[0], 1, 1])  # will end up modifying test data.
        line_storage = np.zeros([0, 2])  # store best line configurations as an index value

        # 2. Set up variables for iteration -- A=#angles, B=#bias, U=#data
        angles = np.linspace(0, np.pi, self.Opts.covIW_angles + 1)[:-1]
        angles = np.array([np.cos(angles), np.sin(angles)]).T  # shape = Ax2
        # not fully convinced with this logic: why punish anisotropy?
        widths = np.absolute(angles) @ np.std(inputs, axis=0).T / (self.Opts.covIW_angles+self.Opts.covIW_biases)
        width_mod = +1
        # widths = np.ones(15)/(np.sqrt(self.Opts.hiddenlayersize)/np.sqrt(2))
        dist = (angles @ inputs.T).T  # shape = UxA
        bias = np.linspace(dist.min(0), dist.max(0), self.Opts.covIW_biases)  # shape = AxB

        test = np.absolute(dist[:, np.newaxis, :] - bias[np.newaxis, :, :])
        test = test < widths

        # 3. Iterate for each line
        for _ in range(self.Opts.hiddenlayersize):
            weighted_test = test*weights  # modify test results with weights
            # mby improvable with tie-breaking for finding best "unused" angle/bias combination
            line = np.argmax(weighted_test.sum(0))
            # line needs to be broken down into 2 indices
            bias_index = int(line / self.Opts.covIW_angles)
            angle_index = line % self.Opts.covIW_angles
            widths[angle_index] = 1/(1/widths[angle_index] + width_mod)  # inverse of line 184-198
            # store best results
            line_storage = np.append(line_storage, [[bias_index, angle_index]], axis=0)
            # Weight modification -- maybe find a way to avoid transposing so often...
            weights = (weights.T - weighted_test[:, bias_index, angle_index] * (1 - self.Opts.covIW_weightmod)).T

        # 4. Determine magnitude of IW -> larger == steeper.
        # sqrt 2 used cause there are 2 entries for the angle, because magnitude == norm(IW)
        IW_angles = angles[line_storage[:, 1].astype(int)]
        bias1 = np.array([bias[line_storage[:, 0].astype(int), line_storage[:, 1].astype(int)]]).T

        # modifies weight based on frequency of appearance
        IW_used = []
        mod = []
        for i in line_storage[:, 1]:
            IW_used.append(i)
            if i not in IW_used:
                mod.append(1)
            else:
                x = np.sum(IW_used == i)
                if int(x) % 2 == 0:
                    mod.append(-x)
                else:
                    mod.append(x)
        IW = (IW_angles.T * mod).T
        IW = IW * np.array([IW_angles @ np.std(inputs, axis=0).T]).T
        return IW, bias1
    # ### Reading only...:
    # # bias1, based on maximizing fit (magnitude fit, simply "+" "-" fit, or agnostic fit?)
    # # Essentially the point is to maximize the CCW "moment" considering the direction of IW_angles
    # # Maybe this will be changed to coverage rather than fit in a "proper initialization"
    # """
    # Given: a*x + b*y + c = 0, and points (xn,yn) (https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line)
    # orthogonal distances = (a * xn + b * yn + c)/sqrt(a^2 + b^2)
    # orthogonal_distances = [(IW[i,0]*x + IW[i,1]*y + bias1[i])/norm(IW[i]) for i in range(self.Opts.hiddenlayersize)]
    # orthogonal_distances = (IW[:,0]*x + IW[:,1]*y + self.bias1) / np.linalg.norm(IW,axis=1)
    # CCW moment = values * orthogonal_distances
    # -> bias1 = values
    #
    # Then whats left is to find max CCW moment for bias1
    # Problem: max CCW will probably be for bias -inf or +inf... i.e. bad method
    # Solution: Penalize distance:
    # Problem: S shape of sigmoid means the lower the distance the less the effect of the function.
    # Therefore (imo) there isn't a simple "fitting" solution besides SVM, but hey, lets work with what we've got.
    # """
    #
    # # NOTE: lines refer to the sequence of points = 0 in the Sigmoid surface
    # # ## OK ideas
    # # 2. Couple the lines to make a "semi-local" hat-like function, "hat" peak used to describe local phenomena
    # #    Maybe maximize orthogonality of hats, but still position them to maximize fit to data.
    # #    Instead, its possible to just use a "coverage" scheme -- doesn't need output data, so its better.
    # #    Just try to get the non-isotropic part of the lines as much as possible
    # # 4. greedy alg. Maximize fit per line initialization. SVM style(is that even needed?)
    # # - use input anisotropy for line directions? i.e. Less lines in long direction, more lines in short direction.
    # # Better: use uniformly distributed line directions based on # of lines
    # # # place similar direction lines (+-180 deg.) apart from one another.
    # # ## Trash ideas, but may have some merit
    # # 1. Find k-means centers, ensure that the lines pass through no more than 1 center, etc. etc.
    # # 3. Each line has its own angle, and width and LW, but positioning (bias1) is ols solved.
    # # 5. Distribute line directions based on PCA, and etc.

    def midIQ(self, inputs):
        # IW ratio btw x1 and x2, based on uniform angle distribution        # Mby think of IW_angles ** 2?
        IW_angles = np.array([[np.cos(x), np.sin(x)] for x in np.linspace(0, 3*np.pi, self.Opts.hiddenlayersize + 1)])
        IW_angles = IW_angles[:-1]  # exclude "pi" as it is in the same direction as "0", which is why the '+ 1' is used

        # base IW off the standard deviation of the input data.
        # Maybe use a different standard deviation or a multiple of it? Maybe the inverse cause its tanh(IW*data)?
        IW = (np.std(inputs, axis=0)) * IW_angles
        bias1 = np.ones([self.Opts.hiddenlayersize, 1])
        bias1 = bias1*np.sqrt(np.array([np.linalg.norm(IW, axis=1)]).T)
        # The square root was added for a better fit, slightly more chaotic.
        return IW, bias1

    def lesslowIQLW(self, outputs):
        # alternating [1,-1,1,-1] array - prevents stacking the positives on one end. and negatives to the other
        a = np.empty(self.Opts.hiddenlayersize,)
        a[::2] = 1
        a[1::2] = -1

        # # way1:
        LW_almost = np.ones([int(self.Opts.outputlayersize), int(self.Opts.hiddenlayersize)])*a
        # Ensure variance in weights, such that training can progress smoothly.
        # create range from 0.5 to 1.5?
        # LW_almost = LW_almost * np.linspace(0.8, 1.2, self.Opts.hiddenlayersize)
        # # way 2
        # LW_almost = np.random.randn(int(self.Opts.outputlayersize), int(self.Opts.hiddenlayersize)) * a
        return 3*np.std(outputs-self.bias2)*LW_almost

    def lowIQLW(self, outputs):
        # alternating [1,-1,1,-1] array - prevents stacking the positives on one end. and negatives to the other
        a = np.empty(self.Opts.hiddenlayersize,)
        a[::2] = 1
        a[1::2] = -1

        LW_almost = np.ones([int(self.Opts.outputlayersize), int(self.Opts.hiddenlayersize)])*a
        return 3*np.std(outputs-self.bias2)*LW_almost

    def lowIQgrid(self):
        hid = self.Opts.hiddenlayersize
        if hid % 2 == 0:
            num_hor_lines = hid / 2
            num_ver_lines = hid / 2
        else:
            num_hor_lines = round(hid / 2) - 1
            num_ver_lines = round(hid / 2)

        x_dist = np.absolute(self.Opts.range[0]).sum() / num_ver_lines
        y_dist = np.absolute(self.Opts.range[1]).sum() / num_hor_lines
        x_pos = np.linspace(self.Opts.range[0, 0] + x_dist/2, self.Opts.range[0, 1] - x_dist/2, int(num_ver_lines))
        y_pos = np.linspace(self.Opts.range[1, 0] + y_dist/2, self.Opts.range[1, 1] - y_dist/2, int(num_hor_lines))

        # The higher modifier value the steeper the S curve is
        mod_y = 1/y_dist
        mod_x = 1/x_dist

        bias1 = -np.append(np.array([x_pos]) * mod_x, np.array([y_pos]) * mod_y, axis=1).T

        rand_x = (np.random.rand(x_pos.shape[0]) + 1)/2
        rand_y = (np.random.rand(y_pos.shape[0]) + 1)/2
        x_IW = np.array([rand_x, rand_x/100]) * mod_x
        y_IW = np.array([rand_y/100, rand_y]) * mod_y

        IW = np.append(x_IW, y_IW, axis=1).T
        bias1 = bias1
        return IW, bias1
