import numpy as np


########################################################################################################################
# default RBF functions
# Contains the forward pass
class NN():
    def forward_rbf(self, X):
        # propagate inputs through network
        Nin = np.size(X, axis=1)  # I inputs
        L_end = np.size(X, axis=0)  # U data entries
        self.vj = np.sum([-(self.IW[:, i:i+1].T * (X[:, i:i+1] - np.ones([1, L_end]).T * self.centers[:, i:i+1].T) ** 2)
                          for i in range(Nin)], axis=0)
        self.yj = self.act_func(self.vj)  # UxH
        self.vk = self.yj @ self.LW.T  # UxH @ (OxH).T -> UxH # HxO -> UxO
        yHat = self.vk  # linear function in output neuron
        return yHat

    def act_func(self, v):
        # activation function (hidden neuron equation) :: sig = 1 / (1 + np.exp(-z))
        # If changing this, change the derivative, and the IW_karlo estimator. Search keyword: "KW:Change"
        return np.exp(v)  # has to be v or -v^2 else its not an RBF

    def act_func_dz(self, v):
        # derivative of activation function :: sig = np.exp(-z) / ((1 + .np.exp(-z)) ** 2)
        return np.exp(v)  # KW:Change -2*v*np.exp(-v**2)

    def costFunction(self, X, y):
        # Compute sum of squared errors (SSE) for given input (X), output (y). uses weights already stored in class.
        self.yHat = self.forward_rbf(X)
        e = 0.5 * np.sum((y - self.yHat) ** 2, axis=0)
        return e


########################################################################################################################
# OLS solver for part 1 and growing/pruning algorithm for part 3
class NNPart1(NN):
    def OLSInspired(self, inputs, y):
        """
        Note: The "inputs" variable for this part is not the data entries, but rather self.yj!
        That is because OLS works by using the data before and after it is modified with the to be optimized variable(s)

        beta = np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y
        err = y - x_matrix @ beta
        """
        self.costFunction(inputs, y)
        beta = np.linalg.pinv(self.yj.T @ self.yj) @ self.yj.T @ y  # tough to invert a singular matrix
        self.dLW = beta.T - self.LW  # stored for the "AdaptiveTraining" function
        self.LW = self.dLW + self.LW
        error = self.costFunction(inputs, y)
        return error

    def CostFuncPrime(self, inputs, y):
        # inputs SHOULD INCLUDE biases.
        # Derivative of the error wrt the weights
        # Error in this case is the sum of squared errors.
        self.yHat = self.forward_rbf(inputs)

        dEdy = (y - self.yHat)  # 1000x1
        dEdvk = -dEdy
        dEdLW = np.sum(self.yj * dEdvk, axis=0)  # 1000x1*1000x8 -> 1000x8. Summed to 1x8
        # Should be summed separately if multiple outputs in a loop to create an array of out*hid form (LW form).
        self.dLW = -self.Opts.mu * dEdLW

    def AdaptiveTraining(self, inputs, y):
        e1 = self.costFunction(inputs, y)
        self.CostFuncPrime(inputs, y)
        self.LW = self.LW + self.dLW
        e2 = self.costFunction(inputs, y)
        if self.Opts.mu_delta > 1:
            if e2 < e1:  # Successful step == E_t+1 < E_t
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta
                return e2
            else:
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                return e1
        else:
            if e2 < e1:  # Successful step == E_t+1 < E_t
                self.Opts.mu = self.Opts.mu / self.Opts.mu_delta
                return e2
            else:
                self.Opts.mu = self.Opts.mu * self.Opts.mu_delta
                self.LW = self.LW - self.dLW
                return e1


########################################################################################################################
# initializations (order is a mess cause of inheritance structure, sorry).

class NNPart1_rand(NNPart1):
    def __init__(self, Opts, inputs):
        self.Opts = Opts
        np.random.seed(self.Opts.seed)
        # Weights (IW = input weights, LW = output weights)
        self.IW = np.random.randn(self.Opts.hiddenlayersize, self.Opts.inputlayersize) * np.sqrt(1/self.Opts.inputlayersize)
        try:  # This is needed because GAP_RBF starts with a hiddenlayersize of 0.
            x = np.sqrt(1/self.Opts.hiddenlayersize)
        except:
            x = 1
        self.LW = np.random.randn(self.Opts.outputlayersize, self.Opts.hiddenlayersize) * x
        # Centers can be chosen arbitrarily or through K-means clustering or w/e (Schwenker p.441/p.3):
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.312&rep=rep1&type=pdf
        center_index = np.random.choice(np.size(inputs, axis=0), self.Opts.hiddenlayersize, replace=False)
        self.centers = inputs[center_index, :]


class NNPart1_alg(NNPart1_rand):
    def __init__(self, Opts, inputs):
        # k-means clustering for centers could be done to include PCA pre-processing:
        # reason why: http://ranger.uta.edu/~chqding/papers/KmeansPCA1.pdf
        # simply put eigenvalue decomposition -> reduces dimensionality -> easier clustering
        # Can even go further and "whiten" the data. However, its kept simple this time - no PCA nor whitening
        NNPart1_rand.__init__(self, Opts, inputs)

        if self.Opts.hiddenlayersize > 0:
            self.kmeans(inputs)  # sets centers
            self.IW = self.IW_karlo(inputs)
            # self.IW = self.PNN()
            # self.LW = np.ones(self.LW.shape)
            pass
        else:
            # in case there are 0 hidden layer units
            print('NN_RBF, NNPart1_alg, __init__: 0 hidden neurons - keeping "random initialization"')

    def kmeans(self, inputs):
        # amazing guide: http://flothesof.github.io/k-means-numpy.html
        for i in range(self.Opts.kmeans_iterations):
            distances = np.linalg.norm(inputs - self.centers[:, np.newaxis], axis=2)
            closest_point_idx = np.argmin(distances, axis=0)
            self.centers = np.array(
                [inputs[closest_point_idx == k].mean(axis=0) for k in range(self.Opts.hiddenlayersize)])
        ### Visualization:
        # # To check what center is responsible for what batch of input data:
        # k = ?  # the center's index
        # plt.plot(inputs[closest_point_idx == k, 0], inputs[closest_point_idx == k, 1], 'bo')
        # plt.plot(self.centers[k, 0], self.centers[k, 1], 'rx')
        # plt.show()

    def IW_karlo(self, inputs):
        """
        Rewritten IW_karlo to avoid re-doing calculations in separate functions
        Goal: Set IW so that at its cluster's mean distance, an amplitude of ~0.68 of its LW is achieved.
        """
        ### Part1: separate duplicates from original centers:
        n_sd = 1  # [-] number of standard deviations on a side
        all_centers = self.centers
        mod = np.zeros([0, 2])  # stores the order of centers and the modifier for the copies
        IW_used = []
        i = 0
        while all_centers.any():  # i.e. while its not empty
            x = all_centers[0]
            dist = np.linalg.norm(all_centers - x, axis=1) > self.Opts.dist
            mod = np.append(mod, [[np.where(dist.__invert__())[0][1:], i]], axis=0)
            all_centers = all_centers[dist]
            IW_used.append(x)
            i = i+1

        all_centers = self.centers  # the previous loop exhausted "all_centers"
        self.centers = np.array(IW_used)

        ### Part2: Do width and anisotropy calculations on the non-duplicate centers
        # Could use error weighted input weights, but that ruins the point of "pre-"processing
        distances = np.linalg.norm(inputs - self.centers[:, np.newaxis], axis=2)
        closest_point_idx = np.argmin(distances, axis=0)
        standard_dev = np.array([np.std(inputs[closest_point_idx == k] - self.centers[k, :], axis=0)
                                for k in range(self.centers.shape[0])])
        # means = np.array([(np.linalg.norm(inputs[closest_point_idx == k] - self.centers[k], axis=1)).mean()
        #                  for k in range(self.centers.shape[0])])
        # width = np.array([means, means]).T
        # modifier = np.array([1/np.linalg.norm(standard_dev, axis=1)]).T  # normalizes (unit size) the anisotropy
        # anisotropy = standard_dev*modifier
        # shape = width*anisotropy

        ### Part3: determine IW from shape
        # Some sort of 6 sigma accuracy idea (3 on either ends)
        # Because IW^2 is used it becomes 36 sigma^2? Feels good man.
        # KW:Change If IW^4 is used, sqrt the IW_almost :/
        IW_almost = 1 / (n_sd * standard_dev * 2)

        ### Part4: Calculate the copied centers' IW based on the original centers' IW.
        # Mostly just list re-constructing
        self.centers = all_centers
        order = []
        multiplier = []
        for x in np.flipud(mod):
            m = 0
            order.insert(0,x[1])
            multiplier.insert(0, 2**m)
            for i in list(x[0]):
                m = m + 1
                order.insert(i, x[1])
                multiplier.insert(i, 2**m)

        IW = IW_almost[order] * np.array([multiplier]).T
        return IW

    ####################################################################################
    # Heritage functions that aren't used.
    ####################################################################################
    def anisotropy(self, inputs):
        # 5. standard deviations from center position. - sets anisotropy
        # Considering IW is only x- and y- variant, and other directions aren't directly obtainable, this is a good fit.
        distances = np.linalg.norm(inputs - self.centers[:, np.newaxis], axis=2)
        closest_point_idx = np.argmin(distances, axis=0)
        standard_dev = np.array([np.std(inputs[closest_point_idx == k] - self.centers[k, :], axis=0)
                                for k in range(self.Opts.hiddenlayersize)])
        modifier = np.array([1/np.linalg.norm(standard_dev, axis=1)]).T  # normalizes (unit size) the anisotropy
        anisotropy = standard_dev*modifier
        return anisotropy

    def alg4(self, inputs):
        # Determines the IW based on the average distance of its "clustered" data
        # A list of the distances from all data entries to each center:
        distances = np.linalg.norm(inputs - self.centers[:, np.newaxis], axis=2)
        # index that shows which center is closest to the corresponding data entry:
        closest_point_idx = np.argmin(distances, axis=0)
        # Input weight calculation based on mean of distances
        means = np.array([(np.linalg.norm(inputs[closest_point_idx == k] - self.centers[k], axis=1)).mean()
                         for k in range(self.Opts.hiddenlayersize)])
        width = np.array([means, means]).T
        return width

    ### Functions that could be used alternatively:
    # # 2. average distance to its cluster's test data  - sets width (Schwenker's alg4)
    # # 3. average relative position to its center  - sets anisotropy
    # avg_relative_position = np.array([(inputs[closest_point_idx == k] - self.centers[k, :]).mean(0)
    #                              for k in range(self.Opts.hiddenlayersize)])
    # # 4. max_distance to a data input in its cluster  - sets anisotropy
    # max_dist_idx = np.array([np.argmax(np.linalg.norm(inputs[closest_point_idx == k] - self.centers[k, :],axis=1))
    #                          for k in range(self.Opts.hiddenlayersize)])
    # for k in range(1,self.Opts.hiddenlayersize):
    #     data = inputs[closest_point_idx == k] - self.centers[k]
    #     max_batch_idx = np.argmax(np.linalg.norm(data, axis=1))
    #     max_dist_relative_coord = data[max_batch_idx]
    #     means = np.linalg.norm(data, axis=1).mean()
    # # 6. use SIFT style histograms.

    def PNN(self):
        # The neighbors are the other centers, not test data (Schwenker p.444)
        try:
            nn = self.Opts.nearest_neighbors
        except:
            nn = 1  # assume 1NN in case nearest_neighbours is unspecified
        all_distances = np.linalg.norm(self.centers - self.centers[:, np.newaxis], axis=2)
        P_closest_distances = [np.partition(all_distances[:, k],(0, nn))[1:nn + 1].mean() for k in
                               range(self.Opts.hiddenlayersize)]
        width = np.array([P_closest_distances, P_closest_distances]).T
        return width

    def PCA(self, inputs):
        # Should be done on inputs only
        # taken from https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
        numdims = self.Opts.PCA_numdims
        # data = np.append(inputs,outputs,axis=1)
        data = inputs
        m,n = data.shape
        data -= data.mean(axis=0)
        # calculate the covariance matrix
        R = np.cov(data, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = np.linalg.eigh(R)
        # sort eigenvectors in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        # sort eigenvalues according to same index
        evals = evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :numdims]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return np.dot(evecs.T, data.T).T

        # # plotting the resulting "less-dimensionalized" data
        # import matplotlib.pyplot as plt
        # xd = np.dot(evecs.T, data.T)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.scatter(xd[0, :], xd[1, :])
        # plt.show()