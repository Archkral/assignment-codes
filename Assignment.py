"""
This code calls upon the classes contained within GAP_RBF, NN_RBF, and NN_RBF_LM.

Made by: Karlo Rado - 4212169
"""
import numpy as np  # its the language used for
import json  # data loading
import time  # for light optimization
from Part3.NN_RBF import NNPart1_rand, NNPart1_alg
from Part3.NN_RBF_LM import NNPart2_alg, NNPart2_rand
from Part3.GAP_RBF import NNPart3
import matplotlib.pyplot as plt

# To avoid getting SciView in pycharm
# import matplotlib
# from mpl_toolkits import mplot3d
# matplotlib.use('Qt5Agg')

# batch seed analysis, fix/remove parts with "KW:batch" to have the normal part3:
data_seeds = [0, 3, 27, 1341, 1342, 1641, 12, 45, 2, 13]
alg_seeds = [0, 3, 12, 125, 2356, 547, 72, 35, 26, 153]
num_hid = np.array([50])*np.ones(len(alg_seeds)).astype('int')
mus = np.array([0.1, 0.001])
mudeltas = np.array([8, 2])
ds = np.array([0.1, 0.001])
epos = np.array([500, 2000])

seeds = np.array([data_seeds, alg_seeds]).T
e_rand_train_list = np.array([])
e_rand_valid_list = np.array([])
e_alg_train_list = np.array([])
e_alg_valid_list = np.array([])
e_GAP_train_list = np.array([])
e_GAP_valid_list = np.array([])
e_jac_train_list = np.array([])
e_jac_valid_list = np.array([])

for its in range(1):
    for numk, k in enumerate(seeds):
        # data seed
        np.random.seed(k[0])  # KW:batch

        dictvalues = json.load(open("../Part2/reconstructed_data.json", "r"))

        x = np.array(dictvalues["x_storage"])
        z = np.array(dictvalues["z_storage"])
        Cm = np.array(dictvalues["Cm"])

        # INPUT
        #########################################################################
        # Defining training and validation data
        input_length = np.size(x, axis=0)
        inputs = np.append(np.array([z[:, 0]]).T, np.array([z[:, 1]]).T, axis=1)
        inputs = inputs  # / np.amax(inputs, axis=0)
        outputs = Cm  # / np.amax(Cm, axis=0)

        # Setting up training data "batches" - if you want to.
        ratio = 0.8  # [-] fraction of data used for training.
        training_size = int(np.ceil(ratio * input_length - 0.5))  # ~80% training data

        index_train = np.random.choice(input_length, training_size, replace=False)
        b = [x for x in range(0, input_length)]
        index_validation = list(set(index_train) ^ set(b))

        inputs_train = inputs[index_train, :]
        outputs_train = outputs[index_train, :]
        inputs_validation = inputs[index_validation, :]
        outputs_validation = outputs[index_validation, :]


        #########################################################################
        # Glorified dictionary (inputs)
        # is defined in the assignment file
        class Opts():
            def __init__(self, inputs, outputs, batch_value, num_hid=50, mu=0.01, d=0.01, mudelta =1.5, epochs=200):  # KW:batch
                # Variables (hyperparameters) regarding NN behaviour
                self.inputlayersize = np.size(inputs, axis=1)
                self.outputlayersize = np.size(outputs, axis=1)
                if self.outputlayersize > 1:
                    raise ValueError(
                        'HARDCODED FOR 1 OUTPUT! Write Karlo to expand it.')  # underlying assumptions and maths
                self.hiddenlayersize = num_hid
                self.mu = mu  # learning rate
                self.mu_delta = mudelta  # the learning rate modifier
                # Set input range from -1 to 1 or weight range?
                self.range = np.array([[np.min(inputs[:, i]), np.max(inputs[:, i])] for i in range(self.inputlayersize)])

                # Variables regarding iteration/optimization process:
                self.epochs = epochs # number of (forward-backward) pass of ALL training samples
                self.goal = 1e-9  # SSE/MSE goal
                self.min_grad = 1e-10  # gradient goal (np.linalg.norm(derivatives))
                # The centers and weights are defined in the NN_RBF.py
                # predictable alg seed generation
                self.seed = batch_value  # KW:batch

                # K-means variables
                # variables for unsupervised learning steps
                self.kmeans_iterations = 100
                # IW_karlo variables:
                # minimum squared distance between centers for IW_karlo
                self.dist = d  # has to be at least 0.005

                ### unused, edit NN_RBF.m:
                # PCA variables:
                self.PCA_numdims = 2
                # PNN variables:
                # Assumes 1NN input weight algorithm if following is commented out:
                self.nearest_neighbors = 3
                if self.nearest_neighbors >= self.hiddenlayersize:
                    raise ValueError('nearest neighbours (IW precalc) is done on other centers, not inputs.')

        if its ==0:
            options = Opts(inputs_train, outputs, k[1], num_hid=num_hid[numk])  # KW:batch
        elif its ==1:
            options = Opts(inputs_train, outputs, k[1], mu=mus[numk])
        elif its == 2:
            options = Opts(inputs_train, outputs, k[1], mudelta=mudeltas[numk])
        elif its ==3:
            options = Opts(inputs_train, outputs, k[1], d=ds[numk])
        elif its ==4:
            options = Opts(inputs_train, outputs, k[1], epochs=epos[numk])
        # Input end
        #########################################################################
        # Code start
        # Part 1a. training the output weights only, 1b. Using supervised training.
        # Part 2. training the full RBF NN with LM
        # Part 3. hyperparameter optimization (only hidden units this time) - GAP algorithm
        # start = time.time()
        # print('start')
        #
        # # PART 1
        # part1_rand = NNPart1_rand(options, inputs_train)
        # part1_alg = NNPart1_alg(options, inputs_train)
        #
        # pre = time.time()
        # print(pre - start, ' seconds')
        # print('done pre-processing \n')

        # # Back-prop output weights.
        # e_storage1 = []
        # for i in range(part1_alg.Opts.epochs):
        #     e = part1_alg.AdaptiveTraining(inputs_train, outputs_train)
        #     e_storage1 = np.append(e_storage1, e)
        #     # print(e)
        #     if (part1_alg.Opts.goal >= e) or (
        #            part1_alg.Opts.min_grad >= np.linalg.norm(part1_alg.dLW)):
        #         break

        # fig0 = plt.figure()
        # ax0 = fig0.gca(projection='3d')
        # ax0.scatter(inputs_train[:, 0], inputs_train[:, 1], part1_alg.forward_rbf(inputs_train))
        # ax0.set_xlabel('Angle of attack [rad]')
        # ax0.set_ylabel('Side-slip angle [rad]')
        # ax0.set_zlabel('Moment coefficient [-]')
        # plt.title('rbf prediction')
        # plt.show()
        #
        # # just OLS
        # e_rand_train = part1_rand.OLSInspired(inputs_train, outputs_train)
        # e_rand_valid = part1_rand.costFunction(inputs_validation, outputs_validation)
        # print('random errors. Train:', e_rand_train, ' Valid:', e_rand_valid)
        #
        # # unsupervised learning algorithms + OLS (only need to iterate once as it is just a least squares)
        # e_alg_train = part1_alg.OLSInspired(inputs_train, outputs_train)
        # e_alg_valid = part1_alg.costFunction(inputs_validation, outputs_validation)
        # print('alg errors. Train:', e_alg_train, ' Valid:', e_alg_valid)
        #
        # # # Plots
        # # fig = plt.figure()
        # # ax = fig.gca(projection='3d')
        # # ax.scatter(inputs_train[:, 0], inputs_train[:, 1], outputs_train)
        # # ax.set_xlabel('Angle of attack [rad]')
        # # ax.set_ylabel('Side-slip angle [rad]')
        # # ax.set_zlabel('Moment coefficient [-]')
        # # plt.title('original data')
        # # plt.show()
        #
        # fig1 = plt.figure()
        # ax1 = fig1.gca(projection='3d')
        # ax1.scatter(inputs_train[:, 0], inputs_train[:, 1], part1_rand.forward_rbf(inputs_train))
        # ax1.set_xlabel('Angle of attack [rad]')
        # ax1.set_ylabel('Side-slip angle [rad]')
        # ax1.set_zlabel('Moment coefficient [-]')
        # plt.title('random initialization')
        # plt.show()
        #
        # fig2 = plt.figure()
        # ax2 = fig2.gca(projection='3d')
        # ax2.scatter(inputs_train[:, 0], inputs_train[:, 1], part1_alg.forward_rbf(inputs_train))
        # ax2.set_xlabel('Angle of attack [rad]')
        # ax2.set_ylabel('Side-slip angle [rad]')
        # ax2.set_zlabel('Moment coefficient [-]')
        # plt.title('IW_karlo-based initialization')
        # plt.show()
        #
        # # fig3 = plt.figure()
        # # ax3 = fig3.add_subplot(111)
        # # ax3.plot(part1_rand.centers[:,0],part1_rand.centers[:,1],'bo', part1_alg.centers[:,0], part1_alg.centers[:,1],'rx')
        # # ax3.set_xlabel('Angle of attack [rad]')
        # # ax3.set_ylabel('Side-slip angle [rad]')
        # # plt.title('centers')
        # # plt.show()
        #
        # time1 = time.time()
        # print('Time =', time1 - pre, 'seconds for part 1')

        # # Backprop & Levenberg-Marquardt
        # print('part 3.2')
        # part2 = NNPart2_alg(options, inputs_train)
        # # you can also use NNPart2_rand for random initialisation
        # e_storage2 = []
        # for i in range(part2.Opts.epochs):
        #     e = part2.AdaptiveBackprop(inputs_train, outputs_train)  # Doesn't work when using AdaptiveTraining(x,y).
        #     print(e)
        #     # e_storage2.append(e)

        # e_alg_valid_list = np.append(e_alg_valid_list, part2.costFunction(inputs, outputs))

        # fig6 = plt.figure()
        # ax6 = fig6.gca(projection='3d')
        # ax6.scatter(inputs_train[:, 0], inputs_train[:, 1], part2.forward_rbf(inputs_train))
        # ax6.set_xlabel('Angle of attack [rad]')
        # ax6.set_ylabel('Side-slip angle [rad]')
        # ax6.set_zlabel('Moment coefficient [-]')
        # plt.title('Jacobian backprop at ' + str(part2.Opts.hiddenlayersize) + ' hidden neurons trained for ' + str(
        #     part2.Opts.epochs) + ' epochs')
        # plt.show()
        #
        # fig7 = plt.figure()
        # ax7 = fig7.add_subplot(111)
        # ax7.plot(np.linspace(1, i+1, i+1), e_storage2)
        # ax7.set_xlabel('Number of epochs [-]')
        # ax7.set_ylabel('Half of sum of squared errors [-]')
        # plt.title('Jacobian backprop at ' + str(part2.Opts.hiddenlayersize) + ' hidden neurons trained for ' + str(
        #     part2.Opts.epochs) + ' epochs')
        # plt.show()
        #
        # fig8 = plt.figure()
        # t1 = np.linspace(0, len(Cm) * 0.01, len(Cm))
        # plt.plot(t1, outputs, 'r', label='measured data')
        # plt.plot(t1, part2.forward_rbf(inputs), 'b', label='Jacobian back-prop')
        # plt.legend()
        # plt.xlabel('time [s]')
        # plt.ylabel('moment coefficient [-]')
        # plt.show()
        #
        # # print(time.time() - time1, ' seconds passed for LM.')
        # # time1 = time.time()


        """
        # Growing and Pruning algorithm
        # "Greedy growing algorithm"
        """

        opts = options  # KW:batch
        margin = 10  # growth margin
        max_grow = margin + opts.hiddenlayersize
        opts.hiddenlayersize = 0
        part3 = NNPart3(opts, inputs_train)
        part3.yHat = np.zeros(outputs_train.shape)
        # when using 1000 epochs for training, 20-50 epochs should be enough to tune the newly grown neuron to the network
        # IF LM WORKED. Now just stick to 1/20 of total epochs
        part3.Opts.epochs = 1 + np.floor(part3.Opts.epochs / 20).astype('int')
        # above 400 the pruning step takes too long (essentially the Hessian inverse)
        e=10
        # while e > 0.01:
        for i in range(max_grow):
            # update yHat
            part3.yHat = part3.forward_rbf(inputs_train)
            part3.grow_step(inputs_train, outputs_train)

            # # LM based update input/output weights and centers
            for j in range(part3.Opts.epochs):
                e = part3.AdaptiveBackprop(inputs_train, outputs_train)
            # # Or use:
            # e = part3.OLSInspired(inputs_train, outputs_train)
            e = part3.costFunction(inputs_train, outputs_train)
            print('e =', e, ' neurons =', part3.Opts.hiddenlayersize)
            if e < part3.Opts.goal:
                print("growing done")
                break

        hid = part3.Opts.hiddenlayersize  # used to remember the old hidden layer size

        # fig7 = plt.figure()
        # ax7 = fig7.gca(projection='3d')
        # ax7.scatter(inputs_train[:, 0], inputs_train[:, 1], part3.forward_rbf(inputs_train))
        # ax7.set_xlabel('Angle of attack [rad]')
        # ax7.set_ylabel('Side-slip angle [rad]')
        # ax7.set_zlabel('Moment coefficient [-]')
        # plt.title('GAP to ' + str(hid) + ', no pruning')
        # plt.show()
        # print('nice')

        # Because it updates output weights after each cut neuron, and hasn't really reached a local minimum its not quite OBS.
        # But whatever, the core code is identical. Here's Optimal Brain Surgeon [OBS]
        e_start = 1.2*part3.costFunction(inputs_train, outputs_train)  # once LM works set this to the goal error.
        e_curr = e_start
        max_pruning = margin  # removes up to the margin.
        for i in range(max_pruning):
            hid_old = part3.Opts.hiddenlayersize
            LW_old = part3.LW
            IW_old = part3.IW
            centers_old = part3.centers
            # idx, part3.LW, part3.IW, part3.centers = part3.OBS_step(inputs_train, outputs_train)
            idx, part3.LW, part3.IW = part3.OBS_step(inputs_train, outputs_train)
            # e = part3.OLSInspired(inputs_train, outputs_train)
            for j in range(part3.Opts.epochs):
                e = part3.AdaptiveBackprop(inputs_train, outputs_train)
            if part3.costFunction(inputs_train, outputs_train) > min(e_curr, e_start):
                part3.Opts.hiddenlayersize = hid_old
                part3.LW = LW_old
                part3.IW = IW_old
                part3.centers = centers_old
                print('Pruning error jumped past tolerated increase in', i, 'iterations')
                break
            e_curr = 1.05*part3.costFunction(inputs_train, outputs_train)
            print('e =', e_curr, ' idx =', idx, ' hidden neurons =', part3.Opts.hiddenlayersize)

        fig8 = plt.figure()
        ax8 = fig8.gca(projection='3d')
        ax8.scatter(inputs_train[:, 0], inputs_train[:, 1], part3.forward_rbf(inputs_train))
        ax8.set_xlabel('Angle of attack [rad]')
        ax8.set_ylabel('Side-slip angle [rad]')
        ax8.set_zlabel('Moment coefficient [-]')
        plt.title('GAP ' + str(hid) + ' to ' + str(part3.Opts.hiddenlayersize) + ' hidden neurons')
        plt.show()
        print('nice')

        # # KW:batch
        # e_rand_train_list = np.append(e_rand_train_list, e_rand_train)
        # e_rand_valid_list = np.append(e_rand_valid_list, e_rand_valid)
        # e_alg_train_list = np.append(e_alg_train_list, e_alg_train)
        # e_alg_valid_list = np.append(e_alg_valid_list, e_alg_valid)
        e_GAP_train_list = np.append(e_GAP_train_list, part3.costFunction(inputs_train, outputs_train))
        e_GAP_valid_list = np.append(e_GAP_valid_list, part3.costFunction(inputs_validation, outputs_validation))
        # e_jac_train_list = np.append(e_jac_train_list, part2.costFunction(inputs_train, outputs_train))
        # e_jac_valid_list = np.append(e_jac_valid_list, part2.costFunction(inputs_validation, outputs_validation))


# plt.plot(t1, Cm, 'r', label='measured data')
# plt.plot(t1, part3.forward_rbf(inputs), 'b', label='GAP 60 to 50')
# plt.legend()
# plt.xlabel('time [s]')
# plt.ylabel('moment coefficient [-]')
# plt.title('2D representation of results using GAP')
# plt.show()