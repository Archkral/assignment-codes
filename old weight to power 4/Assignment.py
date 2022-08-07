"""
This code calls upon the classes contained within GAP_RBF, NN_RBF, and NN_RBF_LM.

Made by: Karlo Rado - 4212169
"""
import numpy as np  # its the language used for
import json  # data loading
import time  # for light optimization
from Part3.NN_RBF import NNPart1_rand, NNPart1_alg
from Part3.NN_RBF_LM import NNPart2_alg
from Part3.GAP_RBF import NNPart3
import matplotlib.pyplot as plt

# To avoid getting SciView in pycharm
# import matplotlib
# from mpl_toolkits import mplot3d
# matplotlib.use('Qt5Agg')

# batch seed analysis, fix/remove parts with "KW:batch" to have the normal part3:
data_seeds = [0]
alg_seeds = [0]
seeds = np.array([data_seeds, alg_seeds]).T
e_rand_train_list = np.array([])
e_rand_valid_list = np.array([])
e_alg_train_list = np.array([])
e_alg_valid_list = np.array([])
e_GAP_train_list = np.array([])
e_GAP_valid_list = np.array([])
hiddenneurons = np.array([50, 100, 150])
for k in seeds:
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
        def __init__(self, inputs, outputs, batch_value):  # KW:batch
            # Variables (hyperparameters) regarding NN behaviour
            self.inputlayersize = np.size(inputs, axis=1)
            self.outputlayersize = np.size(outputs, axis=1)
            if self.outputlayersize > 1:
                raise ValueError('HARDCODED FOR 1 OUTPUT! Write Karlo to expand it.')  # underlying assumptions and maths
            self.hiddenlayersize = 50
            self.mu = 1  # learning rate
            self.mu_delta = 4  # the learning rate modifier
            # Set input range from -1 to 1 or weight range?
            self.range = np.array([[np.min(inputs[:, i]), np.max(inputs[:, i])] for i in range(self.inputlayersize)])

            # Variables regarding iteration/optimization process:
            self.epochs = 500  # number of (forward-backward) pass of ALL training samples
            self.goal = 1e-9  # SSE/MSE goal
            self.min_grad = 1e-10  # gradient goal (np.linalg.norm(derivatives))
            # The centers and weights are defined in the NN_RBF.py
            # predictable alg seed generation
            self.seed = batch_value  # KW:batch

            # K-means variables
            # variables for unsupervised learning steps
            self.kmeans_iterations = 100
            # IW_karlo variables:
            # rounding off at "self.round" decimals in IW_karlo
            self.round = 4

            # unused:
            # # PCA variables:
            # self.PCA_numdims = 2
            # # PNN variables:
            # # Assumes 1NN input weight algorithm if following is commented out:
            # self.nearest_neighbors = 3
            # if self.nearest_neighbors >= self.hiddenlayersize:
            #     raise ValueError('nearest neighbours (IW precalc) is done on other centers, not inputs.')


    options = Opts(inputs_train, outputs, k[1])  # KW:batch
    # Input end
    #########################################################################
    # Code start
    # Part 1a. training the output weights only, 1b. Using supervised training.
    # Part 2. training the full RBF NN with LM
    # Part 3. hyperparameter optimization (only hidden units this time) - GAP algorithm
    start = time.time()
    print('start')

    # PART 1
    part1_rand = NNPart1_rand(options, inputs_train)
    part1_alg = NNPart1_alg(options, inputs_train)

    pre = time.time()
    print(pre-start, ' seconds')
    print('done pre-processing \n')

    # Can use the following with batches:
    e_storage1 = []
    for i in range(part1_alg.Opts.epochs):
        e = part1_alg.AdaptiveTraining(inputs_train, outputs_train)
        e_storage1 = np.append(e_storage1, e)
        # print(e)
        if (part1_alg.Opts.goal >= e) or (
               part1_alg.Opts.min_grad >= np.linalg.norm(part1_alg.dLW)):
            break

    # fig0 = plt.figure()
    # ax0 = fig0.gca(projection='3d')
    # ax0.scatter(inputs_train[:, 0], inputs_train[:, 1], part1_alg.forward_rbf(inputs_train))
    # ax0.set_xlabel('Angle of attack [rad]')
    # ax0.set_ylabel('Side-slip angle [rad]')
    # ax0.set_zlabel('Moment coefficient [-]')
    # plt.title('pnn-based initialization, back-prop trained')
    # plt.show()

    # just OLS
    e_rand_train = part1_rand.OLSInspired(inputs_train, outputs_train)
    e_rand_valid = part1_rand.costFunction(inputs_validation, outputs_validation)
    print('random errors. Train:', e_rand_train, ' Valid:', e_rand_valid)

    # unsupervised learning algorithms + OLS (only need to iterate once as it is just a least squares)
    e_alg_train = part1_alg.OLSInspired(inputs_train, outputs_train)
    e_alg_valid = part1_alg.costFunction(inputs_validation, outputs_validation)
    print('alg errors. Train:', e_alg_train, ' Valid:', e_alg_valid)

    # # Plots
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(inputs_train[:, 0], inputs_train[:, 1], outputs_train)
    # ax.set_xlabel('Angle of attack [rad]')
    # ax.set_ylabel('Side-slip angle [rad]')
    # ax.set_zlabel('Moment coefficient [-]')
    # plt.title('original data')
    # plt.show()
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
    # plt.title('algorithm-based initialization')
    # plt.show()
    #
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111)
    # ax3.plot(part1_rand.centers[:,0],part1_rand.centers[:,1],'bo', part1_alg.centers[:,0], part1_alg.centers[:,1],'rx')
    # ax3.set_xlabel('Angle of attack [rad]')
    # ax3.set_ylabel('Side-slip angle [rad]')
    # plt.title('centers')
    # plt.show()

    time1 = time.time()
    print(time1-pre, ' seconds for 3.1')

    # # Levenberg-Marquardt
    # # Doesn't work
    # print('part 3.2')
    # part2 = NNPart2_alg(Opts(inputs_train, outputs_train, batch_value), inputs_train)
    # # part2.OLSInspired(inputs_train, outputs_train)
    # e_storage2 = []
    # try:
    #     for i in range(part2.Opts.epochs):
    #         e = part2.AdaptiveTraining(inputs_train, outputs_train)
    #         print('error =', e, ' learning rate =', part2.Opts.mu)
    #         if part2.Opts.goal >= e:  # or (part2.Opts.min_grad >= np.sqrt(
    #             #       np.sum(part2.dLW ** 2) + np.sum(part2.dIW ** 2) + np.sum(part2.dcenters ** 2))):
    #             break
    #         e_storage2 = np.append(e_storage2, e)
    # except:
    #     print('Levenberg-Marquardt failed after', i, 'iterations.\nProceeding with GAP (aka Part 3.3).')
    #
    # if e_storage2[0] == e_storage2[-1:]:  # if no change in error:
    #     print('LM failed')
    #
    # print(time.time()-time1, ' seconds passed for LM.')
    # time1 = time.time()

    # Growing and Pruning algorithm
    # "Greedy growing algorithm"
    opts = Opts(inputs_train, outputs_train, k[1])  # KW:batch
    opts.hiddenlayersize = 0
    part3 = NNPart3(opts, inputs_train)
    part3.yHat = np.zeros(outputs_train.shape)
    max_grow = 60  # above 400 the pruning step takes too long (hessian inverse calculation in particular)
    its = 10

    for i in range(max_grow):
        # update yHat
        part3.yHat = part3.forward_rbf(inputs_train)
        part3.grow_step(inputs_train, outputs_train)
        # if i % 3 == 0:  # Can use this to speed up growth a bit, no need to train for each new "neuron", honestly.
        # Because LM doesn't work
        e = part3.OLSInspired(inputs_train, outputs_train)
        print('e =', e, ' neurons =', part3.Opts.hiddenlayersize)
        if e < part3.Opts.goal:
            print("growing done")
            break
        # # LM based update input/output weights and centers
        # for j in range(its):
        #     e = part3.AdaptiveTraining(inputs_train, outputs_train)
        # e = part3.costFunction(inputs_train, outputs_train)
    hid = part3.Opts.hiddenlayersize

    # fig6 = plt.figure()
    # ax6 = fig6.gca(projection='3d')
    # ax6.scatter(inputs_train[:, 0], inputs_train[:, 1], part3.forward_rbf(inputs_train))
    # ax6.set_xlabel('Angle of attack [rad]')
    # ax6.set_ylabel('Side-slip angle [rad]')
    # ax6.set_zlabel('Moment coefficient [-]')
    # plt.title('GAP to ' + str(hid) + ', no pruning')
    # plt.show()
    # print('nice')

    # Because it updates output weights after each cut neuron, and hasn't really reached a local minimum its not quite OBS.
    # But whatever, the core code is identical. Here's Optimal Brain Surgeon [OBS]
    e_start = 1.2*part3.costFunction(inputs_train, outputs_train)  # once LM works set this to the goal error.
    e_curr = e_start
    max_pruning = 10
    for i in range(max_pruning):
        hid_old = part3.Opts.hiddenlayersize
        LW_old = part3.LW
        IW_old = part3.IW
        centers_old = part3.centers
        # idx, part3.LW, part3.IW, part3.centers = part3.OBS_step(inputs_train, outputs_train)
        idx, part3.LW, part3.IW = part3.OBS_step(inputs_train, outputs_train)
        e = part3.OLSInspired(inputs_train, outputs_train)
        if part3.costFunction(inputs_train, outputs_train) > min(e_curr, e_start):
            part3.Opts.hiddenlayersize = hid_old
            part3.LW = LW_old
            part3.IW = IW_old
            part3.centers = centers_old
            print('Pruning error jumped past tolerated increase in', i, 'iterations')
            break
        e_curr = 1.05*part3.costFunction(inputs_train, outputs_train)
        print('e =', e_curr, ' idx =', idx, ' hidden neurons =', part3.Opts.hiddenlayersize)

    # fig6 = plt.figure()
    # ax6 = fig6.gca(projection='3d')
    # ax6.scatter(inputs_train[:, 0], inputs_train[:, 1], part3.forward_rbf(inputs_train))
    # ax6.set_xlabel('Angle of attack [rad]')
    # ax6.set_ylabel('Side-slip angle [rad]')
    # ax6.set_zlabel('Moment coefficient [-]')
    # plt.title('GAP ' + str(hid) + ' to ' + str(part3.Opts.hiddenlayersize) + ' hidden neurons')
    # plt.show()
    # print('nice')

    # KW:batch
    e_rand_train_list = np.append(e_rand_train_list, e_rand_train)
    e_rand_valid_list = np.append(e_rand_valid_list, e_rand_valid)
    e_alg_train_list = np.append(e_alg_train_list, e_alg_train)
    e_alg_valid_list = np.append(e_alg_valid_list, e_alg_valid)
    e_GAP_train_list = np.append(e_GAP_train_list, part3.costFunction(inputs_train, outputs_train))
    e_GAP_valid_list = np.append(e_GAP_valid_list, part3.costFunction(inputs_validation, outputs_validation))
