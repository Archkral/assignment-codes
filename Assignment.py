import numpy as np
import json
import matplotlib.pyplot as plt
# TBD: use CUDA tensors from pytorch instead of numpy arrays for GPU speedup. (factor ~100)
from Part4.NN_FF import NNPart1_alg, NNPart1_rand
from Part4.NN_FF_LM import NNPart2_alg, NNPart2_rand
from Part4.EA_FF import NNPart3

# data seed
np.random.seed(0)

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

# Setting up training data "batches" - learning rate should be adapted based on size?
training_size = int(np.ceil(0.8 * input_length - 0.5))  # ~80% training data

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
class Opts:
    def __init__(self, inputs, outputs):
        # Change the following based on input/output/result of optimization:
        self.inputlayersize = np.size(inputs, axis=1)
        self.outputlayersize = np.size(outputs, axis=1)
        if self.outputlayersize > 1:
            raise ValueError('HARDCODED TO ONLY WORK FOR 1 OUTPUT! Have to modify all sub-functions')
        self.hiddenlayersize = 50
        self.mu = 0.000001  # learning rate
        self.mu_delta = 4  # the learning rate modifier
        # Set input range from -1 to 1 or weight range?
        self.range = np.array([[np.min(inputs[:, i]), np.max(inputs[:, i])] for i in range(self.inputlayersize)])

        # Variables regarding iteration/optimization process:
        self.epochs = 2000  # number of (forward-backward) pass of ALL training samples
        self.goal = 0.7  # SSE/MSE goal
        self.min_grad = 1e-10  # gradient goal (np.linalg.norm(derivatives))
        # The centers and weights are defined in the NN_FF.py
        # predictable seed generation
        self.seed = 1  # alg seed (else use np.random.randint(9.2e18))

        # Initialization variables - "swath" style coverage maximization by looking at discrete solution space
        self.covIW_biases = int(2.5*np.sqrt(self.hiddenlayersize)+1)  # #biases to check for in covIW -- good enough from sqrt(#neurons) to 3xsqrt(#neurons)
        self.covIW_angles = int(1.25*np.sqrt(self.hiddenlayersize)+1)  # #angles to check for in covIW -- from 0.5*sqrt(#neurons) to 2xsqrt(#neurons)
        self.covIW_weightmod = 0.7  # data weight-modifier upon being covered by a "swath". Range  0 < x < 1

        # GA / Genetic algorithm
        self.DNA_size = 7  # max value of data to optimize (2**self.DNA_size)
        self.pop_size = 7  # req: value > 3; desc: number of evolutionary algorithms initialized per update
        self.n_gens = 25  # number of generations; read: "epochs"
        self.mutation_rate = 0.03  # likelihood an entry in the gene changes its "binary"
        self.crossover = 0.7  # crossover rate - parent1 x parent2 -> child mix.


options = Opts(inputs_train, outputs_train)
# Input end
#########################################################################
# Code start
# Part 1. training the output weights only
# Part 2. training the full RBF NN with LM
# Part 3. hyperparameter optimization (SVM/k-nn/RF/3-layer/DT || bayesian/evo) -- selected EA/GA

# PART 1
part1 = NNPart2_alg(options, inputs_train, outputs_train)
part1r = NNPart2_rand(options,inputs_train,outputs_train)
e_storage1 = []
e_storage1_r = []
# for i in range(part1.Opts.epochs):
#     e = part1.AdaptiveBackprop(inputs_train, outputs_train)
#     e_storage1 = np.append(e_storage1, e)
#     print(e)
#     e = part1r.AdaptiveBackprop(inputs_train, outputs_train)
#     e_storage1_r = np.append(e_storage1_r, e)
#     print(e)
breakpoint()

for i in range(part1.Opts.epochs):
    e = part1.AdaptiveBackprop(inputs_train, outputs_train)
    e_storage1 = np.append(e_storage1, e)
    print(e)
    # e = part1r.AdaptiveTraining(inputs_train, outputs_train)
    # e_storage1_r = np.append(e_storage1_r, e)
    # print(e)

# Plots
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(inputs_train[:, 0], inputs_train[:, 1], outputs_train)
ax.set_xlabel('Angle of attack [rad]')
ax.set_ylabel('Side-slip angle [rad]')
ax.set_zlabel('Moment coefficient [-]')
plt.title('original data')
plt.show()

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.scatter(inputs_train[:, 0], inputs_train[:, 1], part1.forward_nn(inputs_train))
ax1.set_xlabel('Angle of attack [rad]')
ax1.set_ylabel('Side-slip angle [rad]')
ax1.set_zlabel('Moment coefficient [-]')
plt.title('trained state')
plt.show()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.scatter(inputs_train[:, 0], inputs_train[:, 1], part1_alg.forward_rbf(inputs_train))
ax2.set_xlabel('Angle of attack [rad]')
ax2.set_ylabel('Side-slip angle [rad]')
ax2.set_zlabel('Moment coefficient [-]')
plt.title('algorithm-based initialization')
plt.show()



# # PART 2 - LM, doesn't work just like with the rbf
# part2 = NNPart2_alg(Opts(inputs_train, outputs_train), inputs_train, outputs_train)
# print('part 2')
# e_storage2 = []
# for i in range(part2.Opts.epochs):
#     e = part2.LM_AdaptiveTraining(inputs_train, outputs_train)
#     print(e,i)
#     # if (part2.Opts.goal >= e) or (part2.Opts.min_grad >= np.sqrt(np.sum(part2.dLW ** 2) + np.sum(part2.dbias2 ** 2)
#     #                               + np.sum(part2.dIW ** 2) + np.sum(part2.dbias1 ** 2))):
#     #     break
#     e_storage2 = np.append(e_storage2, e)
# breakpoint()
#
# """
# # PART 3, hyperparameter optimization
# # proper research was done for GAP RBF and all kinds of local/global search methods. People still are aroused by EAs.
# # To prove the point that genetic algorithms are super easy to code, run the following:
# # Otherwise the RBF GAP can easily be adapted to use the FF covIW function.
# """
# print("part3")
# store = {}
# part3 = NNPart3(Opts(inputs_train, outputs_train), inputs_train, outputs_train)
#
# # Step 1. Set initial population with DNA (read: binary lists)
# pop = np.random.randint(2, size=(part3.Opts.pop_size, part3.Opts.DNA_size))  # initialize the pop DNA
# # Step 2. Loop until desired outcome is reached
# for _ in range(part3.Opts.n_gens):
#     # Step 3. Generate (perhaps mutated) hyperparameters from the DNA:
#     hiddenlayersizes = part3.translateDNA(pop)
#
#     # Step 4. Initialize FF_NNs with the ascertained hyperparameters...
#     opts = [Opts(inputs_train, outputs_train) for i_ in range(part3.Opts.pop_size)]
#     for i_ in range(part3.Opts.pop_size):
#         opts[i_].hiddenlayersize = hiddenlayersizes[i_]
#     nnpart3 = [NNPart3(opts[i_], inputs_train, outputs_train) for i_ in range(part3.Opts.pop_size)]
#     print("FF_NN initialization number: ", _+1)
#     # # Step 5. train them for a bit. Can be done using multiprocessing -- aka parallel processing.
#     # # Also possible using 3-dimensional arrays, both ways save time.
#     # # method WITH storing errors: (too data-bulky)
#     # errors = [nnpart3[i].AdaptiveTraining(inputs_train, outputs_train)
#     #           for i in range(part3.Opts.pop_size) for i_ in range(part3.Opts.epochs)]  # final errors in [:-pop_size]
#     # method without storing errors
#     fitness = np.empty(0)
#     for i in range(part3.Opts.pop_size):
#         print(i)
#         for i_ in range(part3.Opts.epochs):
#             error = nnpart3[i].AdaptiveTraining(inputs_train, outputs_train)  # AdaptiveTraining (LM) doesn't work
#         print(error)
#         fitfunc = 1/(np.array([error]) + nnpart3[i].Opts.hiddenlayersize*(.15 + 10 * (error > nnpart3[i].Opts.goal)))
#         fitness = np.append(fitness, fitfunc)
#
#     # Step 6. pick best performing one:
#     print("Most fitted DNA: ", pop[np.argmax(fitness), :])
#
#     # Step 7. Mutate/crossover the hyperparameters
#     pop = part3.select(pop, fitness)
#     pop_copy = pop.copy()
#     for parent in pop:
#         child = part3.crossover(parent, pop_copy)
#         child = part3.mutate(child)
#         parent[:] = child  # parent is replaced by its child
#
#     store[_, 1] = pop
#     store[_, 2] = fitness
#
# print(store)
#
# fig6 = plt.figure()
# ax6 = fig6.gca(xlabel='generation number [-]', ylabel='number of hidden neurons')
# for i in range(part3.Opts.n_gens):
#     for i2 in range(part3.Opts.pop_size):
#         ax6.plot(i,part3.translateDNA(store[i, 1][i2]), 'b.')
#     ax6.plot(i,part3.translateDNA(store[i, 1][np.argmax(store[i, 2])]),'r.')
# plt.show()

# fig7 = plt.figure()
# ax7 = fig7.gca()
# x0 = []
# x1 = []
# x2 = []
# x3 = []
# x4 = []
# x5 = []
# x6 = []
# for i in range(part3.Opts.n_gens):
#     x0.append(part3.translateDNA(store[i, 1][0]))
#     x1.append(part3.translateDNA(store[i, 1][1]))
#     x2.append(part3.translateDNA(store[i, 1][2]))
#     x3.append(part3.translateDNA(store[i, 1][3]))
#     x4.append(part3.translateDNA(store[i, 1][4]))
#     x5.append(part3.translateDNA(store[i, 1][5]))
#     x6.append(part3.translateDNA(store[i, 1][6]))
# ax7.plot(range(part3.Opts.n_gens), x0,
#          range(part3.Opts.n_gens), x1,
#          range(part3.Opts.n_gens), x2,
#          range(part3.Opts.n_gens), x3,
#          range(part3.Opts.n_gens), x4,
#          range(part3.Opts.n_gens), x5,
#          range(part3.Opts.n_gens), x6)
# plt.show()
# x=[x0,x1,x2,x3,x4,x5,x6]
