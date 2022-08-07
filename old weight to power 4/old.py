import numpy as np
import json
from Part3.NN_RBF import NNPart2, NNPart1

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
iterations = 2
training_size = int(np.ceil(0.8 * input_length / iterations - 0.5) * iterations)  # ~80% training data
batch_size = int(training_size / iterations)

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
        # Variables (hyperparameters) regarding NN behaviour
        self.a = 0.1  # not a scaling parameter... A singular output weight that modifies ALL data? Its not clear.
        # Using the following as placeholder numbers -> change them based on input/output/result of optimization:
        self.inputlayersize = np.size(inputs, axis=1)
        self.outputlayersize = np.size(outputs, axis=1)
        if self.outputlayersize > 1:
            raise ValueError('HARDCODED TO ONLY WORK FOR 1 OUTPUT! Talk to Karlo to expand it.')
        self.hiddenlayersize = 8
        self.mu = 1  # learning rate
        self.mu_delta = 10  # the learning rate modifier
        # Set input range from -1 to 1 or weight range?
        self.range = np.array([[np.min(inputs[:, i]), np.max(inputs[:, i])] for i in range(self.inputlayersize)])

        # Variables regarding iteration/optimization process:
        self.epochs = 20  # number of (forward-backward) pass of ALL training samples
        self.goal = 1e-9  # SSE/MSE goal
        self.min_grad = 1e-10  # gradient goal (np.linalg.norm(derivatives))
        # The centers and weights are defined in the NN_RBF.py
        # predictable seed generation
        self.seed = 0


options = Opts(inputs_train, outputs_train)
# Input end
#########################################################################
# Code start
# Part 1. training the output weights only
# Part 2. training the full RBF NN with LM
# Part 3. hyperparameter optimization (SVM/k-nn/RF/3-layer/DT || bayesian/evo) -- selected EA/GA

# PART 1
part1 = NNPart1(options, inputs_train)

# Can use the following with batches:


e_storage1 = []
for i in range(part1.Opts.epochs):
    for j in range(iterations):
        train_in = inputs_train[j * batch_size:(j + 1) * batch_size]
        train_out = outputs_train[j * batch_size:(j + 1) * batch_size]
        e = part1.AdaptiveTraining(train_in, train_out)
        e_storage1 = np.append(e_storage1, e)
        # print(e)
        if (part1.Opts.goal >= e) or (
                part1.Opts.min_grad >= np.linalg.norm(part1.dLW)):
            break

# PART 2
part2 = NNPart2(options, inputs)
print('part 2')
e_storage2 = []
for i in range(part2.Opts.epochs):
    for j in range(iterations):
        train_in = inputs_train[j * batch_size:(j + 1) * batch_size]
        train_out = outputs_train[j * batch_size:(j + 1) * batch_size]
        e = part2.AdaptiveTraining(train_in, train_out)
        # print(e)
        if (part2.Opts.goal >= e) or (part2.Opts.min_grad >= np.sqrt(
                np.sum(part2.dLW ** 2) + np.sum(part2.dIW ** 2) + np.sum(part2.dcenters ** 2))):
            break
        e_storage2 = np.append(e_storage2, e)

print('part 2a')
part2a = NNPart2(options, inputs)
e_storage2a = []
for i in range(part2a.Opts.epochs):
    for j in range(iterations):
        train_in = inputs_train[j * batch_size:(j + 1) * batch_size]
        train_out = outputs_train[j * batch_size:(j + 1) * batch_size]
        e = part2a.costFunction(train_in, train_out)
        part2a.CostFuncPrime(inputs_train, outputs_train)
        part2a.LW = part2a.LW + part2a.dLW
        part2a.IW = part2a.IW + part2a.dIW
        part2a.centers = part2a.centers + part2a.dcenters
        # print(e)
        if (part2a.Opts.goal >= e) or (part2a.Opts.min_grad >= np.sqrt(
                np.sum(part2a.dLW ** 2) + np.sum(part2a.dIW ** 2) + np.sum(part2a.dcenters ** 2))):
            break
        e_storage2a = np.append(e_storage2a, e)
