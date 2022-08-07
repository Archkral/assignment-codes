from Part4.NN_FF_LM import NNPart2_rand
import numpy as np


class NNPart3(NNPart2_rand):
    # Here we're initializing NNPart3 to use the quicker Xavier initialization.
    """
    Clean enough and easy implementation taken from Morvan Zhou, and in his words:
    Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
    """
    # # Genetic algorithm variables:
    # self.DNA_size = 8  # max value of data to optimize (2**self.DNA_size)
    # self.pop_size = 10  # req: value > 3; desc: number of evolutionary algorithms initialized per update
    # self.n_gens = 100  # number of generations; read: "epochs"
    # self.mutation_rate = 0.003  # likelihood an entry in the gene changes its "binary"
    # self.crossover = 0.8  # crossover rate - parent1 x parent2 -> child mix.

    # F = self.costFunction(inputs, y)

    # convert binary DNA to decimal valued hiddenlayersizes
    def translateDNA(self, pop):
        return pop.dot(2 ** np.arange(self.Opts.DNA_size)[::-1])

    def select(self, pop, fitness):  # nature selection wrt pop's fitness
        idx = np.random.choice(np.arange(self.Opts.pop_size), size=self.Opts.pop_size, replace=True,
                               p=fitness / fitness.sum())
        return pop[idx]

    def crossover(self, parent, pop_copy):  # mating process (genes crossover)
        if np.random.rand() < self.Opts.crossover:
            i_ = np.random.randint(0, self.Opts.pop_size, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, size=self.Opts.DNA_size).astype(np.bool)  # choose crossover points
            parent[cross_points] = pop_copy[i_, cross_points]  # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.Opts.DNA_size):
            if np.random.rand() < self.Opts.mutation_rate:
                child[point] = 1 if child[point] == 0 else 0
        return child
