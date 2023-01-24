from numpy import genfromtxt
from numpy.random import randint
from numpy.random import rand
import sys
import numpy as np
from multiprocessing import Pool
import copy


# debug options
np.set_printoptions(threshold=np.inf)


class Account:
    # history = []

    def __init__(self, initial_amount) -> None:
        self.a_amount = initial_amount
        self.b_amount = 0

    def a_to_b(self, pair_rate):
        self.b_amount = self.a_amount / pair_rate
        self.a_amount = 0
        # self.history.append(self)

    def a_as_b(self, pair_rate):
        return self.a_amount / pair_rate

    def b_to_a(self, pair_rate):
        self.a_amount = self.b_amount * pair_rate
        self.b_amount = 0
        # self.history.append(self)

    def b_as_a(self, pair_rate):
        return self.b_amount * pair_rate

    def __repr__(self) -> str:
        return f'A: {self.a_amount}\nB: {self.b_amount}'


# Returns a array with buy/sell flags
def sma_strategy(data: np.ndarray, params):
    # get close prices from fourth column of data
    closes = data[:, 4]
    sma1 = sma(closes, params.get("sma1"))
    sma2 = sma(closes, params.get("sma2"))
    longs = cross_over(sma1, sma2)
    exits = cross_over(sma2, sma1) * -1

    return merge_signals(longs, exits)


def merge_signals(l, s):
    for i in range(len(l)):
        if s[i] == -1:
            l[i] = -1

    return l

# detect two arrays corssings


def cross_over(a, b):
    crossovers = [np.NaN]
    for i in range(1, len(a)):
        if a[i-1] < b[i-1] and a[i] > b[i]:
            crossovers = np.append(crossovers, [1])
        else:
            crossovers = np.append(crossovers, [0])

    return crossovers

# smooth moving average


def sma(data, window):
    sma = np.array([np.NaN]*window)

    for i in range(window, len(data)):
        sma = np.append(sma, np.average(data[i-window:i]))

    return sma


def simulate(account: Account, data: np.ndarray, strategy, params):
    signals = strategy(data, params)
    prices = data[:, 4]

    for price, signal in zip(prices, signals):
        if signal == 1 and account.a_amount != 0:
            account.a_to_b(price)
        if signal == -1 and account.b_amount != 0:
            account.b_to_a(price)

    return account.a_amount + account.b_as_a(prices[-1])


# tournament selection


def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children


def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    keys = p1.keys()

    if rand() < r_cross:
        for k in keys:
            c1[k] = p2[k]
            c1[k] = p1[k]
    return [c1, c2]

# mutation operator


def mutation(params, r_mut, bounds):
    for k in params.keys():
        diff = randint(-r_mut, r_mut)
        if params[k] + diff in range(bounds.get(k)[0], bounds.get(k)[1]):
            params[k] = params[k] + diff


# genetic algorithm


def genetic_algorithm(sim, strat, acc, data, bounds,  n_iter, n_pop, r_cross, r_mut, processes):
    # initial population of random params
    pop = []
    for _ in range(n_pop):
        params = {}
        for key in bounds:
            params[key] = randint(bounds[key][0], bounds[key][1])

        pop.append(params)

    account = copy.copy(acc)
    # keep track of best solution
    best, best_eval = sys.maxsize, sim(account, data, strat, pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # print("Gen: ", gen)
        with Pool(processes) as p:

            scores = [sim(copy.copy(acc), data, strat, params)
                      for params in pop]

        # print(scores)
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %f" %
                      (gen,  best, scores[i]))
                # select parents
            selected = [selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()

            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i+1]
                # crossover and mutation
                for c in crossover(p1, p2, r_cross):
                    mutation(c, r_mut, bounds)
                #     # store for next generation
                    children.append(c)
                #     # replace population
                    pop = children
    return [best, best_eval]


def main():
    account = Account(100)
    data = genfromtxt('BTCUSDT-1m-2023-01-14.csv', delimiter=',')

    # define range for input
    param_bounds = {
        "sma1": [1, 15],
        "sma2": [2, 10]
    }
    # define the total iterations
    n_iter = 20
    # bits per variable
    n_pop = 30
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1

    processes = 1
    # perform the genetic algorithm search
    best, score = genetic_algorithm(
        simulate, sma_strategy, account, data, param_bounds, n_iter, n_pop, r_cross, r_mut, processes)

    print('Done!')
    # print(simulate(account, data, sma_strategy, best))


if __name__ == "__main__":
    main()
