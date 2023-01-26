import signal as sig
from numpy import genfromtxt
from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt
import sys
import numpy as np
from multiprocessing import Pool
import copy
import time


class Account:
    def __init__(self, initial_amount) -> None:
        self.a_amount = initial_amount
        self.b_amount = 0

    def a_to_b(self, pair_rate):
        self.b_amount = self.a_amount / pair_rate
        self.a_amount = 0

    def a_as_b(self, pair_rate):
        return self.a_amount / pair_rate

    def b_to_a(self, pair_rate):
        self.a_amount = self.b_amount * pair_rate
        self.b_amount = 0

    def b_as_a(self, pair_rate):
        return self.b_amount * pair_rate

    def __repr__(self) -> str:
        return f"A: {self.a_amount}\nB: {self.b_amount}"


# Returns a array with buy/sell flags
def sma_strategy(data: np.ndarray, params):

    sma1 = sma(data, params.get("sma1"))
    sma2 = sma(data, params.get("sma2"))
    longs = cross_over(sma1, sma2)
    exits = cross_over(sma2, sma1) * -1
    return merge_signals(longs, exits)


def merge_signals(l, s):
    for i in range(len(l)):
        if s[i] == -1:
            l[i] = -1
    return l


def cross_over(a, b):
    crossovers = [np.NaN]
    for i in range(1, len(a)):
        if a[i - 1] < b[i - 1] and a[i] > b[i]:
            crossovers = np.append(crossovers, [1])
        else:
            crossovers = np.append(crossovers, [0])
    return crossovers


def sma(data, window):
    sma = np.array([np.NaN] * window)
    for i in range(window, len(data)):
        sma = np.append(sma, np.average(data[i - window : i]))
    return sma


def simulate(account: Account, prices: np.ndarray, strategy, params):
    signals = strategy(prices, params)
    for price, signal in zip(prices, signals):
        if signal == 1 and account.a_amount != 0:
            account.a_to_b(price)
        if signal == -1 and account.b_amount != 0:
            account.b_to_a(price)
    return account.a_amount + account.b_as_a(prices[-1])


def selection(pop, scores, k=3):
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    keys = p1.keys()
    if rand() < r_cross:
        for k in keys:
            c1[k] = p2[k]
            c1[k] = p1[k]
    return [c1, c2]


def mutation(params, r_mut, bounds):
    for k in params.keys():
        diff = randint(-r_mut, r_mut)
        if params[k] + diff in range(bounds.get(k)[0], bounds.get(k)[1]):
            params[k] = params[k] + diff


def genetic_algorithm(
    sim, strat, acc, data, bounds, n_iter, n_pop, r_cross, r_mut, processes
):
    population = []
    for _ in range(n_pop):
        params = {}
        for key in bounds:
            params[key] = randint(bounds[key][0], bounds[key][1])

        population.append(params)

    account = copy.copy(acc)
    best, best_eval = sys.maxsize, sim(account, data, strat, population[0])

    for _ in range(n_iter):
        with Pool(processes) as p:
            args = [(copy.copy(acc), data, strat, params) for params in population]
            scores = p.starmap(sim, args)

        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]

            selected = [selection(population, scores) for _ in range(n_pop)]
            children = list()

            for i in range(0, n_pop, 2):
                p1, p2 = selected[i], selected[i + 1]
                for c in crossover(p1, p2, r_cross):

                    mutation(c, r_mut, bounds)
                    children.append(c)
                    population = children

    return [best, best_eval]


def render(t, p):
    fig, ax = plt.subplots()
    ax.plot(p, t)
    plt.show()


def main():
    print("Calculating...")
    account = Account(100)
    data = genfromtxt("data.csv")

    param_bounds = {"sma1": [1, 100], "sma2": [1, 100]}
    n_iter = 50
    # n_pop must be even number
    # we don't want to leave anyone alone :) right?
    n_pop = 100
    r_cross = 0.3
    r_mut = 3

    times = []
    procs = []
    for processes in range(1, 9):
        start = time.time()
        try:
            # perform the genetic algorithm search
            best, score = genetic_algorithm(
                simulate,
                sma_strategy,
                account,
                data,
                param_bounds,
                n_iter,
                n_pop,
                r_cross,
                r_mut,
                processes,
            )

            finish = time.time()
            run_time = finish - start
            print(f"Done in:{run_time}s, using{processes} processes")
            print(best, score)

            times.append(run_time)
            procs.append(processes)

        except KeyboardInterrupt:
            sig.signal(sig.SIGINT, sig.SIG_IGN)
            sig.signal(sig.SIGTERM, sig.SIG_IGN)

            print("Shutting down")
            exit(0)

    render(times, procs)


if __name__ == "__main__":
    main()
