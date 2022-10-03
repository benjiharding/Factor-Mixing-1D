import numpy as np
import multiprocessing as mp

"""
Parallel Differential Evolution
"""


def denormalize(min_, diff, matrix):
    return min_ + matrix * diff


def random_sample(population, exclude, rng, size=3):
    # Optimized version using numpy
    idxs = list(range(population.shape[0]))
    idxs.remove(exclude)
    sample = rng.choice(idxs, size=size, replace=False)
    return population[sample]


def rand1(target_idx, population, f, rng):
    a, b, c = random_sample(population, target_idx, rng)
    return a + f * (b - c)


def bound_repair(x):
    return np.clip(x, 0, 1)


def binomial_crossover(target, mutant, cr, rng):
    n = len(target)
    p = rng.uniform(size=n) < cr
    if not np.any(p):
        p[rng.integers(0, n)] = True
    return np.where(p, mutant, target)


def pde(
    fobj,
    args,
    bounds,
    mut=(0.5, 1),
    crossp=0.7,
    popsize=15,
    its=1000,
    seed=515151,
    nprocess=None,
):
    """parallel differential evolotuion of objective function `fobj`"""

    rng = np.random.default_rng(seed)
    dimensions = len(bounds)
    pop = rng.uniform(size=(popsize, dimensions))
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = denormalize(min_b, diff, pop)

    # initilaize process pool
    if nprocess is None:
        nprocess = mp.cpu_count() - 1
    pool = mp.Pool(nprocess)

    # call starmap which is blocking and preserves order
    fitness = pool.starmap(fobj, [(ind, args) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    objv = []

    for i in range(its):

        mutants = [
            bound_repair(rand1(j, pop, rng.uniform(low=mut[0], high=mut[1]), rng))
            for j in range(popsize)
        ]
        trials = [
            binomial_crossover(pop[j], mutants[j], crossp, rng) for j in range(popsize)
        ]
        trials_denorm = [denormalize(min_b, diff, trials[j]) for j in range(popsize)]

        fit = pool.starmap(fobj, [(ind, args) for ind in trials_denorm])
        fmin = np.min(fit)
        idx = np.argmin(fit)

        for j, f in enumerate(fit):
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trials[j]

        if fmin < fitness[best_idx]:
            best_idx = idx
            best = trials_denorm[idx]

        objv.append(fitness[best_idx])

    # close the process pool
    pool.close()
    pool.join()

    return best, fitness[best_idx], objv


def de(fobj, args, bounds, mut=(0.5, 1), crossp=0.7, popsize=15, its=1000, seed=515151):
    """differential evolotuion of objective function `fobj`"""

    rng = np.random.default_rng(seed)
    dimensions = len(bounds)
    pop = rng.uniform(size=(popsize, dimensions))
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = denormalize(min_b, diff, pop)
    fitness = np.asarray([fobj(ind, args) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    # main loop over iteations
    for i in range(its):

        # loop over population
        for j in range(popsize):

            d = rng.uniform(low=mut[0], high=mut[1])
            mutant = bound_repair(rand1(j, pop, d, rng))
            trial = binomial_crossover(pop[j], mutant, crossp, rng)
            trial_denorm = denormalize(min_b, diff, trial)
            f = fobj(trial_denorm, args)

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial

                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        yield best, fitness[best_idx]
