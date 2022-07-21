import numpy as np


# def de(fobj, args, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000, seed=515151):
#     """differential evolotuion of fobj"""
#     rng = np.random.default_rng(seed)
#     dimensions = len(bounds)
#     pop = rng.uniform(size=(popsize, dimensions))
#     min_b, max_b = np.asarray(bounds).T
#     diff = np.fabs(min_b - max_b)
#     pop_denorm = min_b + pop * diff
#     fitness = np.asarray([fobj(ind, *args) for ind in pop_denorm])
#     best_idx = np.argmin(fitness)
#     best = pop_denorm[best_idx]
#     for i in range(its):
#         for j in range(popsize):
#             idxs = [idx for idx in range(popsize) if idx != j]
#             a, b, c = pop[rng.choice(idxs, 3, replace=False)]
#             mutant = np.clip(a + mut * (b - c), 0, 1)
#             cross_points = rng.uniform(size=dimensions) < crossp
#             if not np.any(cross_points):
#                 cross_points[rng.randint(0, dimensions)] = True
#             trial = np.where(cross_points, mutant, pop[j])
#             trial_denorm = min_b + trial * diff
#             f = fobj(trial_denorm, *args)
#             if f < fitness[j]:
#                 fitness[j] = f
#                 pop[j] = trial
#                 if f < fitness[best_idx]:
#                     best_idx = j
#                     best = trial_denorm
#         yield best, fitness[best_idx]


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


def random_mutation(x, bounds, rng):
    D = len(bounds)
    xprime = [
        bounds[j][0] + rng.uniform() * (bounds[j][1] - bounds[j][0]) for j in range(D)
    ]
    return np.array(xprime)


def bound_repair(x):
    return np.clip(x, 0, 1)


def binomial_crossover(target, mutant, cr, rng):
    n = len(target)
    p = rng.uniform(size=n) < cr
    if not np.any(p):
        p[rng.integers(0, n)] = True
    return np.where(p, mutant, target)


def de(
    fobj,
    args,
    bounds,
    mut=(0.5, 1),
    crossp=0.7,
    popsize=15,
    its=1000,
    seed=515151,
    K=25,
    g=10e-6,
):
    """differential evolotuion of objective function `fobj`"""

    rng = np.random.default_rng(seed)
    dimensions = len(bounds)
    pop = rng.uniform(size=(popsize, dimensions))
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = denormalize(min_b, diff, pop)
    fitness = np.asarray([fobj(ind, *args) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    # random restart parameters
    f_its = np.zeros((popsize, its + 1))
    df_its = np.zeros((popsize, its + 1))
    rr_it = 0

    # main loop over iteations
    for i in range(its):

        # loop over population
        for j in range(popsize):

            d = rng.uniform(low=mut[0], high=mut[1])
            mutant = bound_repair(rand1(j, pop, d, rng))
            trial = binomial_crossover(pop[j], mutant, crossp, rng)
            trial_denorm = denormalize(min_b, diff, trial)
            f = fobj(trial_denorm, *args)

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial

                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

            # track fitness of population from generation to generation
            f_its[:, i] = fitness
            df_its[:, i] = np.abs(f_its[:, i] - f_its[:, i - 1])

        # do K iterations first before checking random restart criteria
        if i > K:

            # do K more iterations if we have already restarted
            if i > rr_it + K:

                # check the criteria
                check = np.where(df_its[:, (i - K) : i] < g, 1, 0)

                if np.sum(check) == K * popsize:

                    # reinitialize entire population but the best vector
                    fit = np.copy(pop[best_idx])
                    for jj in range(popsize):
                        pop[jj] = random_mutation(pop[jj], bounds, rng)
                    pop[best_idx] = fit
                    rr_it = i
                    print("Random Restart on Iteration:", rr_it + 1)

        yield best, fitness[best_idx]
