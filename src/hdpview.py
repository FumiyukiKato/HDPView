import numpy as np
import multiprocessing
import functools
import itertools

from count_table import NoisedCountTable, BlockResult, calculate_mean_and_aggregation_error, implicit_ae

"""
Python hdpview
"""
def run(block, epsilon, ratio, prng, alpha=2, beta=1.2, gamma=1.0, theta=None, verbose=False):
    """Run HDPView
        1st phase, divide blocks.
        2nd phase, perturbation.
        Prepare parameters and execute HDPView

    Args:
        block (CountTable): block
        epsilon (float): privacy budget
        ratio (float): ubdget ratio of block division and perturbation, 0 to 1 value
        prng (np.random.RandomState): random state
        alpha (float), beta(float), gamma(float)
        verbose (bool)
    """
    seed = prng.randint(0, 2949672950)
    block.set_random(seed)
    if verbose:
        print("seed: ", seed)

    n_dash = block.size()
    kappa = np.ceil(np.log2(n_dash)*beta)
    epsilon_r = epsilon * ratio
    epsilon_p = epsilon * (1 - ratio)
    if theta is None:
        theta = 1/epsilon_p
    epsilon_cut = (1 - gamma) * epsilon_r / kappa
    lamb = ((2 * alpha - 1)/(alpha - 1) + 1) * (2 / (gamma * epsilon_r))
    delta = lamb*np.log(alpha)

    # prepare shared memories for parallelization
    manager = multiprocessing.Manager()
    block_queue = manager.Queue()
    block_queue.put(block)
    block_result_list = []
    
    MAX_PROCESS = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(MAX_PROCESS)
    
    while True:
        async_results = []
        while not block_queue.empty():
            result = pool.apply_async(
                recursive_bisection, (block_queue.get(), block_queue, epsilon_cut, kappa, theta, lamb, delta, verbose)
            )
            async_results.append(result)
        results = list(itertools.chain.from_iterable([ r.get() for r in async_results ]))
        block_result_list.extend(results)
        if block_queue.empty():
            break

    block_result_list.sort(key=functools.cmp_to_key(range__gt__))

    for block_result in block_result_list:
        mean, ae = calculate_mean_and_aggregation_error(block, block_result.domain_dict)
        block_result.mean = mean
        block_result.aggregation_error = ae
        pe = prng.laplace(0.0,  1.0 / epsilon_p)
        block_result.perturbation_error = pe
        
    return NoisedCountTable.from_count_table(block, block_result_list), block_result_list

def recursive_bisection(block, block_queue, epsilon_cut, depth_max, theta, lamb, delta, verbose=False):
    """Random cut and random converge

    Args:
        block_queue (multiprocessing.Queue): Shared queue to store blocks to be executed

    Returns:
        [{"range": {int: (int,int)}, "mondrian_budget": float, "depth": int}]
    """
    # Random cut
    if verbose:
        print('Before cut', block.domain_dict)
    if block.depth > depth_max:
        axis, index = cut_random(block)
    else:
        axis, index = cut_exp_mech(block, epsilon_cut)
    if verbose:
        print(axis, index)
    left_block, right_block = block.split(axis, index)

    # Random converge
    converged_block_results = []
    if left_block.size() == 1:
        converged_block_results.append(BlockResult(left_block.domain_dict, left_block.depth))
    elif random_converge(left_block, left_block.depth, theta, lamb, delta):
        converged_block_results.append(BlockResult(left_block.domain_dict, left_block.depth))
    else:
        block_queue.put(left_block)

    if right_block.size() == 1:
        converged_block_results.append(BlockResult(right_block.domain_dict, right_block.depth))
    elif random_converge(right_block, right_block.depth, theta, lamb, delta):
        converged_block_results.append(BlockResult(right_block.domain_dict, right_block.depth))
    else:
        block_queue.put(right_block)

    return converged_block_results


def range__gt__(block_result, other):
    for dim, dom_range in block_result.domain_dict.items():
        other_range = other.domain_dict[dim]
        if dom_range > other_range:
            return 1
        elif dom_range < other_range:
            return -1
    return 0


def exp_mech(prng, eps , scores, targets, sensitivity):
    """Exponential mechanism
    """
    if sensitivity == 0:
        index = prng.choice(len(targets))
        return targets[index]

    np_scores = np.array(scores)
    score_max = np.max(np_scores)
    weights = np.exp((eps*(np_scores - score_max)) / (2*sensitivity))
    total_weight = np.sum(weights)
    cum_weights = np.cumsum(weights) / total_weight
    index = prng.rand()
    return targets[cum_weights.searchsorted(index)]


def cut_random(block):
    dim_list = list(block.cardinality_dict.keys())
    while True:
        d = block.prng.choice(dim_list)
        cardinality = block.cardinality_dict[d]
        if cardinality > 1:
            break
    i = block.prng.choice(cardinality-1)
    return (d, i)


# left AE + right AE
def cut_exp_mech(block, epsilon_cut_per_depth):
    scores = []
    targets = []
    for d in block.index:
        cardinality = block.cardinality_dict[d]
        for i in np.arange(cardinality-1):
            scores.append(- np.sum(block.split_values(d, i)))
            targets.append((d, i))

    return exp_mech(
        block.prng,
        epsilon_cut_per_depth, 
        scores,
        targets,
        2*(2 - 2/block.size())
    )


def random_converge(block, depth, theta, lamb, delta):
    ae = implicit_ae(block.values(), block.zero_num())
    b = ae - (delta*depth)
    b = max(b, (theta + 2 - delta))
    noise = block.prng.laplace(0.0, scale=lamb)
    noisy_b = b + noise
    return noisy_b <= theta
