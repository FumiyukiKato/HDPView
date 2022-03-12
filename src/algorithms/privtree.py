import itertools
import multiprocessing
import numpy as np
from count_table import NoisedCountTable, BlockResult, calculate_mean_and_aggregation_error, implicit_ae
from query import Query, QueryCondition
import matplotlib.pyplot as plt

## PrivTree
def run(table, prng, epsilon, fanout, theta=0, cut_max=None, workloads=[]):
    range_and_budget_list = PrivTree(table, prng, epsilon, fanout, theta, cut_max)
    return NoisedCountTable.from_count_table(table, range_and_budget_list), range_and_budget_list

def PrivTree(block, prng, epsilon, fanout, theta, cut_max, verbose=False):
    scale = (2*fanout - 1)/(fanout - 1)*(1/(epsilon/2))
    delta = scale*np.log(fanout)
    domain_names = list(block.domain_dict.keys())
    block_result_list = []
    tree_depth = 0
    round_robin_cursor = 0

    if verbose:
        print("theta: ", theta)
        print("lambda: ", scale)
        print("delta: ", delta)

    # prepare shared memories for parallelization
    manager = multiprocessing.Manager()
    unvisited_domain_with_depth_queue = manager.Queue()
    if cut_max:
        unvisited_domain_with_depth_queue.put((list(block.domain_dict.values()), tree_depth, round_robin_cursor))
    else:
        unvisited_domain_with_depth_queue.put((list(block.domain_dict.values()), tree_depth))

    MAX_PROCESS = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(MAX_PROCESS)

    while True:
        async_results = []
        while not unvisited_domain_with_depth_queue.empty():
            seed = prng.randint(2147483647)
            result = pool.apply_async(
                recursive_cut, (
                    block,
                    domain_names,
                    delta,
                    theta,
                    scale,
                    epsilon,
                    seed,
                    cut_max,
                    unvisited_domain_with_depth_queue.get(),
                    unvisited_domain_with_depth_queue)
            )
            async_results.append(result)
        results = list([ r.get() for r in async_results if r.get() is not None])
        block_result_list.extend(results)
        if unvisited_domain_with_depth_queue.empty():
            break

    # single thread version
    # unvisited_domains = []
    # unvisited_domains_per_depth = []
    # tree_depth = 0
    # unvisited_domains.append(list(block.domain_dict.values()))
    # while len(unvisited_domains) > 0:
    #     unvisited_domains_per_depth.extend(unvisited_domains)
    #     unvisited_domains = []
    #     for unvisited_domain in unvisited_domains_per_depth:
    #         count_query = Query([QueryCondition(domain_name, domain_range[0], domain_range[1]) for domain_name, domain_range in zip(domain_names, unvisited_domain)])
    #         count = block.run_query(count_query)
    #         b = count - (delta*tree_depth)
    #         b = max(b, (theta - delta))
    #         noise = prng.laplace(0.0, scale=scale)
    #         noisy_b = b + noise
    #         if (noisy_b > theta) and can_divide(unvisited_domain):
    #             unvisited_domains.extend(split_domains(unvisited_domain))
    #         else:
    #             ranges = domain_dict_from_domain_list(domain_names, unvisited_domain)
    #             values, zero_num = block.get_sub_block_values(list(ranges.items()))
    #             block_result_list.append(BlockResult(
    #                 domain_dict=ranges,
    #                 depth=tree_depth,
    #                 mean=count / count_domain_dict(unvisited_domain),
    #                 aggregation_error=implicit_ae(values, zero_num),
    #                 perturbation_error=prng.laplace(0.0, scale=1/(epsilon/2)),
    #             ))
    #     unvisited_domains_per_depth = []
    #     tree_depth += 1
    
    return block_result_list


def recursive_cut(block, domain_names, delta, theta, scale, epsilon, seed, cut_max, unvisited_domain_with_depth, unvisited_domain_with_depth_queue):
    if cut_max:
        unvisited_domain, depth, round_robin_cursor = unvisited_domain_with_depth
    else:
        unvisited_domain, depth = unvisited_domain_with_depth
    total_count_query = Query([QueryCondition(domain_name, domain_range[0], domain_range[1]) for domain_name, domain_range in zip(domain_names, unvisited_domain)])
    count = block.run_query(total_count_query)
    b = count - (delta*depth)
    b = max(b, (theta - delta))
    prng = np.random.RandomState(seed)
    noise = prng.laplace(0.0, scale=scale)
    noisy_b = b + noise
    if (noisy_b > theta) and can_divide(unvisited_domain):
        if cut_max:
            domains, updated_round_robin_cursor = split_domains(unvisited_domain, cut_max, round_robin_cursor)
            for domain in domains:
                unvisited_domain_with_depth_queue.put((domain, depth+1, updated_round_robin_cursor))
        else:
            for domain in split_domains(unvisited_domain, cut_max):
                unvisited_domain_with_depth_queue.put((domain, depth+1))
        return None
    else:
        ranges = domain_dict_from_domain_list(domain_names, unvisited_domain)
        values, zero_num = block.get_sub_block_values(list(ranges.items()))
        block_result = BlockResult(
            domain_dict=ranges,
            depth=depth,
            mean=count / count_domain_dict(unvisited_domain),
            aggregation_error=implicit_ae(values, zero_num),
            perturbation_error=prng.laplace(0.0, scale=1/(epsilon/2)),
        )
        return block_result


def can_divide(domain):
    return count_domain_dict(domain) > 1


def split_domains(unvisited_domain, cut_max, round_robin_cursor=0):
    half_divided_domains = []
    if cut_max: # for round_robin_cut
        l = len(unvisited_domain)
        round_robin_targets = (list(range(l)) + list(range(l)))[round_robin_cursor:round_robin_cursor+cut_max]
        updated_round_robin_cursor = (round_robin_targets[-1] + 1) % l
        round_robin_targets = set(round_robin_targets)
        for i, domain_range in enumerate(unvisited_domain):
            if domain_range[0] != domain_range[1] and i in round_robin_targets:
                half_divided_domains.append([(domain_range[0], (domain_range[0]+domain_range[1])//2), ((domain_range[0]+domain_range[1])//2+1, domain_range[1])])
            else:
                half_divided_domains.append([(domain_range[0], domain_range[1])])
        return list(itertools.product(*half_divided_domains)), updated_round_robin_cursor
    else:
        for domain_range in unvisited_domain:
            if domain_range[0] == domain_range[1]:
                half_divided_domains.append([(domain_range[0], domain_range[1])])
            else:
                half_divided_domains.append([(domain_range[0], (domain_range[0]+domain_range[1])//2), ((domain_range[0]+domain_range[1])//2+1, domain_range[1])])
        return list(itertools.product(*half_divided_domains))


def domain_dict_from_domain_list(domain_names, domains):
    return { domain_name: domain for domain_name, domain in zip(domain_names, domains)}


def count_domain_dict(domain):
    """All counts from domain dict
    """
    cardinality_list = []
    for val in domain:
        cardinality_list.append(val[1] - val[0] + 1)
    return np.prod(cardinality_list, dtype=np.float)


def plot_data(data, attrs, noised_data):
    assert len(attrs) == 2, "attrs must be 2 length"
    
    x = data.df[list(attrs)].iloc[:, 0].values
    y = data.df[list(attrs)].iloc[:, 1].values
    fig, ax = plot_locations_xy(x, y)
    print("number of rectangle", len(noised_data.blocks))
    for block in noised_data.blocks:
        domain = list(block.ranges.values())
        plot_rect(ax, block.ranges[attrs[0]][0]-0.5, block.ranges[attrs[0]][1]+0.5, block.ranges[attrs[1]][0]-0.5, block.ranges[attrs[1]][1]+0.5)

    v0 = [noised_data.domain_dict[attrs[0]][0], noised_data.domain_dict[attrs[0]][1], noised_data.domain_dict[attrs[1]][0], noised_data.domain_dict[attrs[1]][1]]
    print('v0', v0)
    v0_w = v0[1] - v0[0]
    v0_h = v0[3] - v0[2]

    ax.set_xlim(v0[0]-0.05*v0_w, v0[1]+0.05*v0_w)
    ax.set_ylim(v0[2]-0.05*v0_h, v0[3]+0.05*v0_h)
    plt.show()

    return fig
    
    
def plot_locations_xy(x, y, alpha=0.3, marker='o', marker_size=5, figsize=(16,8), save_plot_dir=None, dpi=250):

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.scatter(x, y,alpha=alpha, marker=marker, s=marker_size)
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)
    ax.grid(alpha=0.8,linewidth=0.5)
        
    return fig, ax

def plot_rect(ax, left, right, bottom, top, fill=False, color='r', edgecolor='r', alpha=1, hatch=None):
    p = plt.Rectangle((left, bottom), right-left, top-bottom, fill=fill, color=color, edgecolor=edgecolor, alpha=alpha, hatch=hatch)
    ax.add_patch(p)