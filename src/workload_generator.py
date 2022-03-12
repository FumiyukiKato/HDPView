import numpy as np
import hdmm.templates, hdmm.workload, hdmm.matrix
import ektelo.workload
import itertools
from query import Query, QueryCondition


ALL_RANGE    = lambda k: f"{k}-way-All-Range"
MARGINAL     = lambda k: f"{k}-way-All-Marginal"
PREFIX       = lambda k: f"Prefix-{k}D"
RANDOM_RANGE = lambda size, dim_size, seed: f"Random-Range-queries-({size}-{dim_size}-{seed})"
IMPLICIT     = lambda query: f"implicit-{query}"
RANDOM_MARGINAL = lambda size, dim_size, seed: f"Random-Marginal-queries-({size}-{dim_size}-{seed})"

def allRangeKway(domain, k):
    combinations = list(itertools.combinations(domain.attrs, k))
    workloads = []
    for attrs in combinations:
        workloads.append((attrs, ektelo.workload.Kronecker([ ektelo.workload.AllRange(domain[d]) for d in attrs ])))
    return ALL_RANGE(k), workloads

def allMarginalKway(domain, k):
    combinations = list(itertools.combinations(domain.attrs, k))
    workloads = []
    for attrs in combinations:
        workloads.append((attrs, ektelo.workload.Kronecker([ ektelo.workload.Identity(domain[d]) for d in attrs ])))
    return MARGINAL(k), workloads

def allPrefixKD(domain, k):
    combinations = list(itertools.combinations(domain.attrs, k))
    workloads = []
    for attrs in combinations:
        workloads.append((attrs, ektelo.workload.Kronecker([ ektelo.workload.Prefix(domain[d]) for d in attrs ])))
    return PREFIX(k), workloads

def random_range_queries(domain, size, dim_size, seed=0):
    return RANDOM_RANGE(size, dim_size, seed), _gen_random_range(domain, size, dim_size, seed=seed, implicit=False)

def random_marginal_queries(domain, size, dim_size, seed=0):
    return RANDOM_MARGINAL(size, dim_size, seed), _gen_random_range(domain, size, dim_size, seed=seed, implicit=False, marginal=True)

def implicit_allRangeKway(domain, k):
    combinations = list(itertools.combinations(domain.attrs, k))
    workloads = []
    for attrs in combinations:
        query = []
        for d in domain.attrs:
            if d in attrs:
                query.append(hdmm.workload.AllRange(domain[d]))
            else:
                query.append(hdmm.workload.Total(domain[d]))
        workloads.append(hdmm.workload.Kronecker(query))
    return IMPLICIT(ALL_RANGE(k)), hdmm.workload.VStack(workloads)

def implicit_allMarginalKway(domain, k, le=False):
    if le:
        k_list = list(range(1, k+1))
    else:
        k_list = [k]
        
    workloads = []
    for i in k_list:
        combinations = list(itertools.combinations(domain.attrs, i))
        for attrs in combinations:
            query = []
            for dom in domain:
                dom_size = domain[dom]
                if dom in attrs:
                    query.append(hdmm.workload.Identity(dom_size))
                else:
                    query.append(hdmm.workload.Total(dom_size))
            workloads.append(hdmm.workload.Kronecker(query))
    return IMPLICIT(MARGINAL(k)), hdmm.workload.VStack(workloads)

def implicit_allPrefixKD(domain, k):
    combinations = list(itertools.combinations(domain.attrs, k))
    workloads = []
    for attrs in combinations:
        query = []
        for d in domain.attrs:
            if d in attrs:
                query.append(hdmm.workload.Prefix(domain[d]))
            else:
                query.append(hdmm.workload.Total(domain[d]))
        workloads.append(hdmm.workload.Kronecker(query))
    return IMPLICIT(PREFIX(k)), hdmm.workload.VStack(workloads)

def implicit_random_range_queries(domain, size, dim_size, seed=0, merge=True):
    return IMPLICIT(RANDOM_RANGE(size, dim_size, seed)), _gen_random_range(domain, size, dim_size, seed=seed, implicit=True, merge=merge)

def implicit_random_marginal_queries(domain, size, dim_size, seed=0, merge=True):
    return IMPLICIT(RANDOM_MARGINAL(size, dim_size, seed)), _gen_random_range(domain, size, dim_size, seed=seed, implicit=True, merge=merge, marginal=True)

def direct_random_range_queries(domain, size, dim_size, seed=0):
    return RANDOM_RANGE(size, dim_size, seed), _gen_random_range(domain, size, dim_size, seed=seed, direct=True)

def direct_random_marginal_queries(domain, size, dim_size, seed=0):
    return RANDOM_MARGINAL(size, dim_size, seed), _gen_random_range(domain, size, dim_size, seed=seed, direct=True, marginal=True)

def _gen_random_range(domain, size, dim_size, seed=0, implicit=False, merge=False, direct=False, marginal=False):
    domain_shape = domain.shape
    if type(domain_shape) is int:
        domain = (domain_shape,)
        
    prng = np.random.RandomState(seed)
    combinations = list(itertools.combinations(domain.attrs, dim_size))
    combination_len = len(combinations)
    queries = {}
    
    for i in range(size):
        random_idx = prng.choice(combination_len)
        selected_domains = combinations[random_idx]
        selected_domain_shape = [ domain[dom] for dom in selected_domains ]
        
        shape = tuple(prng.randint(1, dim+1, None) for dim in selected_domain_shape)
        lb = tuple(prng.randint(0, d - q + 1, None) for d,q in zip(selected_domain_shape, shape))
        if marginal:
            ub = lb
        else:
            ub = tuple(sum(x)-1 for x in zip(lb, shape))

        if queries.get(selected_domains):
            queries[selected_domains].append((lb, ub))
        else:
            queries[selected_domains] = [ (lb, ub) ]

    if direct:
        direct_queries = []
        for domains, bounds in queries.items():
            domains_queries = []
            for lb_tuple, ub_tuple in bounds:
                domains_queries.append(Query([ QueryCondition(d, lb, ub) for d, lb, ub in zip(domains, lb_tuple, ub_tuple) ]))
            direct_queries.append(domains_queries)
        return direct_queries

    
    if implicit:
        # in HDMM workload optimization, we can define workloads collectively or separately
        if merge:
            workloads = []
            for attrs, conditions in queries.items():
                merged_query = []
                conditions_len = len(conditions)
                condition_i = 0
                for dom in domain:
                    if dom in attrs:
                        implicit_w = np.zeros((conditions_len, domain[dom]))
                        for i, (lb_tuple, ub_tuple) in enumerate(conditions):
                            implicit_w[i, lb_tuple[condition_i]:ub_tuple[condition_i]+1] = 1.0
                        condition_i += 1
                        merged_query.append(hdmm.workload.EkteloMatrix(implicit_w))
                    else:
                        merged_query.append(hdmm.workload.Total(domain[dom]))
                workloads.append(hdmm.workload.Kronecker(merged_query))    
            return hdmm.workload.VStack(workloads)
        else:
            workloads = []
            for attrs, conditions in queries.items():
                for lb_tuple, ub_tuple in conditions:
                    query = []
                    i = 0
                    for dom in domain:
                        dom_size = domain[dom]
                        if dom in attrs:
                            implicit_w = np.zeros((1, dom_size))
                            implicit_w[0, lb_tuple[i]:ub_tuple[i]+1] = 1.0
                            query.append(hdmm.workload.EkteloMatrix(implicit_w))
                            i += 1
                        else:
                            query.append(hdmm.workload.Total(dom_size))
                    workloads.append(hdmm.workload.Kronecker(query))
            return hdmm.workload.VStack(workloads)
    else:
        workloads = []
        for attrs, conditions in queries.items():
            selected_domain_shape = [ domain[dom] for dom in attrs ]
            workloads.append((attrs, ektelo.workload.RangeQueries.fromlist(selected_domain_shape, conditions) ))
        return workloads

def Pmatrix(domain, alpha=16):
    p_list = [ int(np.ceil(dom / alpha)) for dom in domain.shape]
    return tuple(p_list)