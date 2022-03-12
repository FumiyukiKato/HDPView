from algorithms.noised_data import NoisedData

def Identity(data, prng, epsilon, workload_optimized=False, workloads=[]):
    if workload_optimized:
        projected_data = {}
        datavector = data.datavector(flatten=False) 
        workload_num = len(workloads)
        for proj, _ in workloads:
            aggregate_cols = filter(lambda x: x not in proj, data.domain.attrs)
            indices = tuple([data.domain.attrs.index(col) for col in aggregate_cols])
            x = datavector.sum(axis=indices)
            y = x + prng.laplace(0.0, 1.0*workload_num/epsilon, data.project(proj).domain.shape)
            projected_data[proj] = y
        return NoisedData(data.domain.attrs, projected_data, workload_optimized)
    else:
        y = data.datavector(flatten=False) + prng.laplace(0.0, 1.0/epsilon, data.domain.shape)
        return NoisedData(data.domain.attrs, y, workload_optimized)
