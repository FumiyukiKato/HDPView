import numpy as np
from ektelo.algorithm.dawa.partition_engines import l1partition
from algorithms.noised_data import NoisedData

def Dawa(data, prng, epsilon, ratio):
    count_tensor = data.datavector(flatten=False)
    count_vector, partition_num = dawaPartition(count_tensor, epsilon, ratio, prng)
    return NoisedData(data.domain.attrs, count_vector.reshape(data.domain.shape), False), partition_num

def dawaPartition(count_tensor, epsilon, ratio, prng):
    """Dawa partitioning with `dpcomp_core.algorithm.dawa`
    Args:
        count_tensor (np.array): raw data of count tensor
        epsilon (float): privacy budget
        ratio (float): budget ratio
        seed (int): random seed
    Returns:
        NoisedData
    """
    count_vector = count_tensor.ravel().astype('int')
    pSeed = prng.randint(1000000)
    # partitioning phase
    partition = l1partition.l1partition_approx_engine().Run(count_vector, epsilon, ratio, pSeed)
    partition_num = len(partition)
    # print('[DAWA] number of dawa partition: ', partition_num)
    # perturbation phase not optimized for workload
    noise_vector = prng.laplace(0.0, 1.0 / ((1-ratio) * epsilon), len(count_vector))
    for (start, end) in partition:
        if start != end:
            bucket_size = end+1-start
            noise_vector[start:end+1] = noise_vector[start] / bucket_size
            count_vector[start:end+1] = count_vector[start:end+1].sum() / bucket_size
    
    count_vector = count_vector.astype('float')
    count_vector += noise_vector

    return count_vector, partition_num