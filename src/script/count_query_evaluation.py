import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
import pickle
import json
import multiprocessing
import argparse

from dataset import Dataset
from count_table import CountTable
from workload_generator import *

from algorithms.identity import Identity
from algorithms.dawa import Dawa
from algorithms.noised_data import SynData

from hdmm import error

parser = argparse.ArgumentParser(description='Evaluate count range queries for different datasets.')
parser.add_argument('--dataset', type=str, help="used dataset [adult, small-adult, nume-adult, trafic, bitcoin, electricity, phoneme, jm, adding-adult] (default: adult)", default="adult")
parser.add_argument('--alg', type=str, help="used algorithm [all, hdpview, dawa, hdmm, identity, privbayes] (default:  all)", default="all")
parser.add_argument('--epsilon', type=float, help="privacy budget (default: 1.0)", default=1.0)
parser.add_argument('--workload', type=str, help="workloads that are used in experiments [small_set, all] (default: all)", default="all")
parser.add_argument('--times', type=int, help="number of synthetic dataset (default: 10)", default=10)
args = parser.parse_args()

root_dir = Path('__file__').resolve().parent
data_dir = root_dir / "data" / "preprocessed" / args.dataset
syn_data_dir = root_dir / "data" / "synthetic" / args.dataset
syn_data_dir.mkdir(parents=True, exist_ok=True)
result_dir = root_dir / "exp" / "result" / "count_query" / args.dataset
result_dir.mkdir(parents=True, exist_ok=True)
p_view_hdpview_dir = root_dir / "save" / "p_view" / "hdpview" / args.dataset
p_view_hdpview_dir.mkdir(parents=True, exist_ok=True)
p_view_privtree_dir = root_dir / "save" / "p_view" / "privtree" / args.dataset
p_view_privtree_dir.mkdir(parents=True, exist_ok=True)

if args.dataset in [ 'small-adult', 'phoneme' ]:
    datasetsize = 'small'

epsilon = args.epsilon
is_parallel = True
alg = args.alg

target_domains = ()
data = Dataset.load(data_dir / 'data.csv', data_dir / 'domain.json')
ct = CountTable.from_dataset(data)
ct.info()

global_data = None
global_noised_data = None

def measuring_job(workload):
    global global_data, global_noised_data
    proj, W = workload
    if type(proj) is not tuple:
        proj = (proj,)
    est = W.dot(global_noised_data.project(proj))
    true = W.dot(global_data.project(proj).datavector())
    err = np.abs(est - true)
    sq_err = np.square(est - true)
    return err, sq_err

def measuring_direct_job(params):
    workload = params
    global global_data, global_noised_data
    est = np.array([ global_noised_data.run_query(query) for query in workload ])
    true = np.array([ global_data.run_query(query) for query in workload ])
    err = np.abs(est - true)
    sq_err = np.square(est - true)
    return err, sq_err


def initializer(data, noised_data):
    global global_data, global_noised_data
    global_data = data
    global_noised_data = noised_data
    

def save_result_json(result_dir, workload_name, method_name, epsilon, target_domains=(), implicit=False, elapsed=0, times=0, rootmse=0, rootmse_std=0, mae=0, mae_std=0, mean_partition_num=0, mean_partition_num_std=0):
    filename = f"{workload_name}-{method_name}-{epsilon}.json"
    if len(target_domains):
        filename = f"{target_domains}-{filename}"
    if implicit:
        result = { "root of MSE": rootmse, "Time": elapsed }
    else:
        result = { "times": times, "rootmse": rootmse, "rootmse_std": rootmse_std, "mae": mae , "mae_std": mae_std}
    if mean_partition_num > 0:
        result['mean_partition_num'] = mean_partition_num
        result['mean_partition_num_std'] = mean_partition_num_std
    with (result_dir / filename).open("w") as f:
        json.dump(result, f)
    print("save result to: ", result_dir / filename)

def start_print(method_name, workload, eps):
    print('-'*60)
    print(f'\tmethod: {method_name}')
    print(f'\tworkload: {workload}')
    print(f'\tepsilon: {eps}')

# Estimation-based evaluation. 
# for HDMM, Identity
def exe_estimate(method_name, data, epsilon, implicit_workload, implicit_workload_name):
    start_print(method_name, implicit_workload_name, epsilon)
    start = time.time()
    
    if method_name == "HDMM":
        hdmm_template = hdmm.templates.KronPIdentity(Pmatrix(data.domain), data.domain.shape)
        hdmm_template.optimize(implicit_workload)
        start = time.time()
        rootmse = error.rootmse(implicit_workload, hdmm_template.strategy(), eps=epsilon)
    elif method_name == "IDENTITY":
        identity_template = hdmm.workload.Kronecker([hdmm.workload.Identity(dom) for dom in data.domain.shape ])
        rootmse = error.rootmse(implicit_workload, identity_template, eps=epsilon)
    else:
        raise Exception("method_name %s is not found in this implicit estimator." % method_name)

    elapsed = time.time() - start

    print(f'\tRMSE: {rootmse}')
    print(f'\tTime: {elapsed} [s]')
    save_result_json(result_dir, implicit_workload_name, method_name, epsilon, target_domains, implicit=True, elapsed=elapsed, rootmse=rootmse)


def exe_measure(method_name, data, epsilon, workload, workload_name, times, parallel=True, direct=False):
    start_print(method_name, implicit_workload_name, epsilon)
    if method_name == "HDPView":
        files = p_view_hdpview_dir.glob(f'p_view_eps_{epsilon}*')
        # files = p_view_hdpview_dir.glob(f'p_view_eps_{epsilon}*para.pickle')
    elif method_name == "PRIVTREE":
        files = p_view_privtree_dir.glob(f'p_view_eps_{epsilon}*')
    elif method_name == "SynPRIVBAYES":
        files = syn_data_dir.glob(f"privbayes_{epsilon}*")
    else:
        assert method_name in [ "IDENTITY", "DAWA" ], ("method_name %s is not found in this implicit estimator." % method_name)

    total = []
    sq_total = []
    partition_num_list = []
    for i in tqdm(range(times)):
        result_batch = []
        sq_batch = []
        prng = np.random.RandomState(i)
        
        if method_name == "IDENTITY":
            noised_data = Identity(data, prng, epsilon, workload_optimized=False, workloads=workload)
        elif method_name == "DAWA":
            noised_data, partition_num = Dawa(data, prng, epsilon, ratio=0.5)
            partition_num_list.append(partition_num)
        elif method_name == "HDPView" or method_name == "PRIVTREE":
            file = next(files)
            with open(file, 'rb') as f:
                noised_data, block_and_noise = pickle.load(f)
            partition_num = len(block_and_noise)
            partition_num_list.append(partition_num)
        elif method_name in [ "SynPRIVBAYES" ]:
            file = next(files)
            syn_df = pd.read_csv(file)
            noised_data = SynData(syn_df, data.domain)
        else:
            raise Exception("method_name %s is not found in this implicit estimator." % method_name)

        if direct:
            parallel = True

        if parallel:
            POOL_SIZE =  multiprocessing.cpu_count() - 1
            if direct:
                with multiprocessing.get_context('fork').Pool(POOL_SIZE, initializer, (data, noised_data)) as pool:
                    results_list = pool.map(measuring_direct_job, workload)
            else:
                with multiprocessing.get_context('fork').Pool(POOL_SIZE, initializer, (data, noised_data)) as pool:
                    results_list = pool.map(measuring_job, workload)

            for err, sq_err in results_list:
                result_batch.extend(list(err))
                sq_batch.extend(list(sq_err))

        else:
            for proj, W in workload:
                if type(proj) is not tuple:
                    proj = (proj,)
                est = W.dot(noised_data.project(proj))
                true = W.dot(data.project(proj).datavector())
                err = np.abs(est - true)
                sq_err = np.square(est - true)
                result_batch.extend(list(err))
                sq_batch.extend(list(sq_err))

        total.append(np.mean(result_batch))
        sq_total.append(np.sqrt(np.mean(sq_batch)))

    mae, mae_std = np.mean(total), np.std(total)
    rootmse, rootmse_std = np.mean(sq_total), np.std(sq_total)
    mean_partition_num = mean_partition_num_std = 0
    if len(partition_num_list) > 0:
        mean_partition_num, mean_partition_num_std = np.mean(partition_num_list), np.std(partition_num_list)

    print(f'\tRMSE: {rootmse}')
    print(f'\tAveraged by: {times} try')
    save_result_json(result_dir, workload_name, method_name, epsilon, target_domains, times=times, mae=mae, mae_std=mae_std, rootmse=rootmse, rootmse_std=rootmse_std, mean_partition_num=mean_partition_num, mean_partition_num_std=mean_partition_num_std)


if __name__ == '__main__':
    all_start_time = time.time()

    explicit_workloads = []
    explicit_random_queries = []
    direct_workloads = [] #  using direct representation, instead using matrix representation for range query for speed up
    implicit_workloads = []

    if args.workload == "all":
        explicit_workloads.append(allRangeKway(domain=data.domain, k=1))
        explicit_workloads.append(allMarginalKway(domain=data.domain, k=2))
        explicit_workloads.append(allMarginalKway(domain=data.domain, k=3))
        explicit_workloads.append(allPrefixKD(domain=data.domain, k=2))
        explicit_workloads.append(allPrefixKD(domain=data.domain, k=3))
        for i in range(1):
            explicit_random_queries.append(random_range_queries(domain=data.domain, size=3000, dim_size=2, seed=i))
            explicit_random_queries.append(random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=i))
            explicit_random_queries.append(random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=i))
        for i in range(1):
            direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=2, seed=i))
            direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=i))
            direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=i))

        implicit_workloads.append(implicit_allRangeKway(domain=data.domain, k=1))
        implicit_workloads.append(implicit_allMarginalKway(domain=data.domain, k=2))
        implicit_workloads.append(implicit_allMarginalKway(domain=data.domain, k=3))    
        implicit_workloads.append(implicit_allPrefixKD(domain=data.domain, k=2))
        implicit_workloads.append(implicit_allPrefixKD(domain=data.domain, k=3))
        for i in range(1):
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=2, seed=i))
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=i))
            implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=i))
    
    elif args.workload == "small_set":
        explicit_workloads.append(allRangeKway(domain=data.domain, k=1))
        explicit_workloads.append(allMarginalKway(domain=data.domain, k=3))
        explicit_workloads.append(allPrefixKD(domain=data.domain, k=3))
        explicit_random_queries.append(random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=0))
        explicit_random_queries.append(random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=0))
        direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=0))
        direct_workloads.append(direct_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=0))

        implicit_workloads.append(implicit_allRangeKway(domain=data.domain, k=1))
        implicit_workloads.append(implicit_allMarginalKway(domain=data.domain, k=3))    
        implicit_workloads.append(implicit_allPrefixKD(domain=data.domain, k=3))
        implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=3, seed=0))
        implicit_workloads.append(implicit_random_range_queries(domain=data.domain, size=3000, dim_size=4, seed=0))        

    else:
        assert False, f"workloads {args.workload} is not existing. [ small_set or all ]."

    for implicit_workload_name, implicit_workload in implicit_workloads:
        if alg in ["all", "hdmm"]:
            exe_estimate("HDMM", data, epsilon, implicit_workload, implicit_workload_name)

        if alg in ["all", "identity"]:
            exe_estimate("IDENTITY", data, epsilon, implicit_workload, implicit_workload_name)

    for explicit_workload_name, explicit_workload in explicit_workloads:
        if alg in ["all", "identity"] and datasetsize == "small":
            exe_measure("IDENTITY", data, epsilon, explicit_workload, explicit_workload_name, args.times, parallel=is_parallel)

        if alg in ["all", "dawa"] and datasetsize == "small":
            exe_measure("DAWA", data, epsilon, explicit_workload, explicit_workload_name, args.times, parallel=is_parallel)

        if alg in ["all", "hdpview"]:
            exe_measure("HDPView", data, epsilon, explicit_workload, explicit_workload_name, args.times, parallel=is_parallel)

        if alg in ["all", "privtree"]:
            exe_measure("PRIVTREE", data, epsilon, explicit_workload, explicit_workload_name, args.times, parallel=is_parallel)

        if alg in ["all", "privbayes"]:
            exe_measure("SynPRIVBAYES", data, epsilon, explicit_workload, explicit_workload_name, args.times, parallel=is_parallel)

    for explicit_workload_name, explicit_workload in explicit_random_queries:
        if alg in ["all", "identity"] and datasetsize == "small":
            exe_measure("IDENTITY", data, epsilon, explicit_workload, explicit_workload_name, args.times, parallel=is_parallel)

        if alg in ["all", "dawa"] and datasetsize == "small":
            exe_measure("DAWA", data, epsilon, explicit_workload, explicit_workload_name, args.times, parallel=is_parallel)

    for direct_workload_name, direct_workload in direct_workloads:
        if alg in ["all", "hdpview"]:
            exe_measure("HDPView", data, epsilon, direct_workload, direct_workload_name, args.times, parallel=is_parallel, direct=True)
        
        if alg in ["all", "privtree"]:
            exe_measure("PRIVTREE", data, epsilon, direct_workload, direct_workload_name, args.times, parallel=is_parallel, direct=True)

        if alg in ["all", "privbayes"]:
            exe_measure("SynPRIVBAYES", data, epsilon, direct_workload, direct_workload_name, args.times, parallel=is_parallel, direct=True)            
    

    print('done: ', time.time() - all_start_time)