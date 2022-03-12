import numpy as np
import pandas as pd
from pathlib import Path
import copy
import json
import re
import argparse


parser = argparse.ArgumentParser(description='Discretize synthesized data.')

parser.add_argument('--dataset', type=str, help="used dataset [adult, small-adult, nume-adult, trafic, bitcoin, electricity, phoneme, jm, adding-adult, all] (default: adult)", default="adult")
parser.add_argument('--epsilon', type=float, help="privacy budget (default: 1.0)", default=1.0)
parser.add_argument('--alg', type=str, help="used algorithm [privbayes, dp-gan] (default:  privbayes)", default="privbayes")
parser.add_argument('--parameter_search', action='store_true')
args = parser.parse_args()

root_dir = Path('__file__').resolve().parent
data_dir = root_dir / "data" / "preprocessed"
raw_name = "raw.csv"
all_raw_name = "all_raw.csv"
domain_name = "domain.json"
privbayes_dir = root_dir / "competitors" / "privbayes"
privbayes_synthetic_data_dir = privbayes_dir / "synthetic_data" / "cq"
syn_data_dir = root_dir / "data" / "synthetic"
syn_data_dir.mkdir(parents=True, exist_ok=True)

epsilon = args.epsilon
dataset = args.dataset
alg = args.alg

def make_discretizer(series, limit_bin):
    if type(series[0]) is str or type(series[0]) is np.bool_:
        value_set = sorted(list(set(series)))
        n_val = len(value_set)
        def discretizer(value):
            return value_set.index(value)
    else:
        max = series.max()
        min = series.min()
        if type(series[0]) in [int, np.int64, np.int32] and max-min <= limit_bin:
            def discretizer(value):
                return int(value-min)
            n_val = int(max-min) + 1
        else:
            def discretizer(value):
                return int(limit_bin * (value-min) / (max+0.1-min))
            n_val = int(limit_bin)
    return discretizer, n_val
    
def discretize(original_df, sampled_df, limit_bin):
    df_ = copy.deepcopy(sampled_df)
    for col in original_df.columns:
        discretizer, n_val = make_discretizer(original_df[col], limit_bin)
        df_[col] = sampled_df[col].map(discretizer)
    return df_

# prepare the same discretizer used in preprocess on the original data
def get_discretizers(original_df, limit_bin):
    discretizers = []
    for col in original_df.columns:
        discretizer, n_val = make_discretizer(original_df[col], limit_bin)
        discretizers.append((col, discretizer))
    return discretizers

def get_limit_bin(dataset):
    return {
        "adult": 100,
        "bitcoin": 30,
        "electricity": 100,
        "phoneme": 10,
        "trafic": 100,
        'nume-adult': 100,
        'small-adult': 100,
        'adult-a': 100,
        'adult-b': 100,
        'adult-c': 100,
        'adult-d': 100,
        'adult-e': 100,
        'adult-f': 100,
        'adult-g': 100,
        'adult-1': 100,
        'adult-2': 100,
        'adult-3': 100,
        'adult-4': 100,
        'adult-5': 100,
        'jm': 10
    }.get(dataset)

def extract_i(file_name):
    pattern = '.+_(\d+).csv$'
    result = re.match(pattern, file_name)
    return result.group(1)

def get_domain(dataset):
    with (data_dir / dataset  / domain_name).open(mode="r") as f:
        domain = json.load(f)
    return domain

def get_privbayes_files(dataset, eps):
    return (privbayes_synthetic_data_dir / dataset).glob(f'raw_privbayes_{eps}*')

def get_syn_data_dir(dataset):
    given_syn_data_dir = syn_data_dir / dataset
    given_syn_data_dir.mkdir(parents=True, exist_ok=True)
    return given_syn_data_dir

def run_privbayes_files(dataset, eps):
    if dataset == "bitcoin":
        original_df = pd.read_csv(data_dir / dataset / all_raw_name, header=None, sep=' ')
    else:
        original_df = pd.read_csv(data_dir / dataset / raw_name, header=None, sep=' ')
    discretizers = get_discretizers(original_df, get_limit_bin(dataset))
    given_syn_data_dir = get_syn_data_dir(dataset)
    columns = list(get_domain(dataset).keys())
    
    files = get_privbayes_files(dataset, eps)
    for file in files:
        print(file)
        i = extract_i(file.name)
        sampled_df = pd.read_csv(file, header=None)
        sampled_df.columns = original_df.columns
        syn_df = copy.deepcopy(sampled_df)        
        
        for col, discretizer in discretizers:       
            syn_df[col] = sampled_df[col].map(discretizer)
        
        syn_df.columns = columns
        syn_df.to_csv(given_syn_data_dir / f"privbayes_{eps}_{i}.csv", index=None)

if __name__ == '__main__':
    run_privbayes_files(dataset, epsilon)
