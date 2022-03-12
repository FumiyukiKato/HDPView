import numpy as np
import pandas as pd
from pathlib import Path
import json

root_dir = Path('__file__').resolve().parent
data_dir = root_dir / "data" / "preprocessed"

data_name = "data.csv"
train_name = "train.csv"
test_name = "test.csv"
domain_name = "domain.json"
config_name = "config.json"


def write_data(config, df, domain, index=None):
    dataset_name = config['dataset_name']
    dataset_name_dir = data_dir / dataset_name
    dataset_name_dir.mkdir(parents=True, exist_ok=True)
    if index:
        df.to_csv(dataset_name_dir / data_name)
    else:
        df.to_csv(dataset_name_dir / data_name, index=None)
    with (dataset_name_dir / domain_name).open(mode="w") as f:
            json.dump(domain, f)
    with (dataset_name_dir / config_name).open(mode="w") as f:
            json.dump(config, f, indent=4)


def rand_geometric(prng, cardinality, data_size, p=0.05):
    x1 = np.random.geometric(p, data_size)
    bins = np.linspace(x1.min(), x1.max(), cardinality)
    return np.digitize(x1, bins=bins) - 1


def rand_uniform(prng, cardinality, data_size):
    return prng.randint(0, cardinality, data_size)


def rand_gauss(prng, cardinality, data_size, loc=0, scale=1):
    x1 = np.random.normal(loc=loc, scale=scale, size=data_size)
    bins = np.linspace(x1.min(), x1.max(), cardinality)
    return np.digitize(x1, bins=bins) - 1


def shuffle_attributes(prng, a, cardinality):
    tansform_table = prng.permutation(np.arange(cardinality))
    transform = np.frompyfunc(lambda value: tansform_table[value], 1, 1)
    return transform(a)


def generate_artificial(schema, seed, data_size=0):
    """
    Args:
        schema ({str: {str: int, str: str}}): {'column name': {'cardinality': int, 'dist': str, 'shuffle': bool}
            ex) {'age': {'cardinality': 10, 'dist': 'uniform', 'shuffle': True}, 'race': {...
    """
    prng = np.random.RandomState(seed)
    generated_data = {}
    domain = {}
    
    for column, item in schema.items():
        if item['dist'] == "uniform":
            generated_data[column] =rand_uniform(prng, item['cardinality'], data_size)
        elif item['dist'] == "geometric":
            if item.get('p'):
                generated_data[column] = rand_geometric(prng, item['cardinality'], data_size, p=item.get('p'))
            else: 
                generated_data[column] = rand_geometric(prng, item['cardinality'], data_size)
        elif item['dist'] == "gauss":
            generated_data[column] = rand_gauss(prng, item['cardinality'], data_size)
        else:
            raise Exception("not found dist: %s" % item['dist'])

        if item.get('shuffle'):
            generated_data[column] = shuffle_attributes(prng, generated_data[column], item['cardinality'])
        
        domain[column] = item['cardinality']

    df = pd.DataFrame(generated_data)        
    return df, domain

def generate_sparse_count_table(schema, seed, data_size=0, sparse_rate=0, p=0.0001):
    prng = np.random.RandomState(seed)
    domain = { column: item['cardinality'] for column, item in schema.items() }
    sparse_size = int(domain_num(domain) * sparse_rate)
    x1 = prng.geometric(p, data_size-sparse_size)
    counts, _ = np.histogram(x1, range=(x1.min(), x1.max()), bins=sparse_size)
    counts = counts + 1
#     index= random_indices(prng, domain, sparse_size)
    index = continuous_indices(domain, sparse_size)
    return pd.DataFrame(counts, index=index), domain


def random_indices(prng, domain, size):
    unique_indices = set()
    index_size = 0
    while index_size < size:
        index = tuple([ prng.randint(0, cardinality) for cardinality in domain.values() ])
        if index in unique_indices:
            continue
        else:
            unique_indices.add(index)
            index_size += 1
    return pd.Index(list(unique_indices))


def continuous_indices(domain, size):
    indices = []
    for i in range(size):
        int_indice = [ int(j) for j in list(str(i).zfill(10))]
        indices.append(tuple(int_indice))
    return  pd.Index(indices)


def sparse_rate(df, domain):
    return len(df.drop_duplicates()) / np.prod(list(domain.values()), dtype=np.float64)


def domain_num(domain):
    return np.prod(list(domain.values()), dtype=np.float64)


def save(config):
    df, domain = generate_artificial(schema=config['schema'], seed=config['seed'], data_size=config.get('data_size'))
    write_data(config, df, domain)
    print(domain)
    print('domain size: ', domain_num(domain))
    print('data size: ', len(df))
    print('sparse rate: ', sparse_rate(df, domain))
    

def save_sparse_data(config):
    df, domain = generate_sparse_count_table(schema=config['schema'], seed=config['seed'], data_size=config.get('data_size'), sparse_rate=config.get('sparse_rate'))
    write_data(config, df, domain, True)
    print(domain)
    print('domain size: ', domain_num(domain))
    print('data size: ', df.values.sum())
    print('df size', len(df))
    return df, domain


if __name__ == '__main__':
    config = {
        'dataset_name': 'geometric-1e2',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e3',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e4',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e5',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e6',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e7',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e8',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e9',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e10',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e12',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},
            '10': { 'cardinality': 10, 'dist': 'geometric'},
            '11': { 'cardinality': 10, 'dist': 'geometric'},           
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e11',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},
            '10': { 'cardinality': 10, 'dist': 'geometric'},          
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e14',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},
            '10': { 'cardinality': 10, 'dist': 'geometric'},
            '11': { 'cardinality': 10, 'dist': 'geometric'},
            '12': { 'cardinality': 10, 'dist': 'geometric'},
            '13': { 'cardinality': 10, 'dist': 'geometric'},             
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e15',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},
            '10': { 'cardinality': 10, 'dist': 'geometric'},
            '11': { 'cardinality': 10, 'dist': 'geometric'},
            '12': { 'cardinality': 10, 'dist': 'geometric'},
            '13': { 'cardinality': 10, 'dist': 'geometric'},
            '14': { 'cardinality': 10, 'dist': 'geometric'},                 
        }
    }
    save(config)

    config = {
        'dataset_name': 'geometric-1e16',
        'seed': 0,
        'data_size': 100000,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},
            '10': { 'cardinality': 10, 'dist': 'geometric'},
            '11': { 'cardinality': 10, 'dist': 'geometric'},
            '12': { 'cardinality': 10, 'dist': 'geometric'},
            '13': { 'cardinality': 10, 'dist': 'geometric'},
            '14': { 'cardinality': 10, 'dist': 'geometric'},      
            '15': { 'cardinality': 10, 'dist': 'geometric'},              
        }
    }
    save(config)





    config = {
        'dataset_name': 'sparse-ct-1e-5',
        'seed': 0,
        'data_size': 1000000,
        'sparse_rate':1e-5,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},        
        }
    }
    save_sparse_data(config)

    config = {
        'dataset_name': 'sparse-ct-5e-6',
        'seed': 0,
        'data_size': 1000000,
        'sparse_rate':5e-6,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},        
        }
    }
    save_sparse_data(config)

    config = {
        'dataset_name': 'sparse-ct-1e-7',
        'seed': 0,
        'data_size': 1000000,
        'sparse_rate':1e-7,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},                
        }
    }
    save_sparse_data(config)

    config = {
        'dataset_name': 'sparse-ct-1e-8',
        'seed': 0,
        'data_size': 1000000,
        'sparse_rate':1e-8,
        'schema': {
            '0': { 'cardinality': 10, 'dist': 'geometric'},
            '1': { 'cardinality': 10, 'dist': 'geometric'},
            '2': { 'cardinality': 10, 'dist': 'geometric'},
            '3': { 'cardinality': 10, 'dist': 'geometric'},
            '4': { 'cardinality': 10, 'dist': 'geometric'},
            '5': { 'cardinality': 10, 'dist': 'geometric'},
            '6': { 'cardinality': 10, 'dist': 'geometric'},
            '7': { 'cardinality': 10, 'dist': 'geometric'},
            '8': { 'cardinality': 10, 'dist': 'geometric'},
            '9': { 'cardinality': 10, 'dist': 'geometric'},                
        }
    }
    df, domain = save_sparse_data(config)