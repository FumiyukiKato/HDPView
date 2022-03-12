import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
import pickle
import json
import argparse

from dataset import Dataset
from count_table import CountTable
from algorithms import privtree

parser = argparse.ArgumentParser(description='Execute PirvTree and save noised blocks')

parser.add_argument('--dataset', type=str, help="used dataset [adult, small-adult, nume-adult, trafic, bitcoin, electricity, phoneme, jm, adding-adult, all] (default: adult)", default="adult")
parser.add_argument('--epsilon', type=float, help="privacy budget (default: 1.0)", default=1.0)
parser.add_argument('--times', type=int, help="number of try (default: 10)", default=10)
parser.add_argument('--time_measure_only', action='store_true', default=False)
args = parser.parse_args()

root_dir = Path('__file__').resolve().parent
data_dir = root_dir / "data" / "preprocessed" / args.dataset
p_view_privtree_dir = root_dir / "save" / "p_view" / "privtree" / args.dataset
p_view_privtree_dir.mkdir(parents=True, exist_ok=True)
result_dir = root_dir / "exp" / "result" / "time" / "privtree" / args.dataset
result_dir.mkdir(parents=True, exist_ok=True)

theta = 0
# theta = 20
times = args.times
epsilon = args.epsilon

if __name__ == '__main__':
    dataset = Dataset.load(data_dir / 'data.csv', data_dir / 'domain.json')
    initial_block = CountTable.from_dataset(dataset)
    initial_block.info()

    fanout = 2**(len(dataset.domain))
    
    print('epsilon: ', epsilon)

    time_list = []
    for i in tqdm(range(times)):
        time_result = {}
        prng = np.random.RandomState(i)
        start = time.time()        
        p_view, block_result_list = privtree.run(initial_block, prng, epsilon=epsilon, fanout=fanout, theta=theta)
        if not args.time_measure_only:
            with open(p_view_privtree_dir / f'p_view_eps_{epsilon}_{i}.pickle', 'wb') as f:
                pickle.dump((p_view, block_result_list), f)
        time_result['execution_time'] = time.time() - start
        time_list.append(time_result)

    with open(result_dir / f'privtree_eps_{epsilon}.json', 'w') as f:
        json.dump({"time": time_list}, f)

    print('done')