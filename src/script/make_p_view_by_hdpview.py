import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
import pickle
import json
import argparse

from dataset import Dataset
from count_table import CountTable
import hdpview

parser = argparse.ArgumentParser(description='Execute HDPView and save generated p-view')

parser.add_argument('--dataset', type=str, help="used dataset [adult, small-adult, nume-adult, trafic, bitcoin, electricity, phoneme, jm, adding-adult, all] (default: adult)", default="adult")
parser.add_argument('--epsilon', type=float, help="privacy budget (default: 1.0)", default=1.0)
parser.add_argument('--times', type=int, help="number of try (default: 10)", default=10)
parser.add_argument('--time_measure_only', action='store_true', default=False)
args = parser.parse_args()

root_dir = Path('__file__').resolve().parent
given_data_dir = root_dir / "data" / "preprocessed" / args.dataset
p_view_hdpview_dir = root_dir / "save" / "p_view" / "hdpview" / args.dataset
p_view_hdpview_dir.mkdir(parents=True, exist_ok=True)
result_dir = root_dir / "exp" / "result" / "time" / "hdpview" / args.dataset
result_dir.mkdir(parents=True, exist_ok=True)

ratio = 0.9
alpha=1.6
beta=1.2
gamma=0.9

times = args.times
epsilon = args.epsilon

if __name__ == '__main__':
    dataset = Dataset.load(given_data_dir / 'data.csv', given_data_dir / 'domain.json')
    initial_block = CountTable.from_dataset(dataset)
    initial_block.info()
    
    print('epsilon: ', epsilon)

    time_list = []
    for i in tqdm(range(times)):
        time_result = {}
        prng = np.random.RandomState(i)
        start = time.time()
        p_view, block_result_list = hdpview.run(initial_block, epsilon, ratio, prng, alpha, beta, gamma)
        if not args.time_measure_only:
            with open(p_view_hdpview_dir / f'p_view_eps_{epsilon}_{i}.pickle', 'wb') as f:
                pickle.dump((p_view, block_result_list), f)
        time_result['execution_time'] = time.time() - start
        time_list.append(time_result)

    with open(result_dir / f'hdpview_eps_{epsilon}.json', 'w') as f:
        json.dump({"time": time_list}, f)

    print('done')