import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path
import re
import sys
import pickle
import math
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help="adult, small-adult, bitcoin, electricity, bank-marketing, phoneme (default: adult)", default="adult")
args = parser.parse_args()

def get_dir_path(dataset):
    root_dir = Path('__file__').resolve().parent
    p_view_hdpview_dir = root_dir / "save" / "p_view" / "hdpview" / dataset
    return p_view_hdpview_dir

def original_noised_data_size(domain_size):
    return  4*domain_size + 96

def byte_size(noised_data):
    cd_size = sys.getsizeof(noised_data.cardinality_dict)
    ind_size = sys.getsizeof(noised_data.index)
    dd_size = sys.getsizeof(noised_data.domain_dict)
    bl_size = sys.getsizeof(noised_data.blocks)
    os_size = sys.getsizeof(noised_data.original_shape)
    noised_data_size = cd_size + ind_size + dd_size + bl_size + os_size

    block = noised_data.blocks[0]
    bs_size = sys.getsizeof(block.size)
    bv_size = sys.getsizeof(block.value)
    br_size = sys.getsizeof(block.ranges)
    blocks_size = len(noised_data.blocks) * (bs_size + bv_size + br_size)

    total_size = noised_data_size + blocks_size
    return total_size

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


if __name__ == '__main__':
    dataset = args.dataset
    p_view_hdpview_dir = get_dir_path(dataset)
    epsilon_list = [1.0]

    size_table = {}
    for epsilon in epsilon_list:
        size_table[epsilon] = []
        files = p_view_hdpview_dir.glob(f'mondrian_eps_{epsilon}*')
        for file in files:
            with open(file, 'rb') as f:
                noised_data, block_and_noise = pickle.load(f)
            size_table[epsilon].append(convert_size(byte_size(noised_data)))
            print(file.name)
            print(convert_size(byte_size(noised_data)))
    with open(p_view_hdpview_dir / f'size-list.json', 'w') as f:
        json.dump(size_table,f)