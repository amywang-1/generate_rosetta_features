import subprocess
#import pandas as pd
#from list_gb1_mutations import all_mutation_strings_list
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import os

def f(mutation, save_dir):
    subprocess.run(['python', 'rosetta_compute_features_GB1.py', mutation, save_dir])


def get_mutation_names(split_path):
    mutations = []
    for root, dirs, files in os.walk(split_path):
        for file in files:
            mutations.append(Path(os.path.basename(file)).stem)
    return mutations

split_path = '/2GI9_splits/two_vs_rest/'

if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        for split in ['val', 'train', 'test']:
            mutations = get_mutation_names(split_path + split)
            for sub_dir in range(0, 10):
                save_path = f'/GB1_features_with_replicates/{split}/{sub_dir}/'
                p.map(partial(f, save_dir=save_path), mutations)