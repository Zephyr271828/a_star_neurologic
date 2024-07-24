import os
import pickle

os.environ['HF_DATASETS_CACHE'] = '/scratch/yx3038/cache'
os.environ['HF_HOME'] = '/scratch/yx3038/cache'

from pprint import pprint


def load_data(split):
    # split = 'train'
    input_path = f'../dataset/commongen/{split}.txt'
    with open(input_path) as fin:

        data = [
            {'input' : line.split('=')[0] + "=", 
            'label' : line.split('=')[1]} 
            for line in fin.read().splitlines()
        ]

    pprint(data[:10])
    print(len(data))

    return data

def save_data(data, split):
    file_path = f'../dataset/commongen/{split}.json'
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    split = 'train'
    data = load_data(split)
    save_data(data, split)