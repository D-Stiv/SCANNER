# MIT License
#
# Copyright (c) 2026 D-Stiv
#
# See the LICENSE file in the repository root for full license text.


# %%
import os
import pickle
import numpy as np
import pandas as pd
import configargparse
import copy

# %%
# %%
metr_la = 'metr-la'
pems_bay = "pems-bay"

def get_root():
    root = 'SCANNER/'
    while not os.path.exists(f"{root}"):
        root = f'../{root}'
    return root

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  New Folder Created: ", path)
        

# %%
def raw_data(dataset_name):
    if dataset_name in ['pems-bay', 'metr-la']:
        path = f'{get_root()}raw_data_metadata/{dataset_name}.h5'
        data = pd.read_hdf(path)
    else:
        raise Exception(f"Dataset '{dataset_name}' is unknown")
    # Use linear interpolation to fill up nulls
    ## Remove time column before interpolation
    df = data.interpolate(method='linear', axis=0).ffill().bfill()
    df = df.sort_index(axis=1).reset_index()
    index = df.columns[0]
    df = df[(df[index].dt.hour >= start_hour)
            & (df[index].dt.hour <= end_hour)]
    df = df.set_index(index)
    print(df.shape)
    return df

# %%
def pearson_correlation(x, y):
    x_bar, y_bar = x.mean(), y.mean()
    eps = 1e-8
    dx = x - x_bar
    dy = y - y_bar
    # corr = (torch.dot(dx,dy))/torch.sqrt(torch.dot(dx,dx)*torch.dot(dy,dy) + eps)
    corr = (np.dot(dx,dy))/np.sqrt(np.dot(dx,dx)*np.dot(dy,dy) + eps)
    return corr

# %%
def get_correlation_matrix(x_train, output_horizon=12, dataset_name=pems_bay):
    sensor_dir = f"{get_root()}sensor_graph/{dataset_name}"
    if dataset_name in [pems_bay, metr_la]:
        _path = f"{sensor_dir}/propagation.pkl"
    else:
        raise Exception(f"Dataset '{dataset_name}' is unknown")
    
    if os.path.exists(_path):
        with open(_path, "rb") as f:
            trans_matrices = pickle.load(f)
        return trans_matrices
    mkdir(sensor_dir)
    
    trans_matrices = []
    num_samples, num_nodes = x_train.shape

    feat_len = num_samples - output_horizon
    for t in range(output_horizon):
        print(f"{dataset_name}: Horizon {t+1:03d}/{output_horizon:02d}")
        trans_matrix = np.zeros((num_nodes, num_nodes))
        for node_1 in range(num_nodes):
            for node_2 in range(num_nodes):
                trans_matrix[node_1, node_2] = pearson_correlation(x_train[:feat_len, node_1], x_train[t:t+feat_len, node_2])
        trans_matrices.append(trans_matrix)        
        
    trans_matrices = np.stack(trans_matrices, axis=0)
    
    with open(_path, "wb") as f:
        pickle.dump(trans_matrices, f)
        print(f"correlation matrix for {dataset_name} saved in {_path}")
    
    return trans_matrices

class Configuration:
    def __init__(self):
        self._parser = configargparse.ArgumentParser()
        
        # config parsed by the default parser
        self._config = None

        # individual configurations for different runs
        self._configs = []
        
        # arguments with more than one value
        self._multivalue_args = []
        
        
        
    def parse(self):
        self._config = self._parser.parse_args()
    
        # find values with more than one entry
        dict_config = vars(self._config)
        for k in dict_config :
            if isinstance(dict_config[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._config)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_config[ma]:
                    current = copy.deepcopy(c)
                    setattr(current, ma, v)
                    new_configs.append(current)

            # store splitted values
            self._configs = new_configs
        
    def get_configs(self):
        return self._configs
    

def setup_config(config):
    print('Configuration setup ...')

    config._parser.add("-dN", "--dataset_name", help="Name of the dataset", default='hannover', nargs='*') # 'pems-bay', 'hannover', 'braunschweig', 'wolfsburg'
    config.parse()

# %%
if __name__ == "__main__":
    config = Configuration()
    setup_config(config)
    
    train_fraction = 0.7
    test_fraction = 0.2
    output_horizon = 12
    start_hour = 5
    end_hour = 22
    
    i = 1
    num_exp = len(config.get_configs())
    for args in config.get_configs():
        print(f"Test: {args.test}")
        print(f'Starting experiment number {i}/{num_exp} ...')
        
        i = i+1
        
        print("loading the data ...")
        dataset_name = args.dataset_name
        # dataset_name = hannover
        df = raw_data(dataset_name)
        num_samples, num_nodes = df.shape
        
        x_train = df.iloc[:int(train_fraction*num_samples)].values
        print("Computing correlation matrix ...")
        mat = get_correlation_matrix(x_train, output_horizon=output_horizon, dataset_name=dataset_name)
        print(mat.shape)
    
    
# %%
