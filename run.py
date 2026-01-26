# MIT License
#
# Original work:
# Copyright (c) <2019> <Zonghan Wu>
#
# Modifications:
# Copyright (c) 2026 D-Stiv
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# ----
# Modifications points: 
# - integreted STNorm in gwnet: lines 168; 187; 252
# - inserted correlation and neighborhood feature enrichment: lines 604-625


import torch
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import linalg
import pickle
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import shutil
import pandas as pd
import configargparse
import yaml
import copy
from datetime import timedelta
import random



metr_la = 'metr-la'
pems_bay = "pems-bay"


def set_seed(seed=42, loader=None):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    try:
        loader.sampler.generatoor.manual_seed(seed)
    except AttributeError:
        pass
    
    
def get_root():
    root = 'SCANNER/'
    while not os.path.exists(f"{root}"):
        root = f'../{root}'
    return root


class TemporalNorm(nn.Module):
    """
    Temporal Normalization (TN)
    Normalizes over the temporal dimension (T)
    """

    def __init__(self, num_nodes, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        # Learnable affine parameters (per node, per feature)
        self.gamma = nn.Parameter(torch.ones(1, num_features, num_nodes, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_features, num_nodes, 1))

    def forward(self, x):
        """
        x: (B, C, N, T)
        """
        # Mean & variance over time
        mean = x.mean(dim=3, keepdim=True)        # (B, C, N, 1)
        var  = x.var(dim=3, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return x_norm * self.gamma + self.beta
    

class SpatialNorm(nn.Module):
    """
    Spatial Normalization (SN)
    Normalizes over the spatial dimension (N)
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        # Shared affine parameters (per feature)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        """
        x: (B, C, N, T)
        """
        # Mean & variance over nodes
        mean = x.mean(dim=2, keepdim=True)        # (B, C, 1, T)
        var  = x.var(dim=2, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return x_norm * self.gamma + self.beta


class STNorm(nn.Module):
    def __init__(self, num_nodes, num_features):
        super().__init__()
        self.tn = TemporalNorm(num_nodes, num_features)
        self.sn = SpatialNorm(num_features)

    def forward(self, x):
        """
        x: (B, C, N, T)
        """
        x_tn = self.tn(x)
        x_sn = self.sn(x)

        # Concatenate along feature dimension
        return [x_tn, x_sn]


class gwnet(nn.Module):
    def __init__(self, device, args, num_nodes, adjinit, in_dim, supports, output_horizon, nhid, dropout, adaptadj, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        out_dim = output_horizon
        residual_channels = nhid
        dilation_channels = nhid
        skip_channels = nhid * 8
        end_channels = nhid * 16
        aptinit = adjinit
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.adaptadj = args.adaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.stnorm = nn.ModuleList()
        num = 3

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.stnorm.append(STNorm(num_nodes, residual_channels))

                self.filter_convs.append(nn.Conv2d(in_channels=num * residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=num * residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):        
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            x_list = []
            x_list.append(x)
            
            # STNorm
            x_list = x_list + self.stnorm[i](x)
            
            # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, skip


class trainer():
    def __init__(self, scaler, num_nodes, adjinit, in_dim, supports):
        kernel_size = int((args.input_horizon-1)/12) + 2
        self.model = gwnet(num_nodes, adjinit, in_dim, supports, kernel_size=kernel_size)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):  # added categories
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, embs = self.model(input)
        output = output.transpose(1, 3)

        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val, params=None):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, _ = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real)
        mape = masked_mape(predict, real).item()
        rmse = masked_rmse(predict, real).item()
        return loss.item(), mape, rmse

# data loader class to generate batches
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, cs=None): 
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        pass

    def shuffle(self, seed=None):
        pass

    def get_iterator(self):
        pass

# standard scaler class to standardise the data
class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_metrics(dataloader, engine, scaler, phase):
    outputs = []
    realy = torch.Tensor(dataloader[f'y_{phase}']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for _, (x, _) in enumerate(dataloader[f'{phase}_loader'].get_iterator()):
        # phasex = torch.Tensor(x).to(device)  # phasex to say valx or testx
        phasex = torch.Tensor(x).to(device)
        phasex = phasex.transpose(1, 3)
        with torch.no_grad():
            preds, _ = engine.model(phasex)
            preds = preds.transpose(1, 3)           
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(output_horizon):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        if phase == "test":
            log = '{} - dataset: {}. Best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(args.model_name, dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    return amae, amape, armse


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# load adjacency matrix and calculate the scaled laplacian matrix
def load_adj(pkl_filename, adjtype):
    pass


def masked_mse(preds, labels, null_val=np.nan, params=None):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, params=None):
    return torch.sqrt(masked_mse(preds, labels, null_val=null_val, params=params))


def masked_mae(preds, labels, null_val=np.nan, params=None):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def get_correlation_matrix(dataset_name=pems_bay):
    sensor_dir = f"{get_root()}sensor_graph/{dataset_name}"
    if dataset_name in [pems_bay, metr_la]:
        adj_path = f"{sensor_dir}/adj_mx.pkl"
        with open(adj_path, "rb") as f:
            _, _, adj = pickle.load(f)
        num_nodes = adj.shape[0]
        num_horizons = 12
        trans_matrices = np.tile(adj, (num_horizons,1)).reshape(num_horizons, num_nodes, num_nodes)
        return trans_matrices
    prop_path = f"{sensor_dir}/propagation.pkl"
    with open(prop_path, "rb") as f:
        trans_matrices = pickle.load(f)
    return trans_matrices


# load and process dataset
def load_dataset(dataset_name='pems-bay', input_horizon=12, output_horizon=12, train_fraction=.7, test_fraction=.2):
    print("Getting the data ...")
    data_ = {}
    root = get_root()
    dir_path = f'{root}{output_dir_}/{dataset_name.lower()}/split_{train_fraction}_{test_fraction}_hor_{input_horizon}_{output_horizon}'
    if os.path.exists(dir_path):
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(dir_path, category + '.npz'))
            data_['x_' + category] = cat_data['x']
            data_['y_' + category] = cat_data['y']
        scaler = StandardScaler(mean=data_['x_train'][..., 0].mean(), std=data_['x_train'][..., 0].std())
        
        # data format
        for category in ['train', 'val', 'test']:
            data_['x_' + category][..., 0] = scaler.transform(data_['x_' + category][..., 0])
        data_['train_loader'] = DataLoader(data_['x_train'], data_['y_train'], batch_size)
        data_['val_loader'] = DataLoader(data_['x_val'], data_['y_val'], batch_size)
        data_['test_loader'] = DataLoader(data_['x_test'], data_['y_test'], batch_size)
        data_['scaler'] = scaler
        return data_
    
    print(f"Generating the new data for {dataset_name} ...")    
        
    if dataset_name in ['pems-bay', 'metr-la']:
        path = f'{root}raw_data_metadata/{dataset_name}.h5'
        data = pd.read_hdf(path)
    else:
        raise Exception(f"Dataset '{dataset_name}' is unknown")

    print(df.shape)
    df = df.sort_index(axis=1).reset_index()
    index = df.columns[0]
    df = df[(df[index].dt.hour >= start_hour)
            & (df[index].dt.hour <= end_hour)]
    
    speed_df = pd.melt(df, id_vars=df.columns[0], value_vars=df.columns[1:])
    date_time_col, sensors_col, speed_col = speed_df.columns[0], speed_df.columns[1], speed_df.columns[2]
    speed_df[sensors_col] = speed_df[sensors_col].astype('int')

    # compute dow, hour, minute, interval
    speed_df['dow'] = speed_df[date_time_col].dt.day_of_week
    speed_df['hour'] = speed_df[date_time_col].dt.hour
    speed_df['minute'] = speed_df[date_time_col].dt.minute
    rate = 5    # minutes
    speed_df['interval'] = speed_df[date_time_col].dt.minute / rate
    speed_df['interval'] = speed_df['interval'].astype(int)
    speed_df['tod_norm'] = (speed_df[date_time_col].dt.hour*60 + speed_df[date_time_col].dt.minute) / (60*24)    # time of the day
    speed_df['dow_norm'] = (speed_df[date_time_col].dt.day_of_week + 1) / 7

    # constrcuction of data split with speed and avg_speed
    x_offsets = np.arange(-input_horizon + 1, 1, 1)
    y_offsets = np.arange(1, output_horizon + 1, 1)
    num_samples = df.shape[0]
    data = []
    speed = pd.pivot_table(speed_df, values=speed_col, index=date_time_col, columns=[sensors_col], aggfunc=np.mean)
    data.append(speed) 
    if args.add_tod:   
        time_of_day = pd.pivot_table(speed_df, values='tod_norm', index=date_time_col, columns=[sensors_col], aggfunc=np.mean)
        data.append(time_of_day)
    if args.add_dow:
        day_of_week = pd.pivot_table(speed_df, values='dow_norm', index=date_time_col, columns=[sensors_col], aggfunc=np.mean)
        data.append(day_of_week)

    data = np.stack(data, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)  # x: (num_samples, input_horizon, num_nodes, input_features)
    y = np.stack(y, axis=0)  # y: (num_samples, output_horizon, num_nodes, output_features)    
    
    if args.use_neighbors and args.num_neighbors > 0:
        num_samples, _, num_nodes, in_feats = x.shape
        args.input_horizon = args.input_horizon + args.num_neighbors*args.neighbors_horizons
        A = get_correlation_matrix(dataset_name)[:, :num_nodes, :num_nodes]
   
        x_neighbors = []

        for h in range(min(args.neighbors_horizons, A.shape[0])):
            np.fill_diagonal(A[h], 0.)
            x_neighbors_h = []
            for j in range(num_nodes):
                topK_j = np.flip(np.argsort(A[h, j]))[:args.num_neighbors]
                x_neighbors_h_j = x[:, h, topK_j]
                x_neighbors_h.append(x_neighbors_h_j)
            x_neighbors_h = np.stack(x_neighbors_h, axis=-2)
                    
            x_neighbors.append(x_neighbors_h)

        x_neighbors = np.stack(x_neighbors, axis=1)
        x_neighbors = x_neighbors.reshape(num_samples, args.num_neighbors*args.neighbors_horizons, num_nodes, in_feats)
                
        x = np.concatenate([x, x_neighbors], axis=1)

    num_samples = x.shape[0]
    num_test = round(num_samples * test_fraction)
    num_train = round(num_samples * train_fraction)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(dir_path, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
        print(f'saved {cat} data at: {dir_path}/{cat}.npz')
        data_['x_' + cat], data_['y_' + cat] = _x, _y
    scaler = StandardScaler(mean=data_['x_train'][..., 0].mean(), std=data_['x_train'][..., 0].std())
    
    # data format
    for category in ['train', 'val', 'test']:
        data_['x_' + category][..., 0] = scaler.transform(data_['x_' + category][..., 0])
    data_['train_loader'] = DataLoader(data_['x_train'], data_['y_train'], batch_size)
    data_['val_loader'] = DataLoader(data_['x_val'], data_['y_val'], batch_size)
    data_['test_loader'] = DataLoader(data_['x_test'], data_['y_test'], batch_size)
    data_['scaler'] = scaler
    
    return data_

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
    
    config._parser.add("-sn", "--snorm", help="spatial normalisation", default=0, type=int, nargs='*')
    config._parser.add("-tn", "--tnorm", help="temporal normalistation", default=0, type=int, nargs='*')
    config._parser.add("-oH", "--output_horizon", help="number of steps to predict", default=12, type=int, nargs='*')
    config._parser.add("-iH", "--input_horizon", help="number of historical steps", default=12, type=int, nargs='*')
    config._parser.add("-tnF", "--train_fraction", help="fraction training samples", default=.7, type=float, nargs='*')
    config._parser.add("-dN", "--dataset_name", help="Name of the dataset", default='pems-bay', nargs='*')
    config._parser.add("-mod", "--model_name", help="Name of the method", default='SCANNER', nargs='*')
    config._parser.add("-ttF", "--test_fraction", help="fraction test samples", default=.2, type=float, nargs='*') 
    config._parser.add("-ep", "--epochs", help="number of epochs", default=5, type=int, nargs='*')
    config._parser.add("-gcn", "--gcn_bool", help="whether to add graph convolution layer", default=1, type=int, nargs='*')
    config._parser.add("-ado", "--aptonly", help="whether only adaptive adj", default=1, type=int, nargs='*')
    config._parser.add("-adp", "--adaptadj", help="whether add adaptive adj", default=1, type=int, nargs='*')
    config._parser.add("-sd", "--seed", help="random seed for all modules", default=42, type=int, nargs='*')
    config._parser.add("-exp", "--expid", help="experiment id", default=-1, type=int, nargs='*')
    config._parser.add("-bS", "--batch_size", help="batch size for train, val and test", default=64, type=int, nargs='*')
    config._parser.add("-aT", "--add_tod", help="add time of the day in list of input features", default=0, type=int, nargs='*')
    config._parser.add("-aD", "--add_dow", help="add day of weeks in list of input features", default=0, type=int, nargs='*')
    config._parser.add("-uN", "--use_neighbors", help="use neighbors information for the prediction", default=0, type=int, nargs='*')
    config._parser.add("-nN", "--num_neighbors", help="number of neighnors to consider for each node", default=1, type=int, nargs='*')
    config._parser.add("-nH", "--neighbors_horizons", help="number of past horizons of each neighbor", default=2, type=int, nargs='*')

    config._parser.add("-stH", "--start_hour", help="starting hour to filter the dataset", default=0, type=int, nargs='*')
    config._parser.add("-edH", "--end_hour", help="ending hour to filter the dataset", default=23, type=int, nargs='*')

    config._parser.add('-pt', '--patience', type=int, default=15, help='patience for early stop', nargs='*')

    config.parse()


def compute(args):
    # define run specific parameters based on user input
    root = get_root()
    save_dir = f'{root}{save}/{model_name}/{dataset_name.lower()}/'
    if dataset_name in ["pems-bay", "metr-la"]:
        adjdata_file = f'{root}{adjdata}/{dataset_name.lower()}/adj_mx.pkl'
    else:
        raise Exception(f"Dataset '{dataset_name}' is unknown")
    
    _, _, adj_mx = load_adj(adjdata_file, adjtype)
    dataloader = load_dataset(dataset_name, args.input_horizon, output_horizon, train_fraction, test_fraction)
    scaler = dataloader['scaler']
    supports = None if adj_mx is None else [torch.tensor(i).to(device) for i in adj_mx]

    print('Configuration done properly ...')

    if adaptadj or supports is None:
        adjinit = None
    else:
        adjinit = supports[0]

    if aptonly:
        supports = None

    # Get Shape of Data
    _, _, num_nodes, in_dim = dataloader['x_train'].shape  # _ is seq_length
    in_dim = in_dim - 1

    engine = trainer(scaler=scaler, num_nodes=num_nodes, adjinit=adjinit, in_dim=in_dim, supports=supports)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('saving directory at ' + save_dir)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    curr_min = np.inf
    wait = 0
    for i in range(1, epochs + 1):
        try:
            if wait >= args.patience:
                print(log, f'early stop at epoch: {i:04d}')
                break
            
            # training
            phase = "train"

            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()

            t_ = time.time()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)   
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)

                metrics = engine.train(trainx, trainy[:, 0, :, :])  # 0: speed

                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % print_every == 0:
                    log = '{} - dataset: {}. Iter: {:03d}, Train Loss: {:.4f}, Time: {:.2f} secs'
                    print(log.format(model_name, dataset_name, iter, train_loss[-1], (time.time()-t_)),
                        flush=True)
                    t_ = time.time()

            t2 = time.time()
            train_time.append(t2 - t1)

            mtrain_loss = np.mean(train_loss)

            # validation
            phase = "val"

            s1 = time.time()
            valid_loss, valid_mape, valid_rmse = get_metrics(dataloader=dataloader, engine=engine, scaler=scaler, phase=phase)
            s2 = time.time()
            log = '{} - dataset: {}. Epoch: {:03d}/{:03d}, Inference Time: {:.4f} secs'
            print(log.format(model_name, dataset_name, i, epochs,(s2 - s1)))
            val_time.append(s2 - s1)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)

            his_loss.append(mvalid_loss)

            log = '{} - dataset: {}. Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val MAPE: {:.4f}, Val RMSE: {:.4f}, Training Time: {:.1f} secs/epoch. Best epoch: {}, best val loss: {:.4f}'
            print(log.format(model_name, dataset_name, i, mtrain_loss, mvalid_loss, mvalid_mape,
                            mvalid_rmse, (t2 - t1), np.argmin(his_loss), np.min(his_loss)), flush=True)
            
            # save only if less than the minimum
            if mvalid_loss == np.min(his_loss):
                wait = 0
                print(f'{model_name} - dataset: {dataset_name}. Epoch: {i:03d} - Val_loss decreases from {curr_min:.4f} to {mvalid_loss:.4f}')
                curr_min = mvalid_loss
                torch.save(engine.model.state_dict(), save_dir + "_exp" + str(expid) + "_best_" + ".pth")
            elif mvalid_loss - np.min(his_loss) < 1e-2: # we are still in the elbow zone
                pass
            else:
                wait += 1
        except:
            if i > 1:
                break
            else:
                raise Exception
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # testing
    phase = "test"
    engine.model.load_state_dict(torch.load(save_dir + "_exp" + str(expid) + "_best_" + ".pth"))

    amae, amape, armse = get_metrics(dataloader=dataloader,engine=engine, scaler=scaler, phase=phase)
    
    mean_amae = np.mean(amae)
    mean_amape = np.mean(amape)
    mean_armse = np.mean(armse)
    log = '{} - dataset: {}. On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(model_name, dataset_name, output_horizon, mean_amae, mean_amape, mean_armse))

if __name__ == "__main__":
    config = Configuration()
    setup_config(config)

    i = 1
    # create one thread for each c
    num_exp = len(config.get_configs())
    print('Number of experiments: ', num_exp)
    for args in config.get_configs():
        set_seed(args.seed)

        print(f'Starting experiment number {i}/{num_exp} ...')
    
        model_name = args.model_name
        dataset_name = args.dataset_name
        save = 'pretrained_model'
        output_dir_ = 'traffic_data'
        adjdata = 'sensor_graph'
        batch_size = args.batch_size
        output_horizon = args.output_horizon
        weight_decay = 0.0001
        dropout = 0.3
        learning_rate = 0.001
        nhid = 32
        epochs = args.epochs
        print_every = 100
        expid = args.expid
        train_fraction = args.train_fraction
        test_fraction = args.test_fraction
        snorm_bool = args.snorm
        tnorm_bool = args.tnorm
        start_hour = args.start_hour 
        end_hour = args.end_hour 
        device = "cuda"
        adjtype = 'doubletransition'
        gcn_bool = args.gcn_bool # True
        aptonly = args.aptonly
        adaptadj = args.adaptadj # True
        device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        print(args)   
        compute(args)
        
        i += 1
