import os
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch_scatter import scatter_add
from torchtools.callbacks import EarlyStopping
import argparse
import matplotlib.pyplot as plt
from Utilities.utils import str2bool
from IPython.display import clear_output 

parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')
# Study Case
parser.add_argument('--train', default=False, type=str2bool, help='train or test')
parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')

# Dataset Parameters
parser.add_argument('--dset_dir', default='Data_pt', type=str, help='dataset directory')

# Net Parameters
parser.add_argument('--n_hidden', default=3, type=int, help='number of hidden layers per MLP')
parser.add_argument('--dim_hidden', default=10, type=int, help='dimension of hidden units')
parser.add_argument('--passes', default=3, type=int, help='number of message passing')

# Training Parameters
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lambda_d', default=1e2, type=float, help='data loss weight')
parser.add_argument('--noise_var', default=1e-2, type=float, help='training noise variance')
parser.add_argument('--batch_size', default=None, type=int, help='training batch size')
parser.add_argument('--max_epoch', default=1000, type=int, help='maximum training iterations')
parser.add_argument('--miles', default=[500, 1000, 1500], nargs='+', type=int, help='learning rate scheduler milestones')
parser.add_argument('--gamma', default=1e-1, type=float, help='learning rate milestone decay')

# Save and plot options
parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
parser.add_argument('--plot_sim', default=True, type=str2bool, help='plot test simulation')

args = parser.parse_args()

class NormalizeOutput:
    def __call__(self, tensor):
        # Calcular la media y la desviación estándar del tensor
        mean_val = 0
        std_val = tensor.std()
        
        # Normalizar el tensor usando la media y la desviación estándar
        epsilon = 1e-10
        tensor = (tensor - mean_val) / (std_val + epsilon)
        
        return tensor
normalize_output = NormalizeOutput()
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class MiDataset(Dataset):
    def __init__(self, sims, dset_dir):
        self.sims = sims
        self.dset_dir = dset_dir
        self.dims = {'n_in': 3, 'out_disp': 6}

    def __getitem__(self, index):
        load = self.sims[index]
        name = os.path.join(self.dset_dir, f'Data_{load}.pt')
        data_ = torch.load(name)
        edge_index_ = data_[f'tensor_nodos_ne_{load}'].long()
        valid_indices = torch.where(edge_index_ != -1)
        src = (valid_indices[0]).to(torch.int64)
        neighbors = (edge_index_[valid_indices]-1).to(torch.int64)
        edge_index = torch.stack((src, neighbors), dim=0)
        input_data = data_[f'tensor_nodos_in_{load}'].float()
        output_data = data_[f'tensor_nodos_out_{load}'].float()
        output_data = normalize_output(output_data)
        return input_data, edge_index, output_data

    def __len__(self):
        return len(self.sims)

def split_dataset(load):
    indices = list(itertools.product(load))
    np.random.shuffle(indices)
    N_sims = len(indices)
    train_sims = indices[:int(0.8*N_sims)]
    val_sims = indices[int(0.8*N_sims):int(0.9*N_sims)]
    test_sims = indices[int(0.9*N_sims):]

    return train_sims, val_sims, test_sims

def load_dataset(args):
    load = list(range(1, 101))
    train_sims, val_sims, test_sims = split_dataset(load)
    train_set = MiDataset(train_sims, args.dset_dir)
    val_set = MiDataset(val_sims, args.dset_dir)
    test_set = MiDataset(test_sims, args.dset_dir)
    return train_set, val_set, test_set

train_dataset_nodos, val_dataset_nodos, test_dataset_nodos = load_dataset(args)

class MLP(nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            self.layers.append(nn.Linear(layer_vec[k], layer_vec[k + 1]))
            if k != (len(layer_vec) - 2): 
                self.layers.append(nn.SiLU())
                self.layers.append(nn.BatchNorm1d(layer_vec[k + 1]))
                self.layers.append(nn.Dropout(p=0.1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EdgeModel(nn.Module):
    def __init__(self, args):
        super(EdgeModel, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.edge_mlp = MLP([3*self.dim_hidden] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_atrr, batch=None):
        out = torch.cat([src, dest, edge_atrr], dim=1)
        out = self.edge_mlp(out)
        return out

class NodeModel(nn.Module):
    def __init__(self, args):
        super(NodeModel, self).__init__()
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        self.node_mlp = MLP([2*self.dim_hidden] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, x, edge_index, edge_attr, batch=None):
        src, dest = edge_index
        out = scatter_add(edge_attr, src, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp(out)
        return out

class MetaLayer(nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, batch=None):
        src = edge_index[0]
        dest = edge_index[1]
        edge_attr = self.edge_model(x[src], x[dest], edge_attr, batch if batch is None else batch[src])
        x = self.node_model(x, edge_index, edge_attr, batch)
        return x, edge_attr

class GNNn(nn.Module):
    def __init__(self, args):
        super(GNNn, self).__init__()
        self.passes = args.passes
        self.n_hidden = args.n_hidden
        self.dim_hidden = args.dim_hidden
        dim_in = 3
        dim_out = 6
        # Encoder
        self.encoder_node_n = MLP([dim_in] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])
        # Processor 
        self.processor = nn.ModuleList()
        for _ in range(self.passes):
            node_model = NodeModel(args)
            edge_model = EdgeModel(args)
            GraphNet = MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)
        # Decoder
        self.decoder_disp = MLP([self.dim_hidden] + self.n_hidden * [self.dim_hidden] + [dim_out])

    def forward(self, n_in, edge_index, batch):
        # Encode
        x = self.encoder_node_n(n_in)
        edge_attr = torch.zeros((edge_index.size(1), self.dim_hidden), device=edge_index.device)
        # Process
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, batch=batch)
            x += x_res
            edge_attr += edge_attr_res
        # Decode
        disp = self.decoder_disp(x)
        return disp

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, edge_index, targets = data
        inputs, edge_index, targets = inputs.to(device), edge_index.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, edge_index, None)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def val(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, edge_index, targets = data
            inputs, edge_index, targets = inputs.to(device), edge_index.to(device), targets.to(device)
            outputs = model(inputs, edge_index, None)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(val_loader)

def solve(model, train_set, val_set, test_set, args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    criterion = torch.nn.MSELoss()
    #criterion = RMSLELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    train_losses, val_losses = [], []
    #plt.ion()
    #fig, ax = plt.subplots(figsize=(10, 5))
    for epoch in range(1, args.max_epoch + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = val(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
#         clear_output(wait=True)
    
#         # Limpiar los ejes
#         ax.clear()
    
#     # Graficar las pérdidas
#         ax.plot(train_losses, label='Train Loss')
#         ax.plot(val_losses, label='Test Loss')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Loss')
#         ax.set_title('Training and Test Loss')
#         ax.legend()
    
#     # Pausar para permitir que la gráfica se actualice
#         plt.pause(0.01)
        print(f"Epoch {epoch}/{args.max_epoch}, Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# # Desactivar el modo interactivo al final del bucle
#     plt.ioff()
#     plt.show()

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, edge_index, targets = data
            inputs, edge_index, targets = inputs.to(device), edge_index.to(device), targets.to(device)
            outputs = model(inputs, edge_index, None)
            print(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    print(f"Test Loss: {running_loss / len(test_loader)}")

model = GNNn(args)
solve(model, train_dataset_nodos, val_dataset_nodos, test_dataset_nodos, args)