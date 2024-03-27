import torch.nn as nn
import torch.nn.functional as F
import math
import csv
import torch
import GPUtil
from tqdm import tqdm
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from copy import deepcopy
from utils_disttack import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, SGConv
from torch.nn import Sequential, Linear, ReLU
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import utils_disttack
from torch_geometric.loader import NeighborSampler

class SAGE(torch.nn.Module):
    def __init__(self, data, test_data, model_type, in_channels, hidden_channels, out_channels,
                 num_layers):
        super(SAGE, self).__init__()
        self.data = data #it will be sent to device later with model
        self.test_data = test_data
        self.model_type = model_type
        self.num_layers = num_layers
        self.lr = 0.01
        self.weight_decay = 5e-4

        self.convs = torch.nn.ModuleList()

        if self.model_type == 'gcn': # default normalize=True
            self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
            for _ in range(self.num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
            self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
        elif self.model_type == 'sage': # default normalize=False
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(self.num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        elif self.model_type == 'gin':
            self.convs.append(GINConv(Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)), train_eps=True))
            for _ in range(self.num_layers - 2):
                self.convs.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)), train_eps=True))
            self.convs.append(GINConv(Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels)), train_eps=True))
        elif self.model_type == 'gat':
            self.convs.append(GATConv(in_channels, hidden_channels//8, heads=8, dropout=0.5))
            for _ in range(self.num_layers - 2):
                self.convs.append(GATConv(hidden_channels, hidden_channels//8, heads=8, dropout=0.5))
            # On the Pubmed dataset, use heads=8 in conv2.
            self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False,
                                 dropout=0.5))
        elif self.model_type == 'sgc':
            self.convs.append(SGConv(in_channels, out_channels, K=3, add_self_loops=False))
        else:
            assert False
    '''
    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index)
        return x.log_softmax(dim=-1)
    '''
    def edge_forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        #x = self.convs[0](x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        #x = self.convs[1](x, edge_index)
        return x.log_softmax(dim=-1)
    
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            if self.model_type == 'gcn':
                x = self.convs[i](x, edge_index)
            else:
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.6, training=self.training)
        return x.log_softmax(dim=-1)
    
    def adj_forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            if self.model_type == 'gcn':
                x = self.convs[i](x, edge_index)
            elif self.model_type == 'sgc':
                x = self.convs(x, edge_index)
            else:
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.65, training=self.training)
        return x.log_softmax(dim=-1)
    
    def predict(self):
        self.eval()
        return self.edge_forward(self.data.x, self.data.edge_index)
    
    @torch.no_grad()
    def inference(self, x_all, device, data):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            edge_index = data.edge_index
            x = x_all
            if self.model_type == 'gcn':
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
            xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all.log_softmax(dim=-1)
    
    @torch.no_grad()
    def adj_inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                if self.model_type == 'gcn':
                    x = self.convs[i](x, edge_index)
                else:
                    x_target = x[:size[1]]
                    x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all
         
    def initialize(self):
        """Initialize parameters of SAGE.
        """
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()    

    def train_with_early_stopping(self, device, train_iters=20, initialize=True, verbose=False, patience=50, batch_size=2048, **kwargs):
        if initialize:
            self.initialize()

        if verbose:
            print('=== training model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, min_lr=0.00001)

        train_loader = NeighborSampler(
            self.data.edge_index,
            node_idx=self.data.train_mask,
            sizes=[25,10],
            batch_size=batch_size,
            shuffle=True,
            num_workers=6
        )

        early_stopping_counter = patience
        best_loss_val = float('inf')
        for epoch in range(train_iters):
            self.train()
            total_loss = 0
            total_grad_norm = {name: 0.0 for name, param in self.named_parameters() if param.requires_grad}
            num_batches = 0

            for batch_size, n_id, adjs in train_loader:
                optimizer.zero_grad()
                x_batch = self.data.x[n_id].to(device)  
                y_batch = self.data.y[n_id[:batch_size]].squeeze().long().to(device)
                adjs = [adj.to(device) for adj in adjs]
                out = self(x_batch, adjs) 
                loss = F.nll_loss(out, y_batch)  
                loss.backward()
                optimizer.step() 
                total_loss += loss.item() 
                #for name, param in self.named_parameters():
                #    if param.requires_grad:
                #        total_grad_norm[name] += param.grad.norm().item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            print(f"\nEpoch {epoch}, Average Loss: {avg_loss:.4f}")
            for name in total_grad_norm.keys():
                avg_grad_norm = total_grad_norm[name] / num_batches
                print(f"Layer: {name} | Average Grad Norm: {avg_grad_norm:.4f}")

            # Validation
            self.eval()
            with torch.no_grad():
                out_val = self.edge_forward(self.data.x.to(device), self.data.edge_index.to(device)) 
                loss_val = F.nll_loss(out_val[self.data.val_mask].to(device), self.data.y[self.data.val_mask].to(device)) 

            scheduler.step(loss_val)

            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, training loss: {total_loss / len(train_loader)}, validation loss: {loss_val.item()}')

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                early_stopping_counter = patience
            else:
                early_stopping_counter -= 1
            if early_stopping_counter <= 0:
                if verbose:
                    print(f'Early stopping at epoch {epoch}, best validation loss: {best_loss_val.item()}')
                break

    def test(self):
        """Evaluate model performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.test_data.test_mask
        labels = self.test_data.y
        output = self.edge_forward(self.test_data.x, self.test_data.edge_index)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils_disttack.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
    
    '''
    def train_with_early_stopping(self, device, train_iters=20, initialize=True, verbose=False, patience=50, batch_size=2048, **kwargs):
        if initialize:
            self.initialize()

        if verbose:
            print('=== training model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        train_loader = NeighborSampler(
            self.data.edge_index,
            node_idx=self.data.train_mask,
            sizes=[25,10],  
            batch_size=batch_size,
            shuffle=True,
            num_workers=6
        )

        early_stopping_counter = patience
        best_loss_val = float('inf')
        for epoch in range(train_iters):
            self.train()
            total_loss = 0
            total_grad_norm = {name: 0.0 for name, param in self.named_parameters() if param.requires_grad}
            num_batches = 0

            for batch_size, n_id, adjs in train_loader:
                optimizer.zero_grad()
                x_batch = self.data.x[n_id].to(device)  
                y_batch = self.data.y[n_id[:batch_size]].to(device) 
                adjs = adjs[1]
                edge_index, _, size = adjs.to(device)  
                out = self.forward(x_batch, edge_index)  
                out = out[:size[1]]  
                loss = F.nll_loss(out, y_batch)  
                #loss = criterion(out, y_batch)  
                loss.backward() 
                optimizer.step() 

                total_loss += loss.item()
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        total_grad_norm[name] += param.grad.norm().item()
                num_batches += 1

            #avg_loss = total_loss / num_batches
            #print(f"\nEpoch {epoch}, Average Loss: {avg_loss:.4f}")
            #for name in total_grad_norm.keys():
            #    avg_grad_norm = total_grad_norm[name] / num_batches
            #    print(f"Layer: {name} | Average Grad Norm: {avg_grad_norm:.4f}")

            # Validation
            self.eval()
            with torch.no_grad():
                out_val = self.forward(self.data.x.to(device), self.data.edge_index.to(device)) 
                loss_val = F.nll_loss(out_val[self.data.val_mask].to(device), self.data.y[self.data.val_mask].to(device))  
                #loss_val = criterion(out_val[self.data.val_mask].to(device), self.data.y[self.data.val_mask].to(device))  

            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, training loss: {total_loss / len(train_loader)}, validation loss: {loss_val.item()}')

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                early_stopping_counter = patience
            else:
                early_stopping_counter -= 1
            if early_stopping_counter <= 0:
                if verbose:
                    print(f'Early stopping at epoch {epoch}, best validation loss: {best_loss_val.item()}')
                break
    '''