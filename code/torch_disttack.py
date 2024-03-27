from torch.nn.modules.module import Module
from torch.cuda.amp import autocast
import numpy as np
import torch
import scipy.sparse as sp
import os.path as osp
import torch.nn.functional as F
from collections import namedtuple
from functools import lru_cache
import utils_disttack as utils
import os
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import GCNConv
import line_profiler
import sys
import torch.nn as nn

from torch_scatter import scatter_add
from torch_geometric.utils import k_hop_subgraph

SubGraph = namedtuple('SubGraph', ['edge_index',
                                   'self_loop', 'self_loop_weight',
                                   'edge_weight','edges_all'])


class BaseAttack(Module):

    def __init__(self, model, nnodes, attack_structure=True, attack_features=True, device='cpu'):
        super(BaseAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device

        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

        self.modified_adj = None
        self.modified_features = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        pass

    def check_adj(self, adj):
        if type(adj) is torch.Tensor:
            adj = adj.cpu().numpy()
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        if sp.issparse(adj):
            assert adj.tocsr().max() == 1, "Max value should be 1!"
            assert adj.tocsr().min() == 0, "Min value should be 0!"
        else:
            assert adj.max() == 1, "Max value should be 1!"
            assert adj.min() == 0, "Min value should be 0!"

    def save_adj(self, root=r'/tmp/', name='mod_adj'):

        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = self.modified_adj

        if type(modified_adj) is torch.Tensor:
            modified_adj = utils.to_scipy(modified_adj)
        if sp.issparse(modified_adj):
            modified_adj = modified_adj.tocsr()
        sp.save_npz(osp.join(root, name), modified_adj)

    def save_features(self, root=r'/tmp/', name='mod_features'):

        assert self.modified_features is not None, \
                'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features

        if type(modified_features) is torch.Tensor:
            modified_features = utils.to_scipy(modified_features)
        if sp.issparse(modified_features):
            modified_features = modified_features.tocsr()
        sp.save_npz(osp.join(root, name), modified_features)

class Disttack(BaseAttack):
    """Disttack proposed in `Disttack: Graph Adversarial Attacks Toward Distributed GNN Training` 

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, adj, nnodes=None, attack_structure=True, attack_features=True, device='cpu'):

        super(Disttack, self).__init__(model=None, nnodes=nnodes,
                                       attack_structure=attack_structure, attack_features=attack_features, device=device)

        adj_coo = adj.tocoo()
        self.values = torch.tensor(adj_coo.data, device=device)
        self.indices = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), device=device)
        self.adj_tensor = torch.sparse_coo_tensor(self.indices, self.values, adj.shape, device=device)
        self.selfloop_degree = torch.sparse.sum(self.adj_tensor, dim=1).to_dense() + 1
        self.ori_adj = adj
        self.adj = adj
        self.edge_index = np.asarray(self.ori_adj.nonzero())
        self.edge_index = torch.as_tensor(self.edge_index, dtype=torch.long, device=self.device)
        
        self.target_node = None
        self.logits = model.predict()
        self.K = 2
        W = model.convs[0].lin_r.weight.to(device)
        b = model.convs[0].lin_r.bias
        if b is not None:
            b = b.to(device)

        self.weight, self.bias = W, b
    @lru_cache(maxsize=1)
    def compute_XW(self):
        return F.linear(self.modified_features, self.weight)
    def sub_XW(self):
        with autocast():
            result = F.linear(self.sub_features, self.weight)
        return result
    def ego_subgraph_prof(self):
        profile = line_profiler.LineProfiler(self.ego_subgraph)
        profile.enable()
        result = self.ego_subgraph()
        profile.disable()
        profile.print_stats(sys.stdout)
        return result
    
    def ego_subgraph(self):

        sub_nodes, sub_edges, *_ = k_hop_subgraph(int(self.target_node), self.K-1, self.edge_index)
        sub_edges = sub_edges[:, sub_edges[0] < sub_edges[1]]
        self.sub_features = self.modified_features[sub_nodes] #sub_features tensor
        self.sub_target = torch.where(sub_nodes == self.target_node)[0].item()
        #print(self.sub_features[self.sub_target].equal(self.modified_features[self.target_node])) its true
        node_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(sub_nodes.tolist())}
        original_indices = torch.tensor(list(node_mapping.keys()), dtype=torch.long, device=self.device)

        new_indices = torch.tensor(list(node_mapping.values()), dtype=torch.long, device=self.device)

        node_mapping_tensor = torch.zeros(original_indices.max() + 1, dtype=torch.long, device=self.device)
        node_mapping_tensor[original_indices] = new_indices

        new_edge_index_0 = node_mapping_tensor[sub_edges[0]]
        new_edge_index_1 = node_mapping_tensor[sub_edges[1]]

        self.new_edge_index = torch.stack([new_edge_index_0, new_edge_index_1], dim=0)
        return sub_nodes, sub_edges

    def attack_prof(self, features, labels, target_node, n_perturbations, direct=True, **kwargs):
        profile = line_profiler.LineProfiler(self.attack)
        profile.enable()
        self.attack(features, labels, target_node, n_perturbations, direct=direct, n_influencers=3, **kwargs)
        profile.disable()
        profile.print_stats(sys.stdout)
    def attack(self, features, labels, target_node, n_perturbations, direct=True, n_influencers=3, **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        features :
            Original (unperturbed) node feature matrix
        adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        target_node : int
            target_node node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        direct: bool
            whether to conduct direct attack
        n_influencers : int
            number of the top influencers to choose. For direct attack, it will set as `n_perturbations`.
        """
        if sp.issparse(features):
            features = features.A

        if not torch.is_tensor(features):
            features = torch.tensor(features, device=self.device)

        if torch.is_tensor(self.adj):
            adj = utils.to_scipy(self.adj).csr()

        self.modified_features = features.requires_grad_(bool(self.attack_features))
        self.sub_features = None
        self.new_edge_index = None

        target_label = torch.LongTensor([labels[target_node]])
        best_wrong_label = torch.LongTensor([(self.logits[target_node].cpu() - 1000 * torch.eye(self.logits.size(1))[target_label]).argmax()])

        self.target_label = target_label.to(self.device)
        self.best_wrong_label = best_wrong_label.to(self.device)
        self.n_perturbations = n_perturbations
        self.target_node = target_node
        self.direct = direct

        attacker_nodes = torch.where(torch.as_tensor(labels) == best_wrong_label)[0]
        subgraph = self.get_subgraph(attacker_nodes, n_influencers)

        if not direct:
        
            mask = torch.logical_or(subgraph.edge_index[0] == target_node, subgraph.edge_index[1] == target_node).to(self.device)

        structure_perturbations = []
        feature_perturbations = []
        num_features = features.shape[-1] #its 602
        for _ in range(n_perturbations):
            edge_grad, features_grad = self.compute_gradient(subgraph)

            if self.attack_structure:
                edge_grad = edge_grad.half()
                #subgraph.edge_weight = subgraph.edge_weight.half()
                batch_size = 2048  # set based on your GPU memory
                num_batches = (edge_grad.shape[0] + batch_size - 1) // batch_size
                if edge_grad.numel() > 0:
                    min_val = edge_grad.min()
                    cuda_stream = torch.cuda.Stream()
                    temp_tensor = torch.empty_like(edge_grad)
                    
                    with torch.no_grad():
                        for i in range(num_batches):
                            start = i * batch_size
                            end = min(start + batch_size, edge_grad.shape[0])
                            with torch.cuda.stream(cuda_stream):
                                temp_tensor[start:end].mul_(-2 * subgraph.edge_weight[start:end] + 1)
                                temp_tensor[start:end].sub_(min_val)
                            cuda_stream.synchronize()
                            
                    edge_grad = temp_tensor 

                    if edge_grad.numel() > 0:
                        _, max_edge_idx = torch.max(edge_grad, dim=0)
                    else:
                        #initialize
                        max_edge_idx = 1 
                    
                    torch.cuda.empty_cache()
                    best_edge = subgraph.edge_index[:, max_edge_idx]
                    subgraph.edge_weight.data[max_edge_idx] = 0.0
                    self.selfloop_degree[best_edge] -= 1.0
                    u, v = best_edge.tolist()
                    structure_perturbations.append((u, v))
                else:
                    pass

            if self.attack_features:
                features_grad = features_grad.half()
                self.sub_features = self.sub_features.half()
                batch_size = 409600  # based on your GPU memory
                num_batches = (features_grad.shape[0] + batch_size - 1) // batch_size
                min_val = features_grad.min()
                cuda_stream = torch.cuda.Stream()
                temp_tensor = torch.empty_like(features_grad)
                with torch.no_grad():
                    for i in range(num_batches):
                        start = i * batch_size
                        end = min(start + batch_size, features_grad.shape[0])
                        with torch.cuda.stream(cuda_stream):    
                            temp_tensor[start:end].mul_(1 - 2 * self.sub_features[start:end])
                            temp_tensor[start:end].sub_(min_val)
                        cuda_stream.synchronize()
                    torch.cuda.empty_cache()

                if not direct:
                    features_grad[target_node] = 0.
                _, max_feature_idx = torch.max(features_grad.view(-1), dim=0)
                #max_feature_score = max_feature_grad.item()
                #max_edge_grad = 0.0
                max_edge_idx = 1  # 这里进行了初始化
                u, v = divmod(max_feature_idx.item(), num_features)
                feature_perturbations.append((u, v))
                #self.sub_features[u] = 1. - self.sub_features[u]
                perturbed_features = 1. - self.sub_features[u].clone()
                self.sub_features[u] = perturbed_features
                # self.sub_features[u, v].data.fill_(1. - self.sub_features[u, v].data)
                new_modified_features = self.modified_features.clone()
                new_modified_features[self.sub_nodes] = self.sub_features.detach().to(new_modified_features.dtype)

                # update self.modified_features
                self.modified_features = new_modified_features
                #self.modified_features[u, v].data.fill_(1. - self.modified_features[u, v].data)
        #ed = time.time()
        #print('Time cost in seconds per attack:', ed - st)
        torch.cuda.empty_cache()

        if structure_perturbations:
            modified_adj = self.adj.tolil(copy=True)
            row, col = list(zip(*structure_perturbations))
            modified_adj[row, col] = modified_adj[col, row] = 1 - modified_adj[row, col].A
            modified_adj = modified_adj.tocsr(copy=False)
            modified_adj.eliminate_zeros()
        else:
            modified_adj = self.adj.copy()

        self.modified_adj = modified_adj
        #self.modified_features = self.modified_features.detach().cpu().numpy()
        self.modified_features = self.modified_features
        self.structure_perturbations = structure_perturbations
        self.feature_perturbations = feature_perturbations
        
    def get_subgraph(self, attacker_nodes, n_influencers=None):
        target_node = self.target_node
        neighbors = self.ori_adj[target_node].indices
        sub_nodes, sub_edges = self.ego_subgraph()
        self.sub_nodes = sub_nodes

        if self.direct or n_influencers is not None:
            influencers = [target_node]
            attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        else:
            influencers = neighbors

        subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)

        if n_influencers is not None and self.attack_structure:
            if self.direct:
                influencers = [target_node]
                #attacker_nodes = self.get_topk_influencers(subgraph, k=self.n_perturbations + 1)
                attacker_nodes = self.get_topk_influencers(subgraph, k=self.n_perturbations)

            else:
                influencers = neighbors
                attacker_nodes = self.get_topk_influencers(subgraph, k=n_influencers)

            subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)
        return subgraph

    def get_topk_influencers(self, subgraph, k):
        _, non_edge_grad, _ = self.compute_gradient(subgraph)
        print("Length of non_edge_grad:", len(non_edge_grad))
        print("k:", k)
        if min(len(non_edge_grad), k) > 300:
            _, topk_nodes = torch.topk(non_edge_grad, k=300, sorted=False)
        else:               
            if len(non_edge_grad) < k:
                _, topk_nodes = torch.topk(non_edge_grad, k=len(non_edge_grad), sorted=False)
            else:
                _, topk_nodes = torch.topk(non_edge_grad, k=k, sorted=False)

        influencers = subgraph.non_edge_index[1][topk_nodes.cpu()]
        return influencers.cpu().numpy()

    def subgraph_processing(self, influencers, attacker_nodes, sub_nodes, sub_edges):
        unique_nodes = np.union1d(sub_nodes.tolist(), attacker_nodes)
        unique_nodes = torch.as_tensor(unique_nodes, device=self.device)
        self_loop = unique_nodes.repeat((2, 1))

        #edges_all = torch.cat([sub_edges, sub_edges[[1, 0]], self_loop], dim=1)
        #edges_all = torch.cat([sub_edges, sub_edges[[1, 0]]], dim=1)
        edges_all = torch.cat([sub_edges], dim=1)
        edge_weight = torch.ones(sub_edges.size(1), device=self.device).requires_grad_(bool(self.attack_structure)).to(torch.float16)
        self_loop_weight = torch.ones(self_loop.size(1), device=self.device).to(torch.float16)

        edge_index = sub_edges
        self_loop = self_loop

        subgraph = SubGraph(edge_index=edge_index, self_loop=self_loop, edges_all=edges_all,
                            edge_weight=edge_weight, self_loop_weight=self_loop_weight)

        return subgraph
    def SimpCov_prof(self, x, edge_index, edge_weight):
            profile = line_profiler.LineProfiler(self.SimpCov)
            profile.enable()
            x = self.SimpCov(x, edge_index, edge_weight)
            profile.disable()
            profile.print_stats(sys.stdout)
            return x
    
    def SimpCov(self, x, edge_index, edge_weight):
        row, col = edge_index
        #col = col.to(self.device)
        edge_weight = edge_weight[:row.shape[0]].unsqueeze(1)
        for _ in range(self.K):
            src = x[row]
            #edge_weight = edge_weight[:row.shape[0]]
            src *= edge_weight
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
            del src
        torch.cuda.empty_cache()
        return x
    
    def compute_gradient_prof(self, subgraph, eps=5.0):
            profile = line_profiler.LineProfiler(self.compute_gradient) 
            profile.enable()
            edge_grad, features_grad = self.compute_gradient(subgraph, eps=5.0)
            profile.disable()
            profile.print_stats(sys.stdout)
            return edge_grad, features_grad
        
    def compute_gradient(self, subgraph, eps=5.0):
        if self.attack_structure:
            # Detach and clone edge weights, convert to float16, and enable gradient tracking
            edge_weight = subgraph.edge_weight.detach().clone().to(torch.float16)
            edge_weight.requires_grad = True
            # No concatenation needed for undirected graph
            weights = edge_weight
        else:
            weights = subgraph.edge_weight.to(torch.float16)
            
        weights = self.gcn_norm(subgraph.edges_all, weights, self.selfloop_degree).to(torch.float16)
        
        #max_edges = 1000
        sub_x = self.sub_XW()
        
        # 计算logits
        logit = self.SimpCov(sub_x, self.new_edge_index, weights)  # [node_num, hidden_size]
        logit = logit[self.sub_target]
        if self.bias is not None:
            logit += self.bias

        # model calibration
        criterion = nn.BCEWithLogitsLoss()
        logit = F.log_softmax(logit.view(1, -1) / eps, dim=1)
        #dont use log_softmax when it's a multiple classification
        loss = F.nll_loss(logit, self.target_label) - F.nll_loss(logit, self.best_wrong_label)

        edge_grad = features_grad = None

        if self.attack_structure and self.attack_features:
            edge_grad, features_grad = torch.autograd.grad(loss, [edge_weight, self.modified_features], create_graph=False)
        elif self.attack_structure:
            edge_grad = torch.autograd.grad(loss, edge_weight, create_graph=False)[0]
        elif self.attack_features:
            features_grad = torch.autograd.grad(loss, self.sub_features, create_graph=False)[0]

        return edge_grad, features_grad


    @ staticmethod
    def gcn_norm(edge_index, weights, degree):
        row, col = edge_index
        inv_degree = torch.pow(degree, -0.5)
        normed_weights = weights * inv_degree[row] * inv_degree[col]
        #normed_weights = inv_degree[row] * inv_degree[col]
        return normed_weights
    '''
    
    def get_subgraph(self, attacker_nodes, n_influencers=None):
        target_node = self.target_node
        neighbors = self.ori_adj[target_node].indices
        sub_nodes, sub_edges = self.ego_subgraph()
        self.sub_nodes = sub_nodes

        if self.direct or n_influencers is not None:
            influencers = [target_node]
            attacker_nodes = np.setdiff1d(attacker_nodes, neighbors)
        else:
            influencers = neighbors

        subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)

        if n_influencers is not None and self.attack_structure:
            if self.direct:
                influencers = [target_node]
                #attacker_nodes = self.get_topk_influencers(subgraph, k=self.n_perturbations + 1)
                attacker_nodes = self.get_topk_influencers(subgraph, k=self.n_perturbations)

            else:
                influencers = neighbors
                attacker_nodes = self.get_topk_influencers(subgraph, k=n_influencers)

            subgraph = self.subgraph_processing(influencers, attacker_nodes, sub_nodes, sub_edges)
        return subgraph
    def subgraph_processing(self, influencers, attacker_nodes, sub_nodes, sub_edges):
        if not self.attack_structure:
            self_loop = sub_nodes.repeat((2, 1))
            edges_all = torch.cat([sub_edges, sub_edges[[1, 0]], self_loop], dim=1)
            edge_weight = torch.ones(edges_all.size(1), device=self.device)

            return SubGraph(edge_index=sub_edges, non_edge_index=None,
                            self_loop=None, edges_all=edges_all,
                            edge_weight=edge_weight, non_edge_weight=None,
                            self_loop_weight=None)

        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        non_edges = np.row_stack([row, col])

        if len(influencers) > 1:
            mask = self.ori_adj[non_edges[0],
                                non_edges[1]].A1 == 0
            non_edges = non_edges[:, mask]

        non_edges = torch.as_tensor(non_edges, device=self.device)
        unique_nodes = np.union1d(sub_nodes.tolist(), attacker_nodes)
        unique_nodes = torch.as_tensor(unique_nodes, device=self.device)
        self_loop = unique_nodes.repeat((2, 1))
        #edges_all = torch.cat([sub_edges, sub_edges[[1, 0]],
        #                       non_edges, non_edges[[1, 0]], self_loop], dim=1)
        edges_all = torch.cat([sub_edges, sub_edges[[1, 0]], self_loop], dim=1)                      

        edge_weight = torch.ones(sub_edges.size(1), device=self.device).requires_grad_(bool(self.attack_structure)).to(torch.float16)
        non_edge_weight = torch.zeros(non_edges.size(1), device=self.device).requires_grad_(bool(self.attack_structure)).to(torch.float16)
        self_loop_weight = torch.ones(self_loop.size(1), device=self.device).to(torch.float16)

        edge_index = sub_edges
        non_edge_index = non_edges
        self_loop = self_loop

        subgraph = SubGraph(edge_index=edge_index, non_edge_index=non_edge_index,
                            self_loop=self_loop, edges_all=edges_all,
                            edge_weight=edge_weight, non_edge_weight=non_edge_weight,
                            self_loop_weight=self_loop_weight)
        return subgraph


    def SimpCov(self, x, edge_index, edge_weight):
        row, col = edge_index
        for _ in range(self.K):
            #src = x[row] * edge_weight.view(-1, 1)
            src = x[row]
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
        return x
        '''