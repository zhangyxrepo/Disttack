import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from sgc import SGC
from deeprobust.graph.global_attack import Random
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from torch_geometric.datasets import Reddit, Yelp, Coauthor, Reddit2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_dir = '/home/zhangyuxiang/EPD_multi/dataset'
dataset = Reddit2(f'{data_dir}/flickr')
data = dataset[0]
graph = data
print(data)

adj = sp.csr_matrix((np.ones(graph.num_edges), (graph.edge_index[0].numpy(), graph.edge_index[1].numpy())),shape=(graph.num_nodes, graph.num_nodes))
features = sp.csr_matrix(graph.x.numpy())
labels = graph.y.numpy()
idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
idx_val = data.val_mask.nonzero(as_tuple=False).view(-1)
idx_test = data.test_mask.nonzero(as_tuple=False).view(-1)
idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Attack Model
model = Random()

n_perturbations = int(len(idx_train)//4)

model.attack(adj, n_perturbations)
modified_adj = model.modified_adj

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True)
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)

modified_adj = normalize_adj(modified_adj)
modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
modified_adj = modified_adj.to(device)


def test(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    #gcn = GCN(nfeat=features.shape[1],
    #          nhid=16,
    #          nclass=labels.max().item() + 1,
    #          dropout=0.5, device=device)
    gcn = SGC(nfeat=features.shape[1],
              nclass=labels.max().item() + 1,
              device=device)

    gcn = gcn.to(device)

    optimizer = optim.Adam(gcn.parameters(),
                           lr=0.01, weight_decay=5e-4)

    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    print('=== testing GCN on perturbed graph ===')
    test(modified_adj)


if __name__ == '__main__':
    main()