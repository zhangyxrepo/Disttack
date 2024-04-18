import argparse
import GPUtil
import numpy as np
import networkx as nx
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from tqdm import tqdm
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import ClusterData, ClusterLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import subgraph as get_subgraph
from torch_sparse import SparseTensor
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils.convert import to_networkx
from gcn import SAGE
from torch_disttack import *
from torch_geometric.datasets import Reddit, Yelp, Coauthor, Reddit2
from torch_geometric.loader import NeighborSampler

from ogb.nodeproppred import PygNodePropPredDataset


def dist_test(model, data):
    model.eval()
    #data = data.to('cuda:0')
    model.to('cpu')
    with torch.no_grad():  # Disable gradient calculation
        # Forward pass - we use the entire graph
        output = model(data.x, data.edge_index)
        test_mask = data.test_mask
        test_preds = output[test_mask].max(1)[1]
        test_labels = data.y[test_mask]
        correct = test_preds.eq(test_labels).sum().item()
        total = test_mask.sum().item()
        accuracy = correct / total
    return accuracy


def run(rank, world_size, dataset, advgraph, subgraph, accs, args):
    print(f'Running on Rank {rank}!')
    os.environ['MASTER_ADDR'] = '127.0.1.1' #'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    data = dataset[0]
    data = advgraph
    data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
    x, y = data.x, data.y
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                   sizes=[25, 15], batch_size=args.batchsize, return_e_id=False,
                                   shuffle=True, num_workers=6)
    if rank == 0:
        test_loader = NeighborSampler(data.edge_index, node_idx=None,
                                          sizes=[-1], batch_size=args.batchsize, return_e_id=False,
                                          shuffle=False, num_workers=6)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SAGE(
        None,
        None,
        args.model,
        dataset.num_features,
        args.hidden,
        dataset.num_classes,
        num_layers = 2
    ).to(rank) 
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10, verbose=True)

    stop_training = torch.tensor(0, dtype=torch.int, device=rank)
    best_loss_val = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    n_epochs_stop = 25  
    grad_diff_array = []
    val_loss_array = []
    
    for epoch in range(1, 251):
        model.train()
        total_loss = 0.0
        for batch_size, n_id, adjs in train_loader:
            optimizer.zero_grad()
            x_batch = x[n_id].to(rank)  
            y_batch = y[n_id[:batch_size]].squeeze().long().to(rank)  
            adjs = [adj.to(rank) for adj in adjs]
            out = model.module.adj_forward(x_batch, adjs)  # forward
            loss = F.nll_loss(out, y_batch)  # compute loss
            loss.backward()  # backwardpropagation
            gradient_norm = 0.0
            for param in model.parameters():
                gradient_norm += param.grad.norm(2).item() ** 2
            gradient_norm = gradient_norm ** 0.5
            print(f'Epoch {epoch}, Rank {rank}, Gradient Norm: {gradient_norm}')
            first_param_grad = next(model.parameters()).grad.data
            # sum all computing nodes gradient by torch.distributed.reduce
            dist.reduce(first_param_grad, dst=0, op=dist.ReduceOp.SUM)
            optimizer.step()  # update
            total_loss += loss.item()  # cal loss
        dist.barrier()
        #print(f'Rank {rank} hit the dist barrier of epoch{epoch}\n', flush=True)
        average_grad = first_param_grad / dist.get_world_size()
        grad_diff = first_param_grad - average_grad
        print(f'Epoch {epoch}, Rank {rank}, Gradient Diff Norm: {grad_diff.norm(2)}')
        grad_diff_array.append(grad_diff.norm(2).item())
        
        if rank == 0:
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    out_val = model.module.edge_forward(data.x.to(rank), data.edge_index.to(rank))  # 如果使用GPU
                    val_loss = F.nll_loss(out_val[data.val_mask].to(rank), data.y[data.val_mask].to(rank))
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    torch.cuda.empty_cache()
                    print(f'Rank {rank}, Epoch {epoch} Testing on {args.dataset}', flush=True)
                    print(f'Average val loss: {val_loss:.3f}', flush=True)
                    scheduler.step(val_loss)
                    val_loss_array.append(val_loss.item())
                    print(val_loss_array)
                for param_group in optimizer.param_groups:
                    print(f'Current learning rate: {param_group["lr"]}')
                res = out_val.argmax(dim=-1) == y.to(rank)
                acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
                acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
                acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
                print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
        dist.barrier()
    dist.destroy_process_group()
    np.save(f'./part/{args.dataset}/grad_diff_{rank}_{args.model}.npy', grad_diff_array)
    if rank == 0:
        np.save(f'./part/{args.dataset}/val_loss_{rank}_{args.model}.npy', val_loss_array)

def main():
    parser = argparse.ArgumentParser(description="multi-gpu")
    parser.add_argument("--dataset", type=str, default='r2',
                        choices=['arxiv', 'papers100M', 'products', 'reddit', 'flickr', 'coauthor-physics', 'r2'])
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument("--num-gpu", type=int, default=-1,
                        help="The number of gpus used.")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="The batch size during training.")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden feature size.")
    parser.add_argument("--model", type=str, default='gat',
                        choices=['gat', 'gcn', 'gin', 'sage', 'sgc'])
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = args.dataset
    world_size = torch.cuda.device_count() if args.num_gpu == -1 else args.num_gpu
    rank = 0
    os.makedirs(f'./part/{args.dataset}', exist_ok=True)
    print_time = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = '/home/zhangyuxiang/EPD_multi/dataset'
    if rank == world_size:
        data = np.zeros((5000, world_size, 2))
        for _ in range(5000):
            GPUs = GPUtil.getGPUs()
            for i in range(world_size):
                gpu = GPUs[i] # CUDA_VISIBLE_DEVICES=1,2,3,4
                data[_, i, 0] = gpu.load * 100
                data[_, i, 1] = gpu.memoryUsed

        for i in range(world_size):
            mask = data[:, i, 0] > 10
            print(f'GPU {i} util {data[:, i, 0][mask].mean()}, memory {data[:, i, 1].max()}')
        exit()

    if args.dataset == 'reddit':
        dataset = Reddit(f'{data_dir}/reddit')
        data = dataset[0]
        data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
        test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
        print(data)
    elif args.dataset == 'r2':
        dataset = Reddit2(f'{data_dir}/reddit2')
        data = dataset[0]
        data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
        test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    elif args.dataset == 'flickr':
        dataset = Reddit2(f'{data_dir}/flickr')
        data = dataset[0]
        data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
        test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
        print(data)
    elif args.dataset == 'arxiv':
        dataset = Reddit2(f'{data_dir}/arxiv2')
        data = dataset[0]
        adj = sp.csr_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),shape=(data.num_nodes, data.num_nodes))
        data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        print(data)
    elif args.dataset.startswith('coauthor'):
        field = args.dataset.split('-')[-1].lower() 
        dataset = Coauthor(f'{data_dir}/Coauthor{field.upper()}', field)
        data = dataset[0]
        print(data)
        data.edge_index.row=data.edge_index[1]
        data.edge_index.col=data.edge_index[0]
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        val_idx = split_idx['valid']
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[val_idx] = True
        test_idx = split_idx['test']
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        x, y = data.x, data.y.squeeze()
        data.y = data.y.squeeze()
    elif args.dataset in ['products', 'papers100M']:       
        dataset = PygNodePropPredDataset(
                name=f'ogbn-{args.dataset}', root=data_dir, transform=None)
        data = dataset[0]
        if args.dataset == 'papers100M':
            pass
        else:
            data.edge_index.row=data.edge_index[1]
            data.edge_index.col=data.edge_index[0]
            data.edge_index = SparseTensor(row=data.edge_index[1], col=data.edge_index[0])
            pass
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True 
        val_idx = split_idx['valid']
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[val_idx] = True
        test_idx = split_idx['test']
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        data.y = data.y.squeeze()
        print(data)
        pause = 1
    else:
        assert 0
    
    subgraph = []

    origraph = subgraph

    graph = data
    attr_graph = to_networkx(dataset[0], to_undirected=True)
    #graph_attributes(attr_graph)
    #exit()
    ######### deal with the adj features and labels ############
    adj = data.edge_index.to_scipy('csr')
    features = sp.csr_matrix(graph.x.numpy())
    labels = graph.y.numpy()
    ############################################################
    degrees = adj.sum(0).A1
    train_idx = graph.train_mask.nonzero(as_tuple=False).view(-1)
    train_nodes = train_idx.cpu().numpy()
    targets_idx = train_nodes[:22735]
    #######################pertube structures#######################
    
    surrogate = SAGE(graph, graph, args.model, dataset.num_features, args.hidden, dataset.num_classes)
    #modified_adj = sp.load_npz(f'./Disttack/adv_adj/{args.dataset}_{device}_adv_adj_4.npz')
    modified_adj = adj
    modified_features = sp.csr_matrix(graph.x.cpu().numpy()) 
    #modified_features = sp.csr_matrix(np.load(f'./Disttack/adv_adj/{args.dataset}_{device}_adv_fea.npz.npy'))
    save_adj = None
    os.makedirs('./Disttack/adv_adj/', exist_ok=True)
    model = Disttack(surrogate, adj=modified_adj, attack_structure=False, attack_features=True, device=device)
    model = model.to(device)
    print(f'Now running on device {device}')
    features = modified_features
    for target_node in tqdm(targets_idx, desc="Perturbing structures"):
        n_perturbations = int(degrees[target_node])
        model.attack(features, labels, target_node, n_perturbations, n_influencers=None)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    #save_adj = modified_adj
    save_fea = modified_features.detach().cpu().numpy()  # 如果 modified_features 是 tensor
    #sp.save_npz(f'./Disttack/adv_adj/{args.dataset}_cuda:0_adv_adj_full.npz', save_adj)
    np.save(f'./Disttack/adv_adj/{args.dataset}_cuda:0_adv_fea_full', save_fea)


    
    '''
    #######################pertube structures ends#######################
    '''
    
    ##############distribute poisoned adj_fea###############
    #modified_adj = sp.load_npz(f'./Disttack/adv_adj/{args.dataset}_cuda:0_adv_adj_full.npz')
    modified_fea = np.load(f'./Disttack/adv_adj/{args.dataset}_cuda:0_adv_fea_full.npy')
    #data.edge_index = torch.tensor(modified_adj.nonzero(), dtype=torch.long)
    data.edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    data.x = torch.tensor(modified_fea, dtype=torch.float)

    modified_graph_1 = subgraph[0]
    modified_adj_1 = sp.load_npz(f'./Disttack/adv_adj/{args.dataset}_cuda:0_adv_adj_16.npz')
    modified_features_1 = np.load(f'./Disttack/adv_adj/{args.dataset}_cuda:0_adv_fea_16.npz.npy')
    modified_graph_1.edge_index = torch.tensor(modified_adj_1.nonzero(), dtype=torch.long)
    modified_graph_1.x = torch.tensor(modified_features_1, dtype=torch.float)
    
    modified_graph_2 = subgraph[1]
    modified_adj_2 = sp.load_npz(f'./Disttack/adv_adj/{args.dataset}_cuda:1_adv_adj_16.npz')
    modified_features_2 = np.load(f'./Disttack/adv_adj/{args.dataset}_cuda:1_adv_fea_16.npz.npy')
    modified_graph_2.edge_index = torch.tensor(modified_adj_2.nonzero(), dtype=torch.long)
    modified_graph_2.x = torch.tensor(modified_features_2, dtype=torch.float)
    modified_graph_3 = subgraph[2]
    modified_adj_3 = sp.load_npz(f'./Disttack/adv_adj/{args.dataset}_cuda:2_adv_adj_16.npz')
    modified_features_3 = np.load(f'./Disttack/adv_adj/{args.dataset}_cuda:2_adv_fea_16.npz.npy')
    modified_graph_3.edge_index = torch.tensor(modified_adj_3.nonzero(), dtype=torch.long)
    modified_graph_3.x = torch.tensor(modified_features_3, dtype=torch.float)
    modified_graph_4 = subgraph[3]
    modified_adj_4 = sp.load_npz(f'./Disttack/adv_adj/{args.dataset}_cuda:3_adv_adj_4.npz')
    modified_features_4 = np.load(f'./Disttack/adv_adj/{args.dataset}_cuda:3_adv_fea_4.npz.npy')
    modified_graph_4.edge_index = torch.tensor(modified_adj_4.nonzero(), dtype=torch.long)
    modified_graph_4.x = torch.tensor(modified_features_4, dtype=torch.float)

    print('Its done')


    #single-computing device test code, dont use it unless neccessary
    test_model = SAGE(modified_graph_2.to(rank), graph.to(rank), args.model, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
    #print(test_model)
    test_model.train_with_early_stopping(train_iters=200, initialize=True, verbose=False, patience=500) # pre train with early-stopping
    print('=== Structure perturbations ===')
    #print(model.structure_perturbations)
    print('=== Feature perturbations ===')
    #print(model.feature_perturbations)
    print('=== testing surrogate model on perturbed graph ===')
    #test(adj, features, target_node)
    test_model.test()
    print('=== testing surrogate model on clean graph ===')
    #test(modified_adj, modified_features, target_node)
    surrogate = SAGE(graph.to(rank), graph.to(rank), args.model, dataset.num_features, args.hidden, dataset.num_classes).to(rank)
    surrogate.train_with_early_stopping(train_iters=200, initialize=True, verbose=False, patience=500)
    surrogate.test()

    
    pause = 1

    world_size = torch.cuda.device_count() if args.num_gpu == -1 else args.num_gpu
    print('Let\'s use', world_size, 'GPUs!')
    accs = []
    mp.spawn(run, args=(world_size, dataset, data, subgraph, accs, args), nprocs=world_size, join=True)
if __name__=="__main__":
    main()
