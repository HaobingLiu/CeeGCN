import numpy as np
import dgl
import torch
import torch.nn.functional as F
from constants import eta, pos_num, neg_num

def sampling(adj_matrix, center_node):
    n_nodes = adj_matrix.shape[0]
    degrees = np.sum(adj_matrix, axis=1)
    probs = degrees / np.sum(degrees)
    neg_sample_probs = probs ** 0.75
    neg_sample_probs /= np.sum(neg_sample_probs)
    neg_samples = np.random.choice(n_nodes, size=(len(center_node), neg_num), replace=True, p=neg_sample_probs)

    pos_samples = []
    for i in range(len(center_node)):
        node_id = center_node[i]
        neighbor_ids = np.where(adj_matrix[node_id] == 1)[0]
        if len(neighbor_ids) > pos_num:
            neighbor_ids = np.random.choice(neighbor_ids, size=pos_num, replace=False)
        pos_samples.append(neighbor_ids)


    return pos_samples, neg_samples

def affinity(inputs1, inputs2):
    pos_score = torch.nn.functional.cosine_similarity(inputs1.unsqueeze(0), inputs2, dim=1)
    pos_score = pos_score.sum()
    pos_score = torch.log(torch.sigmoid(pos_score))
    return pos_score

def neg_affinity(inputs1, inputs2):
    neg_score = torch.nn.functional.cosine_similarity(inputs1.unsqueeze(0), inputs2, dim=1)
    neg_score = neg_score.sum()
    neg_score = inputs2.size(0) * torch.mean(torch.log(torch.sigmoid(-neg_score)))
    return neg_score

def xent_loss(adj, F):
    neg_sample_weights = 1.0
    nodes_score = []
    for i in range (len(adj)):
        id = [i]
        pos_id, neg_id = sampling(adj, id)
        pos_id = np.array(pos_id).flatten()
        neg_id = np.array(neg_id).flatten()#
        device=F.device
        
        input = torch.index_select(F, 0, torch.tensor([i], device=device))#
        pos_sam = torch.index_select(F, 0, torch.tensor(pos_id, device=device))#
        neg_sam = torch.index_select(F, 0, torch.tensor(neg_id, device=device))
        
        aff = affinity(input, pos_sam)
        neg_aff = neg_affinity(input, neg_sam)
        
        nodes_score.append(torch.mean(- aff - neg_aff))
   
    loss = torch.mean(torch.stack(nodes_score))
    return loss

def weighted_modularity(graph, community_matrix):
    num_nodes = graph.number_of_nodes()
    edge_weights = graph.edata['e'].detach()
    m = torch.sum(edge_weights) / 2  

    modularity = 0.0
    for i, j, w_ij in zip(*graph.edges(), edge_weights):
        successors_i = graph.successors(i).detach()
        successors_j = graph.successors(j).detach()
        ki = torch.sum(edge_weights[successors_i])
        kj = torch.sum(edge_weights[successors_j])
        delta_c = 1 if community_matrix[i] == community_matrix[j] else 0
        modularity += (w_ij - (ki * kj) / (2 * m)) * delta_c

    modularity /= (2 * m)
    return modularity



def total_loss(G, adj, pred, F):
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred, device=F.device)


    if isinstance(F, np.ndarray):
        F = torch.tensor(F, device=F.device)

    G=G.to(F.device)
    adj=adj.to(F.device)
    pred=pred.to(F.device)
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy() 
    adj = np.int64(adj > 0)
    x_loss = xent_loss(adj, F)
    mol_loss = weighted_modularity(G, pred)
    tot_loss = x_loss + eta * mol_loss
    return tot_loss
