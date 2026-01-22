import os
import torch
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from torch_geometric.data import Data, Batch
from scipy.sparse import diags
from scipy.sparse.linalg import svds
from scipy.linalg import lstsq
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class PPR:
    def __init__(self, adj_mat, n_order=10):
        self.n_order = n_order
        self.adj_mat = adj_mat.astype(float)  

    def compute_pagerank(self, A, d=0.5, tol=1e-8, max_iter=100):
        n = A.shape[0]
        M = A / A.sum(axis=0)
        v = np.random.rand(n, 1)
        v = v / np.linalg.norm(v, 1)
        last_v = np.ones((n, 1), dtype=np.float32) * np.inf

        iter_count = 0
        while np.linalg.norm(v - last_v, 2) > tol and iter_count < max_iter:
            last_v = v.copy()
            v = d * (M @ v) + (1 - d) / n
            iter_count += 1

        return v

    def compute_pagerank_vectors(self, A, params):
        pagerank_vectors = [self.compute_pagerank(A, d=w).flatten() for w in params]
        return np.array(pagerank_vectors).T

    def approximate_pagerank(self, U, M, b):
        W = U.T @ M @ U
        y, _, _, _ = lstsq(W, U.T @ b)
        return U @ y

    def deim(self, U, k):
        indices = []
        P = np.zeros((U.shape[0], k))
        for i in range(k):
            index = np.argmax(np.linalg.norm(U - P @ (np.linalg.lstsq(P, U, rcond=None)[0]), axis=1))
            indices.append(index)
            P[:, i] = U[:, index]
        return indices

    def search(self, k=20, params=np.linspace(0.1, 0.95, 50), sc=0.0000035):
        A = self.adj_mat

        X = self.compute_pagerank_vectors(A, params)

        U = svds(X, k=k)[0]

        M = A  # PageRank matrix
        b = np.ones(A.shape[0]) / A.shape[0]  # Right-hand side vector
        scores = self.approximate_pagerank(U, M, b)

        if sc is not None:
          neighbor = np.where(scores > sc)[0]
        else:
          neighbor = np.arange(len(scores))

        return neighbor


    def process(self, path, seed, sc):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            neighbor = self.search(sc=sc)
            torch.save(neighbor, ppr_path)

    def search_all(self, path, selected_nodes, sc=0.00035):
        neighbor = {}
        if os.path.isfile(path + '_neighbor') and os.stat(path + '_neighbor').st_size != 0:
            neighbor = torch.load(path + '_neighbor')
        else:
            os.makedirs(path, exist_ok=True)

            from functools import partial
            process_func = partial(self.process, path, sc=sc)

            with mp.Pool() as pool:
                pool.map(process_func, selected_nodes, chunksize=1000)

            for i in selected_nodes:
                ppr_path = os.path.join(path, 'ppr{}'.format(i))
                if os.path.isfile(ppr_path):
                    neighbor[i] = torch.load(ppr_path)
 
            torch.save(neighbor, path + '_neighbor')
            os.system('rm -r {}'.format(path))
        return neighbor

class Subgraph:
    def __init__(self, adj_mat, x, path='./subgraph_data', n_order=10):
        self.x = x
        self.path = path
        self.adj_mat = adj_mat.astype(float) 
        self.node_num = adj_mat.shape[0]
        self.n_order = n_order

        self.ppr = PPR(adj_mat)

        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}

    def adjust_edge(self, idx):
        new_index = [[], []]
        new_weights = []
        nodes = set(idx)
        for i in idx:
            for j in nodes:
                if self.adj_mat[i, j] != 0:
                    new_index[0].append(i)
                    new_index[1].append(j)
                    new_weights.append(self.adj_mat[i, j])
        return torch.LongTensor(new_index), torch.FloatTensor(new_weights)

    def build(self, selected_nodes, sc=0.0000018):
        if selected_nodes is None:
            selected_nodes = list(range(self.node_num))

        self.neighbor = self.ppr.search_all(self.path, selected_nodes, sc=sc)
        for i in selected_nodes:
            if i not in self.neighbor:
                
                nodes = [i]
            else:
                nodes = self.neighbor[i]
            
                if isinstance(nodes, torch.Tensor):
                    nodes = nodes.tolist()
            x = self.x[nodes]
            edge_index, edge_attr = self.adjust_edge(nodes)
            self.subgraph[i] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        torch.save(self.subgraph, self.path + '_subgraph')

    def search(self, node_list):
        batch = []
        index = []
        size = 0
        for node in node_list:
            if node in self.subgraph:
                batch.append(self.subgraph[node])
                index.append(size)
                size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        batch = Batch.from_data_list(batch)
        return batch, index