import os
import scipy.sparse as sp
import networkx as nx
import numpy as np
import dgl
import torch
import tensorflow as tf
from sampling_center import sample_center


def adj_convert_to_graph(adj):
    graph = nx.Graph()
    weight = []
    for i in range(len(adj)):
        for j in range(len(adj)):
            if(adj[i][j] != 0):
              graph.add_edge(i, j)
    graph = dgl.DGLGraph(graph)
    edge = graph.edges()
    source = edge[0].numpy()
    target = edge[1].numpy()

    for i in range(len(adj)):
        for j in range(len(adj)):
            for k in range(len(source)):
              if(source[k] == i and target[k] == j):
                 weight.append(adj[i][j])
    graph.edata['w'] = torch.tensor(weight)
    graph.edata['w'] = graph.edata['w'].unsqueeze(1)
    return graph

def load_data():
    selected_nodes = sample_center()
    dataset_folder = "boat"
    content_d = {}
    with open(os.sep.join([dataset_folder, "boat_info.csv"]), "r") as fin_c:
        classes = set()
        docs = set()
        for line in fin_c:
            splitted = line.strip().split(",")
            content_d[splitted[0]] = {"class": splitted[1]}
            classes.add(splitted[1])
            docs.add(splitted[0])
    n = len(docs)
    class_to_id = {}
    for i, class_ in enumerate(list(classes)):
        class_to_id[class_] = i

    for i, doc in enumerate(list(docs)):
        content_d[doc]["id"] = i
    labels = []

    for key in content_d.keys():
        labels.append([content_d[key]["id"], class_to_id[content_d[key]["class"]]])

    labels = sorted(labels, key=lambda y: y[0])
    cites = []
    with open(os.sep.join([dataset_folder, "boat_gra.csv"]), "r") as fin_c:
        for line in fin_c:
            splitted = line.strip().split(",")
            if (len(splitted) > 1 and splitted[0] in content_d and splitted[1] in content_d):
                cites.append([content_d[splitted[0]]["id"], content_d[splitted[1]]["id"], int(splitted[2])])
                cites.append([content_d[splitted[1]]["id"], content_d[splitted[0]]["id"], int(splitted[2])])
    cited = np.array(cites).T
    adj_complete = sp.csr_matrix((cited[2], (cited[0], cited[1])))
    lenth = len(class_to_id.keys())
    adj = adj_complete.toarray()
    # adj = torch.from_numpy(adj).long()
    selected_nodes_new_id = [content_d[str(node)]['id'] for node in selected_nodes if str(node) in content_d]
    # graph = adj_convert_to_graph(adj)
    graph = 1

    return graph, labels, lenth, adj, n, selected_nodes_new_id


def convert_sparse_matrix_to_sparse_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    indices_matrix = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    sparse_tensor = tf.SparseTensor(indices=indices_matrix, values=values, dense_shape=shape)
    return tf.cast(sparse_tensor, dtype=tf.float32)
