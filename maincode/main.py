from subgraph_utils import load_data
from sampling_graph import *
from merge_subgraph import *
import torch
import pandas as pd

def shai(node, labels):
    
    data = []

    
    for label in labels:
        if label[0] in node:
            data.append({'node_id': label[0], 'label': label[1]})

    
    df = pd.DataFrame(data)

    
    csv_file_path = 'node_labels.csv'
    df.to_csv(csv_file_path, index=False)
    print(f'节点标签已保存为 {csv_file_path}')

if __name__ == '__main__':
    
    data = load_data()
    graph = data[0]
    labels = data[1]
    n_clusters = data[2]
    adj = data[3]
    n_nodes = data[4]
    selected_nodes = data[5]

    print("selected_nodes长度", len(selected_nodes), selected_nodes[:10])
    
   
    if isinstance(adj, torch.Tensor):
        nonzero_count = int(torch.count_nonzero(adj))
    else:
        
        import numpy as np
        adj_np = adj.toarray() if hasattr(adj, "toarray") else np.array(adj)
        nonzero_count = int(np.count_nonzero(adj_np))
    print("adj 非零元素个数：", nonzero_count)
    if nonzero_count == 0:
        print("邻接矩阵 adj 全是 0,说明没有任何边,后续子图必然没边。")

    x = torch.randn(n_nodes, 16)
    n_order = 2  
    subgraph_extractor = Subgraph(adj, x)
    subgraph_extractor.build(selected_nodes,sc=0.000085)
    batch, index = subgraph_extractor.search(selected_nodes)
    node = merge()
    print("边信息已合并并保存到edge_data.csv")
    shai(node,labels)



