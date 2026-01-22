import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def merge():
    subgraph_file_path = './subgraph_data_subgraph'
    loaded_subgraphs = torch.load(subgraph_file_path)

    edge_index_list = []
    edge_attr_list = []

    for data in loaded_subgraphs.values():
        edge_index_list.append(data.edge_index)
        edge_attr_list.append(data.edge_attr)

    edge_index = torch.cat(edge_index_list, dim=1).numpy()
    edge_attr = torch.cat(edge_attr_list, dim=0).numpy().astype(int)
    node_ids = np.unique(edge_index)
  
    edge_data = {
        'node1': edge_index[0],
        'node2': edge_index[1],
        'weight': edge_attr
    }
    edge_df = pd.DataFrame(edge_data)
    
    csv_file_path = 'edge_data.csv'
    edge_df.to_csv(csv_file_path, index=False)

    return node_ids
