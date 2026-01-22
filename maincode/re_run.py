from utils import load_data
from metrics import  AC, F1
import pandas as pd
import torch
import numpy as np
from model import MyModel
from constants import num_heads
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def total_data():

    path = "boat-data"
    data = load_data(path,device=device)
    graph = data[0]
    labels = [x[1] for x in data[1]]
    n_clusters = data[2]
 
    init_w = torch.randn(1612, 16).float()

    num_community = n_clusters
    n_in_feat = init_w.shape[1]

    model = MyModel(graph, num_heads, num_community, n_in_feat,device=device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    with torch.no_grad(): 
        model(init_w).to(device)
        Z = model.getZ()
        Z_np = np.array(Z)
        pred = Z_np.argmax(1)

    ac = AC(labels, pred)
    f1 = F1(labels, pred)
    print("ac:", ac)
    print("f1:", f1)


if __name__ == "__main__":
    total_data()
