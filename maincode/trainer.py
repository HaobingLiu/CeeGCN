from losses import total_loss
from metrics import  AC, F1
import torch
from model import MyModel
from constants import num_heads
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


class Trainer():
    def __init__(self,device='cuda'):
        self.device=device
    def initialize_data(self, adj, graph, eta, n_clusters, lr1, clustering_labels, epochs, n):
        self.epochs = epochs
        self.graph = graph.to(self.device)
        self.eta = eta
        self.adj = adj.to(self.device)
        self.clustering_labels = clustering_labels
        self.k = n_clusters

        self.init_w = torch.randn(n, 16).float().to(self.device)

        n_in_feat = self.init_w.shape[1]
        self.model = MyModel(self.graph, num_heads, self.k, n_in_feat,device=self.device).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr1)

    def train_loop(self):
        self.model(self.init_w)
        Z = self.model.getZ()
        F = self.model.get_emb()
        # print(f"Node embedding shape: {F.shape}")
        W = self.model.getW()
        self.graph.edata['a'] = W.detach()
        pred = Z.argmax(1)
        loss = total_loss(self.graph, self.adj, pred, F)
        print(f"Loss: {loss.item()}")

        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()

        self.optimizer.step()

        return pred, self.model


    def train(self):
        best_ac=0
        best_pred_labels = None
        for i in range(self.epochs):
            train = self.train_loop()
            pred_labels_z = train[0]
            M = train[1]
            
            ac = AC(self.clustering_labels, pred_labels_z)
            f1 = F1(self.clustering_labels, pred_labels_z)
          
            if ac > best_ac:
                best_ac = ac
                best_pred_labels = pred_labels_z
                torch.save(M.state_dict(), 'best_model.pth')
            print(f"epoch:{i}")
            print("ac:", ac)
            print("f1:", f1)
       
        self.save_predictions(best_pred_labels)
        return best_pred_labels
    
    def save_predictions(self, pred_labels):

        if torch.is_tensor(pred_labels):
            pred_labels = pred_labels.cpu().numpy()
      
        df = pd.DataFrame({
            'node_id': range(len(pred_labels)),  
            'predicted_label': pred_labels,      
            'true_label': self.clustering_labels 
        })
        
        df.to_csv('cluster_predictions.csv', index=False)
        print("预测标签已保存到 cluster_predictions.csv")