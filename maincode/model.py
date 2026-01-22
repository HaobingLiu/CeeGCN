import torch.nn.functional as F
from layers import *
from FCM import *
from constants import hidden1_dim, hidden2_dim


class MyModel(nn.Module):

    def __init__(self, g, num_heads, num_community, n,device='cuda'):
        super(MyModel, self).__init__()
        self.k = num_community
        self.device=device
        self.GAT_1 = MultiHeadGATLayer(g, n, hidden1_dim, num_heads).to(device)
        self.GAT_2 = MultiHeadGATLayer(g, hidden1_dim, hidden2_dim, num_heads).to(device)
        self.GAT_3 = MultiHeadGATLayer(g, hidden2_dim, n, 1).to(device)

    def forward(self, h):
        h = self.GAT_1(h)
        h = F.elu(h)
        h = self.GAT_2(h)
        h = F.elu(h)
        self.emb = self.GAT_3(h)

        self.z = FCM(self.emb, self.k, 2)

    def getZ(self):
        return self.z[0]

    def get_emb(self):
        return self.emb

    def getW(self):
        return self.GAT_3.getW().to(self.device)