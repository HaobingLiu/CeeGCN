import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import alpha_entmax
import dgl.function as fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)


    def edge_attention(self, edges):
        
        batch_size=32
        z_src=edges.src['z']
        z_dst=edges.dst['z']
        z2_batches=[]
        
        for i in range(0,len(z_src),batch_size):
            # print(f"z_src shape: {z_src[i:i+batch_size].shape}")
            # print(f"z_dst shape: {z_dst[i:i+batch_size].shape}")
            z2_batches.append(torch.cat([z_src[i:i+batch_size],z_dst[i:i+batch_size]],dim=1))
        z2=torch.cat(z2_batches,dim=0)
        #z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        e = self.attn_fc(z2)
        e = F.leaky_relu(e)
        e = e + edges.data['w']
        return {'e': e}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        if not hasattr(self, 'printed_shape'):
            print("nodes.mailbox['e'] shape:", nodes.mailbox['e'].shape)
            self.printed_shape = True
        alpha = alpha_entmax(nodes.mailbox['e'], alpha = 1.5, dim = 1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads,device='cuda'):
        super(MultiHeadGATLayer, self).__init__()
        self.device=device
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))

        self.weights = nn.Parameter(torch.ones(num_heads))


    def forward(self, h):
        h=h.to(self.device)
        head_outs = [attn_head(h) for attn_head in self.heads]
        weights_softmax = torch.softmax(self.weights, dim=0)
        weighted_sum = torch.stack(head_outs, dim=-1) * weights_softmax
        output = torch.sum(weighted_sum, dim=-1)

        alpha_out = torch.stack([head.g.edata['e'] for head in self.heads], dim=-1)
        alpha_weighted_sum = alpha_out + weights_softmax
        self.alpha_output = torch.mean(alpha_weighted_sum, dim=-1)

        return output

    def getW(self):
        return self.alpha_output


