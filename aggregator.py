import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn import Module, Parameter

class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))

    def forward(self, batch_size, itm_emb, ave_sess_itms_emb, glob_adj_itm_embs, glob_adj_wgt_embs):
        ave_sess_itms_emb = ave_sess_itms_emb.unsqueeze(2).repeat(1, 1, glob_adj_itm_embs.shape[2], 1)
        glob_adj_wgt_embs = glob_adj_wgt_embs.unsqueeze(-1)
        
        # Attention coefficients of different adjacent items
        alpha = ave_sess_itms_emb * glob_adj_itm_embs
        alpha = torch.cat([alpha, glob_adj_wgt_embs], -1)
        alpha = torch.matmul(alpha, self.w_1).squeeze(-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = torch.matmul(alpha, self.w_2).squeeze(-1) 
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        
        # Attentive adjacent item embeddings (session aware attention)
        att_adj_itms_embs = torch.sum(alpha * glob_adj_itm_embs, dim=-2)
        
        # Aggregate the session item embeddings and corresponding adjacent item embeddings 
        global_itm_emb = torch.cat([itm_emb, att_adj_itms_embs], -1)
        global_itm_emb = F.dropout(global_itm_emb, self.dropout, training=self.training)
        global_itm_emb = torch.matmul(global_itm_emb, self.w_3)
        global_itm_emb = global_itm_emb.view(batch_size, -1, self.dim)
        global_itm_emb = torch.relu(global_itm_emb)
        return global_itm_emb


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        self.linear_in1= nn.Linear(self.gate_size, self.gate_size, bias=True)
        self.linear_in2= nn.Linear(self.gate_size, self.gate_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        
        test = torch.cat([inputs,hidden], 2)
        test1 = self.linear_in1(test)
        test2 = self.linear_in2(test)
        
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        
        gi = gi+test1
        gh = gh+test2
        
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden