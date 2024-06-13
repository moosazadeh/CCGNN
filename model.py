import datetime
import math
import numpy as np
import torch
from utils import process_adj
import pickle
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator, GNN
from torch.nn import Module, Parameter, Linear
import torch.nn.functional as F


class CCGNN(Module):
    def __init__(self, opt, num_item, num_cat, num_total, category):
        super(CCGNN, self).__init__()
        self.opt = opt
        self.dataset = opt.dataset
        self.batch_size = opt.batch_size
        self.dim = opt.hiddenSize
        self.itm_adj_sample = opt.itm_adj_sample
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.num_layer = opt.num_layer

        # self.num_item = num_item
        self.num_cat = num_cat
        self.num_total = num_total
        self.category = category
        
        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.gnn = GNN(self.dim)
        self.local_agg_node = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.global_agg = GlobalAggregator(self.dim, opt.dropout_gcn)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_total, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.a1 = Parameter(torch.Tensor(1))
        self.a2 = Parameter(torch.Tensor(1))
        self.a3 = Parameter(torch.Tensor(1))

        self.w_1 = nn.Parameter(torch.Tensor(2*self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(2*self.dim, self.dim))
        self.w_3 = nn.Parameter(torch.Tensor(3 * self.dim, 2*self.dim))        

        self.q1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.q2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.q3 = nn.Parameter(torch.Tensor(2*self.dim, 1))

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim)
        self.glu4 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu5 = nn.Linear(2*self.dim, 2*self.dim)
        self.glu6 = nn.Linear(2*self.dim, 2*self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        
        all_cats = []
        for c in range(1, num_total-num_cat+1):
            all_cats += [category[c]]
        all_cats = np.asarray(all_cats)  
        self.all_cats =  trans_to_cuda(torch.Tensor(all_cats).long())

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, glob_itm_emb, locl_itm_emb, locl_cat_emb, locl_cat_itm_emb, locl_itm_cat_emb, mask):
        ''' Compute predicted probability tensor (scores) of all items. '''
        mask = mask.float().unsqueeze(-1)
        batch_size = locl_itm_emb.shape[0]
        len = locl_itm_emb.shape[1]
        
        if self.dataset == 'diginetica':
            itm_emb = locl_itm_emb + locl_cat_itm_emb * self.a1 + glob_itm_emb * self.a3
            cat_emb = locl_cat_emb + locl_itm_cat_emb * self.a2
        else:
            # Item embedding
            locl_itm_emb = locl_itm_emb + locl_cat_itm_emb * self.a1
            x = torch.cat([locl_itm_emb, glob_itm_emb], -1)
            x = torch.matmul(x, self.w_1)
            x = torch.tanh(x)
            # Attentive item embedding
            gamma = torch.sigmoid(self.glu1(x) + self.glu2(locl_itm_emb))
            gamma = torch.matmul(gamma, self.q1)
            gamma = gamma * mask
            itm_emb = gamma * locl_itm_emb
    
            # Category embedding
            locl_cat_emb = locl_cat_emb + locl_itm_cat_emb * self.a2
            y = torch.cat([locl_cat_emb, glob_itm_emb], -1)
            y = torch.matmul(y, self.w_2)
            y = torch.tanh(y)
            # Attentive category embedding
            delta = torch.sigmoid(self.glu3(y) + self.glu4(locl_cat_emb))
            delta = torch.matmul(delta, self.q2)
            delta = delta * mask
            cat_emb = delta * locl_cat_emb
        
        # Collaborative embedding
        coll_emb = torch.cat([itm_emb, cat_emb],-1)
        # Collaborative session embedding
        coll_sess_emb = torch.sum(coll_emb * mask, -2) / torch.sum(mask, 1)
        coll_sess_emb = coll_sess_emb.unsqueeze(-2).repeat(1, len, 1)
        
        # Position embedding 
        pos_emb = self.pos_embedding.weight[:len]          
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # Position aware collaborative embedding
        z = torch.cat([pos_emb, coll_emb], -1)
        z = torch.matmul(z, self.w_3)
        z = torch.tanh(z)

        # Attention coefficients
        beta = torch.sigmoid(self.glu5(z) + self.glu6(coll_sess_emb))
        beta = torch.matmul(beta, self.q3)
        beta = beta * mask

        # Collaborative attentive session embedding
        coll_att_sess_emb = torch.sum(beta * coll_emb, 1)   

        # Concatenate the embedding of all items and their corresponding categories
        all_itm_emb = self.embedding.weight[1: self.num_total-self.num_cat+1]
        all_cat_emb = self.embedding(self.all_cats)
        all_itm_cat_emb = torch.cat([all_itm_emb, all_cat_emb],-1)

        scores = torch.matmul(coll_att_sess_emb, all_itm_cat_emb.transpose(1, 0))
        return scores

    def forward(self, sess_idxs, global_adjs, urev_sess_itms, local_adj_itms, mask, rev_sess_itms,
                                            urev_sess_cats, local_adj_cats, urev_sess_nods, local_adj_nods):

        """ Global Embedding """
        batch_size = urev_sess_itms.shape[0]  # Sometimes the last batch is shorter than others!
        seqs_len = urev_sess_itms.shape[1]

        # Build adjacent graph based on each session item and corresponding neighbor session items.
        sess_items = trans_to_cpu(urev_sess_itms).numpy()
        smpl_adj_itms, smpl_adj_wgts = process_adj(global_adjs, sess_idxs, sess_items, seqs_len, self.itm_adj_sample)
        smpl_adj_itms = trans_to_cuda(torch.Tensor(smpl_adj_itms)).long()
        smpl_adj_wgts = trans_to_cuda(torch.Tensor(smpl_adj_wgts)).float() 

        # Embedding
        global_adj_itms = smpl_adj_itms.view(batch_size, (seqs_len * self.itm_adj_sample))   # Flatten
        global_adj_itms_emb = self.embedding(global_adj_itms)
        glob_adj_itm_embs = global_adj_itms_emb.view(batch_size, -1, self.itm_adj_sample, self.dim)

        global_adj_wgts = smpl_adj_wgts.view(batch_size, (seqs_len * self.itm_adj_sample))   # Flatten
        global_adj_wgts = global_adj_wgts.view(batch_size, -1, self.itm_adj_sample)

        # Average of item embeddings of the current session
        usess_itm_emb = self.embedding(urev_sess_itms)
        sess_itms_emb = self.embedding(rev_sess_itms) * mask.float().unsqueeze(-1) 
        ave_sess_itms_emb = torch.sum(sess_itms_emb, 1) / torch.sum(mask.float(), -1).unsqueeze(-1)
        ave_sess_itms_emb = ave_sess_itms_emb.unsqueeze(-2)
        ave_sess_itms_emb = ave_sess_itms_emb.repeat(1, usess_itm_emb.shape[1], 1)
        
        # Aggregate embeddings of the session items and its adjacent items. 
        global_itm_emb = self.global_agg(batch_size, usess_itm_emb, ave_sess_itms_emb, glob_adj_itm_embs, global_adj_wgts)
        global_itm_emb = global_itm_emb.view(batch_size, seqs_len, self.dim)
        global_itm_emb = F.dropout(global_itm_emb, self.dropout_global, training=self.training)

        """ Local Embedding """ 
        local_itm_emb = self.embedding(urev_sess_itms)
        local_cat_emb = self.embedding(urev_sess_cats)
        local_node_emb = self.embedding(urev_sess_nods)
        
        # Local aggregation
        local_itm_emb = self.local_agg(local_itm_emb, local_adj_itms)
        local_cat_emb = self.gnn(local_adj_cats, local_cat_emb)
        for l in range(self.num_layer):
            local_node_emb = self.local_agg_node(local_node_emb, local_adj_nods)
            
        # Dropout
        local_itm_emb = F.dropout(local_itm_emb, self.dropout_local, training=self.training)
        local_cat_emb = F.dropout(local_cat_emb, self.dropout_local, training=self.training)
        local_node_emb = F.dropout(local_node_emb, self.dropout_local, training=self.training)

        return global_itm_emb, local_itm_emb, local_cat_emb, local_node_emb
    
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, bch_data, global_adjs):
            
    (sess_idxs, alias_rev_sess_itms, local_adj_itms, urev_sess_itms, 
      mask, targets, rev_sess_itms, 
      alias_rev_sess_cats, local_adj_cats, urev_sess_cats, 
    alias_cat_itms, alias_itm_cats, local_adj_nods, urev_sess_nods)= bch_data
    
    sess_idxs = sess_idxs.numpy()
    urev_sess_itms = trans_to_cuda(urev_sess_itms).long()
    mask = trans_to_cuda(mask).long()

    rev_sess_itms = trans_to_cuda(rev_sess_itms).long()
    alias_rev_sess_cats = trans_to_cuda(alias_rev_sess_cats).long()
    urev_sess_cats = trans_to_cuda(urev_sess_cats).long()

    local_adj_cats = trans_to_cuda(local_adj_cats).float()

    alias_rev_sess_itms = trans_to_cuda(alias_rev_sess_itms).long()
    local_adj_itms = trans_to_cuda(local_adj_itms).float()

    alias_cat_itms = trans_to_cuda( alias_cat_itms).long()
    alias_itm_cats = trans_to_cuda(alias_itm_cats).long()
    local_adj_nods = trans_to_cuda(local_adj_nods).float()
    urev_sess_nods = trans_to_cuda(urev_sess_nods).long()

    global_itm_emb, local_itm_emb, local_cat_emb, local_node_emb = model(sess_idxs, global_adjs, urev_sess_itms,
                            local_adj_itms, mask, rev_sess_itms, urev_sess_cats, local_adj_cats , urev_sess_nods, local_adj_nods)
    
    # alias_rev_sess_itms represents the relative position of the items clicked sequentially in each session in the "item" list of this session.
    get1 = lambda i: global_itm_emb[i][alias_rev_sess_itms[i]]
    glob_itm_emb = torch.stack([get1(i) for i in torch.arange(len(alias_rev_sess_itms)).long()])

    get2 = lambda i: local_itm_emb[i][alias_rev_sess_itms[i]]      
    locl_itm_emb = torch.stack([get2(i) for i in torch.arange(len(alias_rev_sess_itms)).long()])

    get3 = lambda i: local_cat_emb[i][alias_rev_sess_cats[i]] 
    locl_cat_emb = torch.stack([get3(i) for i in torch.arange(len(alias_rev_sess_cats)).long()])
    
    get1_mix = lambda i: local_node_emb[i][alias_cat_itms[i]] 
    locl_cat_itm_emb = torch.stack([get1_mix(i) for i in torch.arange(len(alias_cat_itms)).long()])
    
    get2_mix = lambda i: local_node_emb[i][alias_itm_cats[i]]
    locl_itm_cat_emb = torch.stack([get2_mix(i) for i in torch.arange(len(alias_itm_cats)).long()])
    
    scores = model.compute_scores(glob_itm_emb, locl_itm_emb, locl_cat_emb, locl_cat_itm_emb, locl_itm_cat_emb, mask)

    return targets, scores

def train_test(model, train_data, test_data, global_train_adjs, global_test_adjs):

    print('.... Training Step ....')
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=model.batch_size, shuffle=True, pin_memory=True)
    for bch_data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, bch_data, global_train_adjs)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    
    print('.... Evaluating Step ....')    
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size, shuffle=False, pin_memory=True)
    result = []
    hit_k10, mrr_k10, hit_k20, mrr_k20 = [], [], [], []
    
    for bch_data in tqdm(test_loader):
        targets, scores = forward(model, bch_data, global_test_adjs)
        sub_scores_k20 = scores.topk(20)[1]
        sub_scores_k20 = trans_to_cpu(sub_scores_k20).detach().numpy()
        sub_scores_k10 = scores.topk(10)[1]
        sub_scores_k10 = trans_to_cpu(sub_scores_k10).detach().numpy()
        targets = targets.numpy()
        
        for score, target, mask in zip(sub_scores_k20, targets, test_data.mask):
            hit_k20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k20.append(0)
            else:
                mrr_k20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k10, targets, test_data.mask):
            hit_k10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k10.append(0)
            else:
                mrr_k10.append(1 / (np.where(score == target - 1)[0][0] + 1))
    
    result.append(np.mean(hit_k10) * 100)
    result.append(np.mean(mrr_k10) * 100)
    result.append(np.mean(hit_k20) * 100)
    result.append(np.mean(mrr_k20) * 100)  

    return result