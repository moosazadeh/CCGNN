import numpy as np
import torch
from torch.utils.data import Dataset

def process_adj (global_adjs, bch_sess_idxs, bch_suitems, bch_sess_len, itm_adj_sample):
    """ Create two matrices that include adjacents of each item and corrosponding weights. """    
    bch_global_adj_item_mat = []
    bch_global_adj_weight_mat = []
    
    for sidx, sess_uitems in zip(bch_sess_idxs, bch_suitems):
        sess_adjs = global_adjs[sidx]
        for itm in sess_uitems:
            if itm == 0:
                bch_global_adj_item_mat.append([0]*itm_adj_sample)
                bch_global_adj_weight_mat.append([0]*itm_adj_sample)
            else:
                sess_adj_itm = sess_adjs[itm][0][:itm_adj_sample]
                sess_adj_weight = sess_adjs[itm][1][:itm_adj_sample]
                num_adj = len(sess_adj_itm)
                if num_adj < itm_adj_sample: 
                    sess_adj_itm = sess_adj_itm + (itm_adj_sample - num_adj) * [0]
                    sess_adj_weight = sess_adj_weight + (itm_adj_sample - num_adj) * [0]
                    
                bch_global_adj_item_mat.append(sess_adj_itm)
                bch_global_adj_weight_mat.append(sess_adj_weight)

    return bch_global_adj_item_mat, bch_global_adj_weight_mat

def process_data(all_sess_itms):
    all_sess_lens = [len(sess_itms) for sess_itms in all_sess_itms]
    max_len = max(all_sess_lens)
        
    # To compute the position attention, reverse each session.
    all_rev_sess_itms = [list(reversed(sess_itms)) + [0] * (max_len - sess_len) if sess_len < max_len else list(reversed(sess_itms[-max_len:]))
               for sess_itms, sess_len in zip(all_sess_itms, all_sess_lens)]

    # Create mask of each session
    mask = [[1] * sess_len + [0] * (max_len - sess_len) if sess_len < max_len else [1] * max_len
               for sess_len in all_sess_lens]
    return all_rev_sess_itms, mask, max_len

def process_cats(category, rev_sess_itms):  
    rev_sess_cats = []
    for item in rev_sess_itms:
       if item == 0:
          rev_sess_cats += [0]
       else:
          rev_sess_cats += [category[item]]
    return rev_sess_cats

class Data(Dataset):
    def __init__(self, data, category):
        all_rev_sess_itms, mask, max_len = process_data(data[1])
        self.category = category
        self.all_rev_sess_itms = np.asarray(all_rev_sess_itms)
        self.targets = np.asarray(data[2])
        self.mask = np.asarray(mask)
        self.length = len(data[1])
        self.max_len = max_len

    def __getitem__(self, sess_idxs):
        rev_sess_itms, mask, target = self.all_rev_sess_itms[sess_idxs], self.mask[sess_idxs], self.targets[sess_idxs]
        rev_sess_cats = process_cats(self.category, rev_sess_itms)
        rev_sess_nods = np.append(rev_sess_itms, rev_sess_cats)
        rev_sess_nods = rev_sess_nods[rev_sess_nods > 0]

        max_n_itm = self.max_len
        urev_itms = np.unique(rev_sess_itms)
        urev_cats = np.unique(rev_sess_cats)
        urev_nods = np.unique(rev_sess_nods)
        if len(urev_nods)<max_n_itm*2:
          urev_nods= np.append(urev_nods,0)
          
        urev_sess_itms = urev_itms.tolist() + (max_n_itm - len(urev_itms)) * [0]
        urev_sess_cats = urev_cats.tolist()  + (max_n_itm - len(urev_cats)) * [0]
        urev_sess_nods = urev_nods.tolist() + (max_n_itm*2 - len(urev_nods)) * [0]
        
        adj_itms = np.zeros((max_n_itm, max_n_itm))
        adj_cats = np.zeros((max_n_itm, max_n_itm))
        adj_nods = np.zeros((max_n_itm*2, max_n_itm*2))


        # Fill adj_itms based on item to item edges.       
        # Define different types of edges by 1, 2, 3, and 4        
        for i in np.arange(len(rev_sess_itms) - 1):
            u = np.where(urev_itms == rev_sess_itms[i])[0][0]
            adj_itms[u][u] = 1
            if rev_sess_itms[i + 1] == 0:
                break
            v = np.where(urev_itms == rev_sess_itms[i + 1])[0][0]
            if u == v or adj_itms[u][v] == 4:
                continue
            adj_itms[v][v] = 1
            if adj_itms[v][u] == 2:
                adj_itms[u][v] = 4
                adj_itms[v][u] = 4
            else:
                adj_itms[u][v] = 2
                adj_itms[v][u] = 3
        # alias_rev_sess_itms represents the relative position of the items clicked sequentially
                    # in the "item" list of this session.
        alias_rev_sess_itms = [np.where(urev_itms == i)[0][0] for i in rev_sess_itms]
        
        
        # Fill adj_nods based on cat to item edges and item to cat edges.       
        # Define different types of edges by 1, 2, 3, and 4        
        for i in np.arange(len(rev_sess_itms) - 1):
            u = np.where(urev_nods == rev_sess_itms[i])[0][0]
            c = np.where(urev_nods == self.category[rev_sess_itms[i]])[0][0]
            adj_nods[u][u] = 1
            adj_nods[c][c] = 4
            adj_nods[u][c]= 2
            adj_nods[c][u]= 3
            if rev_sess_itms[i + 1] == 0:
                break          
            u2 = np.where(urev_nods == rev_sess_itms[i + 1])[0][0]
            c2 = np.where(urev_nods == self.category[rev_sess_itms[i + 1]])[0][0]
            adj_nods[u][u2] = 1
            adj_nods[u2][u] = 1
            
            adj_nods[c][c2] = 4
            adj_nods[c2][c] = 4
        alias_cat_itms = [np.where(urev_nods == i)[0][0] for i in rev_sess_itms]
        alias_itm_cats = [np.where(urev_nods == i)[0][0] for i in rev_sess_cats]
        
        
        # Fill adj_cats based on cat to cat edges
        for i in np.arange(len(rev_sess_cats) - 1):     # from 0 to n-1
            if rev_sess_cats[i + 1] == 0:
                break
            u = np.where(urev_cats == rev_sess_cats[i])[0][0]
            v = np.where(urev_cats == rev_sess_cats[i + 1])[0][0]
            adj_cats[u][v] += 1    # Capture the number of repetitions of this edge
            
        u_sum_in = np.sum(adj_cats, 0)                  # Calculate in-degree of each category
        u_sum_in[np.where(u_sum_in == 0)] = 1           # to prevent divide by zero
        adj_cats_in = np.divide(adj_cats, u_sum_in)     # Divide the number of repetitions of each category by its in-degree
        
        u_sum_out = np.sum(adj_cats, 1)                 # Calculate out-degree of each category
        u_sum_out[np.where(u_sum_out == 0)] = 1         # to prevent divide by zero
        adj_cats_out = np.divide(adj_cats.transpose(), u_sum_out) # Divide the number of repetitions of each category by its out-degree
        
        adj_cats = np.concatenate([adj_cats_in, adj_cats_out]).transpose()
        # alias_rev_sess_cats represents the relative position of the categories clicked sequentially 
                    # in the "category" list of this session.
        alias_rev_sess_cats = [np.where(urev_cats == i)[0][0] for i in rev_sess_cats]
        
        
        return [torch.tensor(sess_idxs), torch.tensor(alias_rev_sess_itms), torch.tensor(adj_itms), torch.tensor(urev_sess_itms),
                torch.tensor(mask), torch.tensor(target), torch.tensor(rev_sess_itms),
                torch.tensor(alias_rev_sess_cats), torch.tensor(adj_cats), torch.tensor(urev_sess_cats),
                torch.tensor(alias_cat_itms),torch.tensor(alias_itm_cats),torch.tensor(adj_nods),torch.tensor(urev_sess_nods)]

    def __len__(self):
        return self.length
