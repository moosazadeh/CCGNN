import argparse
import pickle
import os

import numpy as np
import math
import pandas as pd
from tqdm import tqdm   # Progress bar


dataset = 'diginetica'      # diginetica/nowplaying/tmall
sample_num = 150       
ipl = 2         # Item Position Lambda
icl = 1.5       # Item Category Lambda
stl = 5         # Session Time Lambda => a week


print('\n\n-- Loading data --\n')
category = pickle.load(open(dataset + '/category.txt', 'rb'))
all_train_seq = pickle.load(open(dataset + '/all_train_seq.txt', 'rb'))
all_train_seq_items = all_train_seq[1]
all_train_seq_times = all_train_seq[3]          

train = pickle.load(open(dataset + '/train.txt', 'rb'))
test = pickle.load(open(dataset + '/test.txt', 'rb'))

# Collect index of all sessions in the train set that include the given item
item_sess_map = {}
for neigh_idx, neigh_items in zip(all_train_seq[0], all_train_seq[1]):
    for itm in neigh_items:
        if itm not in item_sess_map.keys():
            item_sess_map[itm] = []
        item_sess_map[itm].append(neigh_idx)
        


def cosine_similarity(neigh_items, sess_items, item_pos_weight):
    ''' Calculate cosine similarity for two sessions based on position weight of items'''
    neigh_length = len(neigh_items)
    intersection = neigh_items & sess_items
    intersection_pos_sum = 0
    s = 0
    for itm in neigh_items:
        s += item_pos_weight[itm] * item_pos_weight[itm]
        if itm in intersection:
            intersection_pos_sum += item_pos_weight[itm]

    similarity = intersection_pos_sum / (math.sqrt(s) * math.sqrt(neigh_length))
    # similarity = np.around(similarity, 4)  # digits = 4
    return similarity

def find_nearest_neighbors(sess_items, sess_time):
    ''' At first, find all sessions that include target items and have occurred before the current session '''
    sess_neigh_idxs = []
    usess_items = np.unique(sess_items)
    for itm in usess_items:
        poss_neigh_idxs = item_sess_map[itm]    # Select all sessions that contain this item.
        poss_neigh_idxs = [pnidx for pnidx in poss_neigh_idxs if all_train_seq_times[pnidx] < sess_time]      # Just keep previous neighbors.

        if len(poss_neigh_idxs) <= sample_num:
            itm_knn = poss_neigh_idxs                     # Item K nearest neighbors
        else:
            neigh_indexs = []
            neigh_sims = []
            item_pos = sess_items.index(itm)     
            for nidx in poss_neigh_idxs:                  # Select each neighbor
                neigh_items = all_train_seq_items[nidx]   # Specify neighbor items.
                
                ''' Factor-1: Items that are closer to target item are assigned higher weightage. '''                    
                item_weight = {}             
                for nitm in neigh_items:
                    nitm_pos = neigh_items.index(nitm)                  
                    pos_diff = abs(nitm_pos - item_pos)
                    item_weight[nitm] =  math.exp(-pos_diff / ipl)
                    
                    ''' Factor-2: Items with same category as the target item are assigned higher weightage. '''
                    if category[itm] == category[nitm]:
                        item_weight[nitm] = item_weight[nitm] * icl
                    
                # Calculate Cosine similarity by considering weight of items.
                similarity = cosine_similarity(set(neigh_items), set(sess_items), item_weight)
        
                ''' Factor-3: Sessions closer to current_session are assigned higher weightage'''
                neigh_time = all_train_seq_times[nidx]
                sess_time_diff = abs(sess_time - neigh_time)
                decay = math.exp(-sess_time_diff / (stl * 24*60*60))    # Transform to day
                similarity *= decay

                neigh_indexs.append(nidx)
                neigh_sims.append(similarity)
                
            nidxs = np.array(neigh_indexs)
            nsims = np.array(neigh_sims)
            sorted_idcs = np.argsort(nsims)[::-1]       # Sort similarity score in descending order.  
            itm_knn = nidxs[sorted_idcs][:sample_num]   # Sort neigh_indexs based on sorted_idcs.

        sess_neigh_idxs.append(itm_knn)
    
    return sess_neigh_idxs

def get_neighbors(name, data, length): 
    ''' For each session, find its neighbors and similarities.'''
    neigh_data_map = {}
    for sess_idx in tqdm(data[0]):
        sess_items = data[1][sess_idx]
        sess_times = data[3][sess_idx]
        sess_itm_neigh = find_nearest_neighbors(sess_items, sess_times)
        neigh_data_map.update({sess_idx : sess_itm_neigh})
    pickle.dump(neigh_data_map, open(dataset + '/' + name + '_neighbors' + str(sample_num) + '.txt', 'wb'))
    return

print('\ndataset =' , dataset, ', sample_num =', sample_num)
print('\n-- Find neighbors of all sessions --\n')
length = len(train[0])    
neigh_data_map = get_neighbors('train', train, length)

length = len(test[0]) 
neigh_data_map = get_neighbors('test', test, length)

