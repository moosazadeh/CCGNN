import pickle
import os
from tqdm import tqdm   # Progress bar
import random
import math
from itertools import repeat


dataset = 'diginetica'      # diginetica/nowplaying/tmall
sess_neigh_sample = 300
itm_adj_sample = 20
max_adj_dist= 10     # Maximum distance of adjacent items (MD)

print('\n-- Loading data --\n')
train = pickle.load(open(dataset + '/train.txt', 'rb'))
test = pickle.load(open(dataset + '/test.txt', 'rb'))    
train_seq = pickle.load(open(dataset + '/all_train_seq.txt', 'rb'))
all_train_seq = train_seq[1]                        # Just take sessions


def build_graph (name, all_sess_idxs, all_sess_items, neighbors):     
    ''' Choose each session and build initial graph based on relations between its items.
                consider an edge between each item and all of the i next and i prev items.'''
    total_sess_adjs = {}
    for sidx in tqdm(all_sess_idxs):        
        edge_count = {}
        sess_items = all_sess_items[sidx]
        for item in sess_items:
            edge_count.update({item : {}})
            
        for dist in range(1, max_adj_dist+1):   # max_adj_dist=1,2,3 
            for pos in range(len(sess_items)):
                u = sess_items[pos]
                if (pos + dist) in range(len(sess_items)):
                    v = sess_items[pos + dist]
                    score = max_adj_dist+3-dist
                    if v in edge_count[u]:  
                        edge_count[u][v] += score     # if the edge[u,v] exists, increase its score
                    else:
                        edge_count[u][v] = score      # else, create an edge with score = max_adj_dist+3-dist                        
                if (pos - dist) in range(len(sess_items)):
                    v = sess_items[pos - dist]
                    score = max_adj_dist+1-dist
                    if v in edge_count[u]:  
                        edge_count[u][v] += score     # if the edge[u,v] exists, increase its score
                    else:
                        edge_count[u][v] = score      # else, create an edge with score = max_adj_dist+1-dist

        ''' Now choose neighbors of this session and add their relations into the initial graph.'''
        nidxs = []
        neigh_idxs = neighbors.get(sidx) 
        nidxs = neigh_idxs[0][:sess_neigh_sample]        # Just take idxs
        for nidx in nidxs:                # neighbor indexes of the current session.
            nitems = all_train_seq[nidx]
            for dist in range(1, max_adj_dist+1):        # max_adj_dist=1,2,3,4 
                for pos in range(len(nitems)):
                    u = nitems[pos]
                    if u in sess_items:                      # If u belongs to current session items
                        if (pos + dist) in range(len(nitems)):
                            v = nitems[pos + dist]
                            score = max_adj_dist+3-dist
                            if v in edge_count[u]:  
                                edge_count[u][v] += score     # if the edge[u,v] exists, increase its score
                            else:
                                edge_count[u][v] = score      # else, create an edge with score = max_adj_dist+3-dist
                        if (pos - dist) in range(len(nitems)):
                            v = nitems[pos - dist]
                            score = max_adj_dist+1-dist
                            if v in edge_count[u]:  
                                edge_count[u][v] += score     # if the edge[u,v] exists, increase its score
                            else:
                                edge_count[u][v] = score      # else, create an edge with score = max_adj_dist+1-dist

        ''' Sort edges based on weights'''
        New_sess_adjs = {}
        for itm in sess_items:  # Choose each item as source node u
            edges = [e for e in sorted(edge_count[itm].items(), reverse=True, key=lambda x: x[1])]
            sess_adj_items = [e[0] for e in edges]  
            sess_adj_weights = [e[1] for e in edges]          
            New_sess_adjs.update({itm : [sess_adj_items[:itm_adj_sample], sess_adj_weights[:itm_adj_sample]]})
        total_sess_adjs.update({sidx : New_sess_adjs})
    
    pickle.dump(total_sess_adjs, open(dataset + '/' + name + '_adjs' + str(sess_neigh_sample) + '-'+str(max_adj_dist)+'.txt', 'wb'))

    return 
    
print('\ndataset =', dataset, ', sess_neigh_sample =', sess_neigh_sample,\
      ', itm_adj_sample =', itm_adj_sample, ', max_adj_dist=', max_adj_dist)
print('\n-- Building train sessions global graph --\n')
neighbors = pickle.load(open(dataset + '/train_neighbors' + str(sess_neigh_sample) + '.txt', 'rb'))
all_sess_items = train[1]
length = len(train[0])
all_sess_idxs = train[0] 
build_graph ('train', all_sess_idxs, all_sess_items, neighbors)

print('\n-- Building test sessions global graph --\n')
neighbors = pickle.load(open(dataset + '/test_neighbors' + str(sess_neigh_sample) + '.txt', 'rb'))
all_sess_items = test[1]
length = len(test[0])
all_sess_idxs = test[0]
build_graph ('test', all_sess_idxs, all_sess_items, neighbors)
