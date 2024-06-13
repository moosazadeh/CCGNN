import time
import argparse
import pickle
import gc
from model import *
from utils import *
from tqdm import tqdm


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='diginetica/nowplaying/tmall')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--itm_adj_sample', type=int, default=12, help= 'Max number of adjacent nodes')
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--dropout_gcn', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')
parser.add_argument('--dropout_global', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=5)

opt = parser.parse_args()


def main():
    init_seed(2021)

    if opt.dataset == 'diginetica':
        num_item = 43098     #(43097+1)
        num_cat = 996        #(995+1)
        opt.dropout_local = 0.0
        opt.num_layer = 2
    elif opt.dataset == 'nowplaying':
        num_item = 60417     #(60416+1)
        num_cat = 11462      #(11461 + 1)
        opt.dropout_local = 0.0
        opt.num_layer = 2
    elif opt.dataset == 'tmall':
        num_item = 40728     #40727 + 1
        num_cat = 712        #711 + 1
        opt.dropout_local = 0.5
        opt.num_layer = 1
    
    print(opt)
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    category = pickle.load(open('datasets/' + opt.dataset + '/category.txt', 'rb'))
    global_train_adjs = pickle.load(open('datasets/' + opt.dataset +'/train_adjs'+'.txt', 'rb'))
    global_test_adjs = pickle.load(open('datasets/' + opt.dataset +'/test_adjs'+'.txt', 'rb'))
    
    train_data = Data(train_data, category)
    test_data = Data(test_data, category)
    num_total = num_item + num_cat -1

    model = trans_to_cuda(CCGNN(opt, num_item, num_cat, num_total, category))
    start = time.time()
    best_result_k10 = [0, 0]
    best_result_k20 = [0, 0]
    best_epoch_k10 = [0, 0]
    best_epoch_k20 = [0, 0]
    bad_counter_k20 = bad_counter_k10 = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_k10, mrr_k10, hit_k20, mrr_k20 = train_test(model, train_data, test_data, global_train_adjs, global_test_adjs) 
        
        flag_k10 = 0
        if hit_k10 >= best_result_k10[0]:
            best_result_k10[0] = hit_k10
            best_epoch_k10[0] = epoch
            flag_k10 = 1
        if mrr_k10 >= best_result_k10[1]:
            best_result_k10[1] = mrr_k10
            best_epoch_k10[1] = epoch
            flag_k10 = 1            
        print('\n')
        print('Best @10 Result:')
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (
            best_result_k10[0], best_result_k10[1]))
        bad_counter_k10 += 1 - flag_k10
        
        flag_k20 = 0
        if hit_k20 >= best_result_k20[0]:
            best_result_k20[0] = hit_k20
            best_epoch_k20[0] = epoch
            flag_k20 = 1
        if mrr_k20 >= best_result_k20[1]:
            best_result_k20[1] = mrr_k20
            best_epoch_k20[1] = epoch
            flag_k20 = 1
        print('Best @20 Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (
            best_result_k20[0], best_result_k20[1]))
        bad_counter_k20 += 1 - flag_k20
        
        if ((bad_counter_k20 >= opt.patience) and (bad_counter_k10 >= opt.patience)):
            break
    print('-------------------------------------------------------')
    end = time.time()
    print('Run time: ', ((end - start)/60), ' minutes')
    gc.collect()

if __name__ == '__main__':
    main()
