import argparse
import time
import csv
import pickle
import operator
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

if opt.dataset == 'diginetica':
    file = 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    file = 'yoochoose-clicks.dat'
elif opt.dataset =='sample':
    file = 'train-item-views.csv'
    
print("\n-- Starting --")
with open('raw/' + opt.dataset + '/' + file, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0 
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['sessionId']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['itemId'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    sess_date[curid] = date
    
with open('raw/' + opt.dataset + '/product-categories.csv',"r") as f:          
    reader = csv.DictReader(f,delimiter=';')
    item_category = {}
    for data in reader:
        item_id = data['itemId']
        item_category[item_id] = data['categoryId']

with open('raw/' + opt.dataset + '/products.csv',"r") as f:          
    reader = csv.DictReader(f,delimiter=';')
    item_price = {}
    for data in reader:
        item_id = data['itemId']
        item_price[item_id] = data['pricelog2']
        
print("\n-- Reading data --")

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

print("\n-- Splitting train set and test set --")
# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

if opt.dataset == 'sample':
    splitdate = maxdate - 86400 * 1
    traindate = maxdate - 86400 * 3     # 3 days to train
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tra_sess = filter(lambda x: x[1] > traindate, tra_sess)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)
else:    
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print('\nOriginal Train sessions:', len(tra_sess))    # 186670    # 7966257
print('\nOriginal Test sessions:', len(tes_sess))    # 15979     # 15324


# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
new_item_category = {}
new_item_price = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    idx = 0
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                new_item_category[item_ctr] = item_category[i] 
                new_item_price[item_ctr] = item_price[i]
                
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [idx]
        train_dates += [date]
        train_seqs += [outseq]
        idx += 1
    print('\nitem_ctr:', item_ctr)     # 43098, 37484

    change = {}                      
    #category_ctr = 1
    category_ctr = 1 + 43097    
    for k,v in new_item_category.items():
        if v in change:
            new_item_category[k] = change[v]
        else:
            change[v] = category_ctr
            new_item_category[k] = change[v]
            category_ctr += 1
    print("\ncategory_ctr:",category_ctr-43098)
    return train_ids, train_seqs, train_dates, new_item_category, new_item_price

# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    idx = 0
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [idx]
        test_dates += [date]
        test_seqs += [outseq]
        idx += 1
    return test_ids, test_seqs, test_dates

def process_seqs(iseqs, idates):
    idx = 0
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for seq, date in zip(iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [idx]
            idx += 1
    return ids, out_seqs, labs, out_dates


train_ids, train_seqs, train_dates, new_item_category, new_item_price = obtian_tra()
test_ids, test_seqs, test_dates = obtian_tes()
all_train_seq = (train_ids, train_seqs, ['Nothing'], train_dates)
all = 0
for seq in train_seqs:
    all += len(seq)
for seq in test_seqs:
    all += len(seq)
print('\nAvg session length: ', all/(len(train_seqs) + len(test_seqs) * 1.0))


tr_ids, tr_seqs, tr_labs, tr_dates = process_seqs(train_seqs, train_dates)
te_ids, te_seqs, te_labs, te_dates = process_seqs(test_seqs, test_dates)
processed_train = (tr_ids, tr_seqs, tr_labs, tr_dates)
processed_test = (te_ids, te_seqs, te_labs, te_dates)
print('\nTrain sessions:', len(tr_seqs))
print('\nTest sessions:', len(te_seqs))


if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(processed_train, open('diginetica/train.txt', 'wb'))
    pickle.dump(processed_test, open('diginetica/test.txt', 'wb'))
    pickle.dump(all_train_seq, open('diginetica/all_train_seq.txt', 'wb'))
    pickle.dump(new_item_category,open('diginetica/category.txt', 'wb'))
    pickle.dump(new_item_price,open('diginetica/price.txt', 'wb'))

    
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(processed_test, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(processed_test, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = train_seqs[tr_ids[-split4]:], train_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))


else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(processed_train, open('sample/train.txt', 'wb'))
    pickle.dump(processed_test, open('sample/test.txt', 'wb'))
    pickle.dump(all_train_seq, open('sample/all_train_seq.txt', 'wb'))
    pickle.dump(new_item_category,open('sample/category.txt', 'wb'))
    pickle.dump(new_item_price,open('sample/price.txt', 'wb'))