import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

with open('raw/tmall/tmall_data.csv', 'w') as tmall_data:
    with open('raw/tmall/dataset15.csv', 'r') as tmall_file:
        header = tmall_file.readline()
        tmall_data.write(header)
        for line in tmall_file:
            data = line[:-1].split('\t')
            if int(data[2]) > 120000:
                break
            tmall_data.write(line)

print("\n-- Starting --")
with open('raw/tmall/tmall_data.csv', "r") as f:
    reader = csv.DictReader(f, delimiter='\t')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = int(data['SessionId'])
        if curdate and not curid == sessid:
            date = curdate
            sess_date[curid] = date
        curid = sessid
        item = int(data['ItemId'])
        curdate = float(data['Time'])

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = float(data['Time'])
    sess_date[curid] = date

print("\n-- Reading data --")


with open('raw/tmall/Tmall_category.csv',"r") as f:
    reader = csv.DictReader(f)
    item_category = {}
    for data in reader:
        item_id = int(data['item_id'])
        item_category[item_id] = int(data['category_id'])

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
    if len(filseq) < 2 or len(filseq) > 40:
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

# the last of 100 seconds for test
splitdate = maxdate - 100
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
                
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [idx]
        train_dates += [date]
        train_seqs += [outseq]
        idx += 1
    print('\nitem_ctr:', item_ctr)     # 40728

    change = {}                      
    #category_ctr = 1
    category_ctr = 1 + 40727    
    for k,v in new_item_category.items():
        if v in change:
            new_item_category[k] = change[v]
        else:
            change[v] = category_ctr
            new_item_category[k] = change[v]
            category_ctr += 1
    print("\ncategory_ctr:",category_ctr-40728)
    return train_ids, train_seqs, train_dates, new_item_category

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


train_ids, train_seqs, train_dates, new_item_category = obtian_tra()
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

if not os.path.exists('tmall'):
    os.makedirs('tmall')
pickle.dump(processed_train, open('tmall/train.txt', 'wb'))
pickle.dump(processed_test, open('tmall/test.txt', 'wb'))
pickle.dump(all_train_seq, open('tmall/all_train_seq.txt', 'wb'))
pickle.dump(new_item_category,open('tmall/category.txt', 'wb'))
