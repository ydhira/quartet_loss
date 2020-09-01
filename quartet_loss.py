import os, sys 
import numpy as np
from random import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class QuartetLoss(nn.Module):
    def __init__(self):
        super(QuartetLoss, self).__init__()
        self.loss = 0.0

    #@staticmethod
    def forward(self, features, bs, featureD, N):
        """
        Args:
            bs (int) : batch size
            features (torch.cuda.FloatTensor): size (bs x feat_dim) 
            featureD (int): dimension of the features  
            N (int): Num of pairs in the batch. 
        Returns:
            float: average quartet loss on the features. 
        """

        # feature consists of [same_s_id, same_s_id, ... ]
        # calculate distance for every same/cross speaker pairs
        pair_num = bs / 2 # num of pairs that are in the mini-batch.
        m = 40 # reppresents the numbers of negative pairs tried before finding the worst negative pair. Refer to the paper. 

        # cumulate loss over all same/cross pair combination
        # most similar cross speaker pair
        for i in range(0, N):
            ablist = []
            for j in range(m):
                a = np.random.randint(0,bs)
                b = np.random.randint(0,bs)
                if (b < N):
                    while (b==a or b==(a+1)):
                        b = np.random.randint(0,bs)
                else:
                    while (b==a):
                        b = np.random.randint(0,bs)
                ablist.append((a,b))

            max_dist = [-10**6]
            max_idx = m-1
            for k in range(len(ablist)):
                x_pair = ablist[k]
                pair_1, pair_2 = x_pair[0], x_pair[1]
                dist = F.cosine_similarity(features[pair_1].view(-1,featureD), features[pair_2].view(-1,featureD))

                if (dist.cpu().data.numpy()[0] > max_dist):
                    max_idx = k
                    max_dist = dist.cpu().data.numpy()[0]

            self.loss += F.sigmoid(((F.cosine_similarity(features[ablist[max_idx][0]].view(-1,featureD),\
                                                           features[ablist[max_idx][1]].view(-1,featureD))))\
                                    - ((F.cosine_similarity(features[i].view(-1,featureD),\
                                                           features[i+N].view(-1,featureD)))))
        self.loss = self.loss / (N)
        return self.loss.view(1,-1)

# Note: You would need to implement the parse function according to your 
# dataset. 
# generate speaker-utterance list from training data
def parse_data(data_dir):
    '''
        Args:
            data_dir (str): path to the director containing features. 
        Returns:
            spk_dict:  mapping from spk to the list of spk files e.g {'id1: [utt1, utt2], 'id2': [utt1]}
            spk_count: {id_{i}: 0} (used in the batch generation function)
            spk_order_all: list of all speakers 
            spk_order: speakers with more than one sample
            total_files: total num of files
    '''
    spk2utt_dict = {}
    spk_count = {}
    spk_order_same = []
    total_files = 0
    for root, directories, filenames in os.walk(data_dir):
        if (len(filenames)>1):
            spk_order_same.append(root.split('/')[-1])
        for filename in filenames:
            if filename.endswith('.mbk'):  # filename: 4253-f-sre2006-jhio-B-f17943.mbk
                total_files += 1
                spk_id = filename.split('-')[0]   # spk_id: 4253
                # if frame_n < 2048:
                #     continue

                # root = /home/hyd/yandong_sre_data/vad_train_mbk_norm_300_full/8825
                spk_dir = root.split('/')[-1] + '/'  # spk_dir = 8825/

                # don't include 'data_dir', this can save 50% spaces for 'same_spk_pairs'
                if spk_id not in spk2utt_dict:
                    spk2utt_dict[spk_id] = [spk_dir + filename]   # {'id1: [utt1, utt2], 'id2': [utt1]}
                else:
                    spk2utt_dict[spk_id].append(spk_dir + filename)
                spk_count[spk_id] = 0
    print('totalspk: {}, spk_count: {}, spk_order_all: {}, spk_order_samespk: {}, total_files: {}'\
                        .format(len(spk2utt_dict), len(spk_count), len(spk2utt_dict.keys())\
                                 , len(spk_order_same), total_files))

    return spk2utt_dict, spk_count, list(spk2utt_dict.keys()), spk_order_same, total_files

class TrainSet():
    def __init__(self, data_dir, bs):
        self.N = bs // 3 # 
        self.anchors = []
        self.positives = []
        self.negatives = []
        self.new_b_files = []
        self.spk_done = 0
        self.spk_dict, self.spk_count, self.spk_order_all, self.spk_order, \
                self.total_files = parse_data(data_dir)
        self.n_index=0
        self.total_same_spk = len(self.spk_order)


    def __len__(self):
        return len(self.file_list)

    def reset(self):
        self.spk_done = 0
        self.n_index = 0
        for k in self.spk_count:
            self.spk_count[k] = 0
        shuffle(self.spk_order)
        for spk in self.spk_order:
            shuffle(self.spk_dict[spk])

    def get_batch(self):
        self.new_b_files = []
        self.anchors = []
        self.positives = []
        self.negatives = []
        i = 0
        skipped = 0
        count_done = 0
        spk_seen = []
        total_spk = len(self.spk_order_all)
        ### making N same speaker pairs
        while i < self.N:
            if (self.n_index==self.total_same_spk):
                self.n_index = 0
            c_spk = self.spk_order[self.n_index]
            diff = len(self.spk_dict[c_spk]) - self.spk_count[c_spk]
            if (self.spk_done == self.total_same_spk):
                break
            count_done = 0
            if (diff == 0):
                self.n_index+=1
                continue
            c_spk_f1 = self.spk_dict[c_spk][self.spk_count[c_spk]]
            #self.new_b_files.append(c_spk_f1)
            self.anchors.append(c_spk_f1)
            if (diff >= 2):
                c_spk_f2 = self.spk_dict[c_spk][self.spk_count[c_spk]+1]
                #self.new_b_files.append(c_spk_f2)
                self.positives.append(c_spk_f2)
                self.spk_count[c_spk] += 2
                if (diff == 2):
                    self.spk_done += 1

            elif (diff == 1):
                rand_int = randint(0, len(self.spk_dict[c_spk])-2)
                c_spk_f2 = self.spk_dict[c_spk][rand_int]
                #self.new_b_files.append(c_spk_f2)
                self.positives.append(c_spk_f2)
                self.spk_count[c_spk] += 1
                self.spk_done += 1
            
            neg_rand_int = randint(0, total_spk-1)
            while (neg_rand_int == self.n_index):
                neg_rand_int = randint(0, total_spk-1)
            c_spk_n = self.spk_order_all[neg_rand_int]
            rand_int = randint(0, len(self.spk_dict[c_spk_n])-1)
            c_spk_f3 = self.spk_dict[c_spk_n][rand_int]
            self.negatives.append(c_spk_f3)
            spk_seen.append(c_spk)
            self.n_index += 1
            i = i + 1

        self.anchors.extend(self.positives)
        self.anchors.extend(self.negatives)
        return self.anchors, i


if __name__ == "__main__":
    batchsize = 30 
    feat_dim = 5
    data_dir = 'path/to/data/dir/'
    trainset = TrainSet(data_dir, batchsize)
    batch, N = trainset.get_batch()
    criterion = QuartetLoss()
    feature = torch.FloatTensor(np.random.rand(batchsize, feat_dim))
    loss = criterion.forward(feature, batchsize, feat_dim, N)
    print("Loss: ", loss)

    ## The general structure of the code should be: 
    '''
    for epoch in range(NO_EPOCH):
        scheduler.step()
        data_list, N = trainset.get_batch()
        while (len(data_list) == batchsize and trainset.spk_done < trainset.total_same_spk):
            for i, data in enumerate(dataloader):
                optimizer.zero_grad()
                # torch.FloatTensor
                data = Variable(data.cuda(), requires_grad=True)
                feature = net(data)
                loss = criterion.forward(feature, batchsize)
                loss.backward()
                optimizer.step()
                data_list, N = trainset.get_batch()
        # reset variables for trainset
        trainset.reset()
    '''