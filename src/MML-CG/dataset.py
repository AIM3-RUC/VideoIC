'''
    Dataset
'''

import os
import sys
import time
import json

import torch
from torch.utils.data import Dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocabs, rev_vocabs, images, left_time_range, right_time_range, max_len, max_cnum, is_train, set_name='train'):
        
        # set parameter
        self.data_path = data_path
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.images = images
        self.left_time_range = left_time_range
        self.right_time_range = right_time_range
        self.max_len = max_len
        self.max_cnum = max_cnum
        self.is_train = is_train
        self.set_name = set_name
        
        self.BOS = vocabs['<BOS>']
        self.EOS = vocabs['<EOS>']
        self.UNK = vocabs['<UNK>']
        self.PAD = vocabs['<PAD>']

        if self.PAD != 0:
            print('Error! Please set <PAD> id 0 in the dict!')
            sys.exit()
        
        # load data
        if self.set_name == 'test':
            self.load_testdata()
        else:
            self.load_data()
        
    def load_data(self):
        count = 1000 # no use, just load small part of data when debug
        start_time = time.time()
        self.datas = []
        self.processed_img = {}
        with open(self.data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                count -= 1
                jterm = json.loads(line)
                video_id = jterm['video']
                video_time = jterm['time']
                sample = {'video_id': video_id,
                          'video_time': video_time,
                          'comment': jterm['comment'],
                          'context': jterm['context']}
                start_time = video_time - self.left_time_range
                end_time = video_time + self.right_time_range
                
                # format video feature
                video_feature = self.load_imgs(video_id, start_time, end_time)
                if video_feature is None:
                    continue
                sample['video_feature'] = video_feature
                
                # format ground truth comments
                sample['context_feature'] = self.load_comments(jterm['context'], start_time, end_time, video_time)
                sample['comment_feature'] = self.padding(jterm['comment'])

                self.datas.append(sample)

        print('Finish loading data ', len(self.datas), ' samples')
        print('Time ', time.time() - start_time)
        
    def load_testdata(self):
        count = 1000
        start_time = time.time()
        self.datas = []
        self.processed_img = {}
        with open(self.data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                count -= 1
                jterm = json.loads(line)
                video_id = jterm['video']
                video_time = jterm['time']
                sample = {'video_id': video_id,
                          'video_time': video_time,
                          'comment': jterm['comment'],
                          'context': jterm['context']}
                start_time = video_time - self.left_time_range
                end_time = video_time + self.right_time_range
                
                # format video feature
                video_feature = self.load_imgs(video_id, start_time, end_time)
                if video_feature is None:
                    continue
                sample['video_feature'] = video_feature
                
                # format ground truth comments
                sample['context_feature'] = self.load_comments(jterm['context'], start_time, end_time, video_time)

                #sample['comment_feature'] = self.padding(jterm['comment'])
                if 'candidate' in jterm:
                    sample['candidate'] = jterm['candidate']
                    sample['candidate_feature'] = [self.padding(c) for c in jterm['candidate']]
                self.datas.append(sample)

        print('Finish loading data ', len(self.datas), ' samples')
        print('Time ', time.time() - start_time)        
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        V = data['video_feature']
        T = data['context_feature']
        Y = data['comment_feature']
        return V, T, Y
    
    def get_data(self, index):
        return self.datas[index]
                
    def load_imgs(self, video_id, start_time, end_time):
        img_list = []
        for time in range(start_time, end_time+1):
            if time not in self.images[video_id]:
                print('Image Wrong. Video: ', video_id, ' time: ', time)
                return None
            img_list.append(torch.from_numpy(self.images[video_id][time]))
        return torch.stack(img_list)
    
    def load_comments(self, comments, start_time, end_time, video_time):
        comment_list = []
        for time in range(start_time, end_time+1):
            if time == video_time:
                # skip ground truth
                continue
            else:
                time = str(time)
                c_list = [] # multiple comments at each time
                if time in comments:
                    for c in comments[time]:
                        c_list.append(self.padding(c))
                        if len(c_list) == self.max_cnum:
                            break
                    # padding
                    while len(c_list) <  self.max_cnum:
                        c_list.append(self.padding(''))
                else:
                    while len(c_list) <  self.max_comment:
                        c_list.append(self.padding(''))
                comment_list.append(torch.stack(c_list)) # append comments at this second
                
        return torch.stack(comment_list)
    
    def padding(self, data):
        data = data.split(' ')
        if len(data) > self.max_len-2:
            data = data[:self.max_len-2]
        Y = list(map(lambda t: self.vocabs.get(t, self.UNK), data))
        # start and end token
        Y = [self.BOS] + Y + [self.EOS]
        length = len(Y)
        # padding
        Y = torch.cat([torch.LongTensor(Y), torch.zeros(self.max_len - length).long()])
        return Y
        
        
                
            
        
                
        
        
        