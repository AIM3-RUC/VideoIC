import os
import json
import torch

def load_vocabs(data_path):
    vocab_path = os.path.join(data_path, 'dict.json')
    vocabs = json.load(open(vocab_path, 'r', encoding='utf-8'))['word2id']
    rev_vocabs = {vocabs[k]: k for k in vocabs}
    #rev_vocabs = json.load(open(vocab_path, 'r', encoding='utf-8'))['id2word']
    print('Load vocabs ', len(vocabs))
    return vocabs, rev_vocabs

def load_images(data_path):
    img_path = os.path.join(data_path, 'image.pkl')
    images = torch.load(open(img_path, 'rb'))
    print('Loading Images ')
    return images
    