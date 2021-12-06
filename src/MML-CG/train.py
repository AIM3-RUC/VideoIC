'''
    Re-organize the MMIG model
    2021-09-20
'''

import os
import sys
import time
import json
import logging
import argparse

import torch
import torch.optim as Optim
from torch.autograd import Variable

import utils
import modules
import dataset
import metrics


# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

parser = argparse.ArgumentParser(description='train.py')
# set model parameters
parser.add_argument('-n_emb', type=int, default=512, help='Embedding size')
parser.add_argument('-n_hidden', type=int, default=512, help='Hidden size')
parser.add_argument('-n_head', type=int, default=8, help='Number of head')
parser.add_argument('-n_block', type=int, default=1, help="Number of block") # value 1 for livebot dataset, value 6 for videoic dataset

parser.add_argument('-max_len', type=int, default=20, help="Limited length for text")
parser.add_argument('-time_range', type=int, default=5, help='Time range')
parser.add_argument('-max_cnum', type=int, default=15, help="Max comments each second")
parser.add_argument('-beam_size', type=int, default=1, help="Bean size") # 1 means greedy search, which is the same with our paper implement

# training setting
parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
parser.add_argument('-epoch', type=int, default=100, help='Number of epoch')
parser.add_argument('-dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('-lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('-weight_decay', type=float, default=0.001, help="Learning rate")
parser.add_argument('-early_stop', type=float, default=20, help="Early Stop")

# data path
parser.add_argument('-dataset', type=str, default='livebot', help='name of dataset')
parser.add_argument('-data_path', type=str, default=None, help='dict and image path')
parser.add_argument('-out_path', type=str, default=None, help='out path')
parser.add_argument('-outfile', type=str, default=None, help='outfile for generation')
parser.add_argument('-restore', type=str, default=None, help="Restoring model path")
parser.add_argument('-mode', type=str, default=None)
args = parser.parse_args()

# set random seed
torch.manual_seed(116)
torch.cuda.manual_seed(116)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# log file
if args.mode == 'train':
    if args.dataset == 'livebot':
        args.data_path = '/data2/wwy/barrage/data/livebot'
        args.out_path = 'livebot'
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        logger.addHandler(logging.FileHandler(os.path.join(args.out_path, 'livebot_log'), "w"))
    elif args.dataset == 'aim3':
        args.data_path = '/data2/wwy/barrage/data/aim3'
        args.out_path = 'aim3'
        os.mkdir(args.out_path)
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        logger.addHandler(logging.FileHandler(os.path.join(args.out_path, 'aim3_log'), "w"))
else:
    if args.dataset == 'livebot':
        args.data_path = '/data2/wwy/barrage/data/livebot'
        args.out_path = 'livebot'
    elif args.dataset == 'aim3':
        args.data_path = '/data2/wwy/barrage/data/aim3'
        args.out_path = 'aim3'     
    
# load img
images = utils.load_images(args.data_path)

# load vocabs
vocabs, rev_vocabs = utils.load_vocabs(args.data_path)
#logger.info('Load vocabs file ' + str(len(vocabs)))

def get_dataset(data_path, images, is_train, set_name):
    return dataset.Dataset(data_path = data_path,
                           vocabs = vocabs,
                           rev_vocabs=rev_vocabs,
                           images = images,
                           left_time_range = args.time_range,
                           right_time_range = args.time_range,
                           max_len = args.max_len,
                           max_cnum = args.max_cnum,
                           is_train = is_train,
                           set_name = set_name)
    
def get_dataloader(dataset, batch_size, is_train):
    return torch.utils.data.DataLoader(dataset = dataset,
                                       batch_size = batch_size,
                                       shuffle = is_train)
    
def save_model(path, model):
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, path)


def train():
    # load dataset
    train_set = get_dataset(data_path = os.path.join(args.data_path, 'train.json'),
                            images = images,
                            is_train = True)
    valid_set = get_dataset(data_path = os.path.join(args.data_path, 'dev.json'),
                            images = images,
                            is_train = False)
    train_batch = get_dataloader(dataset = train_set,
                                 batch_size = args.batch_size,
                                 is_train = True)
    
    model = modules.Model(n_embs = args.n_emb,
                          n_hidden = args.n_hidden,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          max_len = args.max_len,
                          dropout = args.dropout,
                          vocab_size = len(vocabs),
                          left_range = args.time_range,
                          right_range = args.time_range)
    
    if args.restore is not None:
        model_dict = torch.load(args.restore)
        model.load_state_dict(model_dict)
    
    model.cuda()
    optim = Optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, weight_decay=args.weight_decay)
    best_score = -100000
    early_stop_count = 0
        
    for i in range(args.epoch):
        model.train()
        report_loss, start_time, n_samples = 0, time.time(), 0
        
        for batch in train_batch:
            model.zero_grad()
            V, S, Y = batch
            
            # V: video feature
            V = Variable(V).cuda()
            # S: Surrounding comments
            S = Variable(S).cuda()
            # Y: Ground truth
            Y = Variable(Y).cuda()
            
            loss = torch.sum(model(V, S, Y))
            loss.backward()
            optim.step()
            
            report_loss += loss.item()
            n_samples += V.size(0)
        
        # report loss
        print('\nEpoch: %d, report_loss: %.3f, time: %.2f'
              % (i+1, report_loss / n_samples, time.time() - start_time))
        logger.info('\nEpoch '+str(i) + ', report_loss: '+str(report_loss/n_samples) + ' , time: ' + str(time.time() - start_time))
        
        # eval
        score = eval(model, valid_set)
        if score > best_score:
            best_score = score
            print('Best score ', best_score)
            save_model(os.path.join(args.out_path, 'best_checkpoint.pt'), model)
            logger.info('Evaluation score ' + str(score) + ', Best score ' + str(best_score))
            early_stop_count = 0
        else:
            early_stop_count += 1
            save_model(os.path.join(args.out_path, 'checkpoint.pt'), model)
            print('Evaluation score ', score, '. Best score ', best_score, '. Early stop count ', early_stop_count)
            if early_stop_count == args.early_stop:
                sys.exit()
    return 0

def eval(model, valid_set):
    print('Start Evaluation ... ')
    start_time = time.time()
    model.eval()
    valid_batch = get_dataloader(valid_set, args.batch_size, is_train=False)
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in valid_batch:
            V, S, Y = batch
            V = Variable(V).cuda()
            S = Variable(S).cuda()
            Y = Variable(Y).cuda()
            
            total_loss += torch.sum(model(V, S, Y)).item()
            total_samples += V.size(0)

    loss = total_loss / total_samples
    print('Loss: ', loss)
    print("evaluting time:", time.time() - start_time)
    return -loss

def test_generation():
    # build model
    test_set = get_dataset(data_path = os.path.join(args.data_path, 'test.json'),
                           images = images,
                           is_train = False,
                           set_name = 'test')
    
    model = modules.Model(n_embs = args.n_emb,
                          n_hidden = args.n_hidden,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          max_len = args.max_len,
                          dropout = args.dropout,
                          vocab_size = len(vocabs),
                          left_range = args.time_range,
                          right_range = args.time_range)
    
    if args.restore is not None:
        model_dict = torch.load(args.restore)
        model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
    else:
        print('Error! Fail to load model for test mode')
        sys.exit()
        
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        with open(args.outfile, 'w') as fout:
            for i in range(len(test_set)):
                data = test_set.get_data(i)
                V = data['video_feature']
                S = data['context_feature']
                V = Variable(V).cuda()
                S = Variable(S).cuda()
                comment_ids = model.generate(V, S, BOS_token=vocabs['<BOS>'], EOS_token=vocabs['<EOS>'], beam_size=args.beam_size).data.tolist()
                comment = transform(comment_ids[0])
                for key in data:
                    print(key)
                sample = {'video_time': data['video_time'],
                          'context': data['context'],
                          'comment': data['comment'],
                          'candidate': data['candidate'],
                          'generation': comment}
                term = json.dumps(sample, ensure_ascii=False)
                fout.write(str(term)+'\n')

def transform(ids):
    sentences = []
    for wid in ids:
        if wid == vocabs['<BOS>']:
            continue
        if wid == vocabs['<EOS>']:
            break
        sentences.append(rev_vocabs[wid])
    return sentences

def test_ranking():
    # build model
    test_set = get_dataset(data_path = os.path.join(args.data_path, 'test.json'),
                           images = images,
                           is_train = False,
                           set_name = 'test')
    
    model = modules.Model(n_embs = args.n_emb,
                          n_hidden = args.n_hidden,
                          n_head = args.n_head,
                          n_block = args.n_block,
                          max_len = args.max_len,
                          dropout = args.dropout,
                          vocab_size = len(vocabs),
                          left_range = args.time_range,
                          right_range = args.time_range)
    
    if args.restore is not None:
        model_dict = torch.load(args.restore)
        model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
    else:
        print('Error! Fail to load model for test mode')
        sys.exit()
        
    model.cuda()
    model.eval()
    
    predictions, references = [], []
    
    with torch.no_grad():
        for i in range(len(test_set)):
            data = test_set.get_data(i)
            V = Variable(data['video_feature']).cuda()
            S = Variable(data['context_feature']).cuda()
            C = Variable(torch.stack(data['candidate_feature'])).cuda()
            comment_ids = model.ranking(V, S, C).data
            
            candidate = []
            comments = list(data['candidate'].keys())
            for id in comment_ids:
                candidate.append(comments[id])
            predictions.append(candidate)
            references.append(data['candidate'])

    recall_1 = metrics.recall(predictions, references, 1)
    recall_5 = metrics.recall(predictions, references, 5)
    recall_10 = metrics.recall(predictions, references, 10)
    mr = metrics.mean_rank(predictions, references)
    mrr = metrics.mean_reciprocal_rank(predictions, references)
    print('Report ranking result')
    print('Recall 1: ', recall_1)
    print('Recall 5: ', recall_5)
    print('Recall 10: ', recall_10)
    print('MR: ', mr)
    print('MRR: ', mrr)


if __name__ == '__main__':
    if args.mode == 'train':
        print('-----------Train Mode-----------')
        train()
    elif args.mode == 'generate':
        print('-----------Generation Mode-----------')
        test_generation()
    elif args.mode == 'ranking':
        print('-----------Ranking Mode-----------')
        test_ranking()
    else:
        print('Wrong Mode')