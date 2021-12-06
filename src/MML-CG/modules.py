import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np


class Model(nn.Module):
    def __init__(self, n_embs, n_hidden, n_head, n_block, max_len, dropout, vocab_size, left_range, right_range, img_dim=2048):
        super(Model, self).__init__()
        
        # set parameter
        self.n_embs = n_embs
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.n_block = n_block
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout
        self.img_dim = img_dim
        self.left_range = left_range
        self.right_range = right_range
        
        # video encoder
        self.img_global_encoder = VideoLSTM(img_dim, n_hidden, dropout)
        self.img_local_encoder = VideoAttention(img_dim, n_hidden, n_head, n_embs, dropout)
        
        # text encoder
        self.word_embedding = nn.Embedding(vocab_size, n_embs)
        self.sentence_encoder = TextLSTM(n_embs, n_hidden, dropout)
        
        # temporal predictor
        self.c_attn = MultiModalAttention(n_embs, dropout)
        self.temporal_predictor = nn.Linear(2*n_hidden, 2)
        
        # comments decoder
        self.decoder = CommentDecoder(n_embs, n_hidden, n_head, n_block, dropout)
        self.output_layer = nn.Linear(self.n_embs, self.vocab_size)
        
        # loss
        self.criterion_generation = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)
        self.criterion_classification = nn.CrossEntropyLoss()
        
    def encode_image(self, V):
        img_global = self.img_global_encoder(V)
        img_local = self.img_local_encoder(V)
        return img_global, img_local
    
    def encode_text(self, T):
        T_embs = self.word_embedding(T)
        T_encode = self.sentence_encoder(T_embs)
        return T_encode
    
    def decode(self, Y, text_v, mask):
        embs = self.word_embedding(Y)
        out = self.decoder(embs, text_v, mask)
        out = self.output_layer(out)
        return out
        
    def forward(self, V, T, Y):
        # encode image
        img_global, img_local = self.encode_image(V)

        # encode surrounding comments
        text = self.encode_text(T)

        # classification
        loss_c, text_v = self.classifier(img_global, img_local, text)
        mask = Variable(subsequent_mask(Y.size(0), Y.size(1)-1), requires_grad=False).cuda()
        
        # decode
        outs = self.decode(Y[:,:-1], text_v, mask)

        # compute loss
        Y = Y.t()
        outs = outs.transpose(0, 1)
        loss_g = self.criterion_generation(outs.contiguous().view(-1, self.vocab_size), Y[1:].contiguous().view(-1))
        loss = 0.7*torch.mean(loss_g) + 0.3*loss_c
        # print('Total loss', loss)
        return loss
        
    def classifier(self, img_global, img_local, text):
        '''
            Whether comments appears time stamp t
        '''
        batch_size = img_global.size(0)
        IG = (img_global.unsqueeze(1)).repeat(1, text.size(1), 1)
        c = self.c_attn(IG.view(-1, 1, IG.size(-1)), text.view(-1, text.size(-2), text.size(-1)))
        text_topic = c.view(batch_size, -1, c.size(-1))
        #print('Text topic ', text_topic.size())
        vt_feature = torch.cat((text_topic, img_global.unsqueeze(1), img_local.unsqueeze(1)), dim=1)
        #print('Vt_feature ', vt_feature.size())
        
        # classification
        vt = torch.cat((text_topic, (img_local.unsqueeze(1)).repeat(1, text_topic.size(1), 1)), dim=-1)
        vt_predict = self.temporal_predictor(vt)
        before = Variable(torch.LongTensor(batch_size, self.left_range).fill_(1)).cuda()
        after = Variable(torch.LongTensor(batch_size, self.right_range).fill_(0)).cuda()
        labels = torch.cat((before, after), dim=1)
        loss = self.criterion_classification(vt_predict.view(-1, 2), labels.view(-1))

        return loss, vt_feature
    
    def ranking(self, V, T, C):
        nums = len(C)
        img_global, img_local = self.encode_image(V.unsqueeze(0))
        text = self.encode_text(T.unsqueeze(0))
        loss_c, text_v = self.classifier(img_global, img_local, text)
        text_v = text_v.repeat(nums, 1, 1)
        mask = Variable(subsequent_mask(C.size(0), C.size(1) - 1), requires_grad=False).cuda()
        outs = self.decode(C[:, :-1], text_v,  mask)

        C = C.t()
        outs = outs.transpose(0, 1)

        loss = self.criterion_generation(outs.contiguous().view(-1, self.vocab_size), C[1:].contiguous().view(-1))

        loss = loss.view(-1, nums).sum(0)
        return torch.sort(loss, dim=0, descending=False)[1]
        
        
    
    def generate(self, V, T, BOS_token, EOS_token, beam_size):
        img_global, img_local = self.encode_image(V.unsqueeze(0))
        text = self.encode_text(T.unsqueeze(0))
    
        loss_c, text_v = self.classifier(img_global, img_local, text)
        comments = self.beam_search(text_v, BOS_token, EOS_token, beam_size)   
        return comments
    
    def beam_search(self, text_v, BOS_token, EOS_token, beam_size):
        LENGTH_NORM = True
        batch_size = text_v.size(0)
        startTokenArray = Variable(torch.LongTensor(batch_size, 1).fill_(BOS_token)).cuda()
        #print('Start matrix ', startTokenArray.size())

        backVector = torch.LongTensor(beam_size).cuda()
        torch.arange(0, beam_size, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batch_size, 1)
        backVector = Variable(backVector)
        #print('Back matrix ', backVector.size())

        tokenArange = torch.LongTensor(self.vocab_size).cuda()
        torch.arange(0, self.vocab_size, out=tokenArange)
        tokenArange = Variable(tokenArange)
        #print('Token matrix ', tokenArange.size())

        beamTokensTable = torch.LongTensor(batch_size, beam_size, self.max_len).fill_(EOS_token)
        beamTokensTable = Variable(beamTokensTable.cuda())
        #print('beam Table ', beamTokensTable.size())

        backIndices = torch.LongTensor(batch_size, beam_size, self.max_len).fill_(-1)
        backIndices = Variable(backIndices.cuda())
        #print('Back Indices ', backIndices.size())

        aliveVector = beamTokensTable[:, :, 0].eq(EOS_token).unsqueeze(2)
        #print('AliveVector ', aliveVector.size())

        for i in range(self.max_len-1):
            if i  == 0:
                Cap = startTokenArray
                mask = Variable(subsequent_mask(Cap.size(0), Cap.size(1))).cuda()
                #print('Mask ', mask.size())
                out = self.decode(Cap, text_v, mask)
                #print('Out ', out.size())
                probs = out[:, -1]
                topProbs, topIdx = probs.topk(beam_size, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                ProbSums = topProbs
            else:
                Cap = beamTokensTable[:, :, :i].squeeze(0)
                mask = Variable(subsequent_mask(Cap.size(0), Cap.size(1))).cuda()
                out = self.decode(Cap, text_v.repeat(beam_size, 1, 1), mask)
                probCurrent = out[:, -1,:].view(batch_size, beam_size, self.vocab_size)

                if LENGTH_NORM:
                    probs = probCurrent * (aliveVector.float() / (i+1))
                    coeff_ = aliveVector.eq(0).float() + (aliveVector.float() * i / (i+1))
                    probs += ProbSums.unsqueeze(2) * coeff_
                else:
                    probs = probCurrent * (aliveVector.float())
                    probs += ProbSums.unsqueeze(2)

                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocab_size)
                mask_[:, :, 0] = 0
                minus_infinity_ = torch.min(probs).item()

                probs.data.masked_fill_(mask_.data, minus_infinity_)
                probs = probs.view(batch_size, -1)

                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).repeat(batch_size, beam_size, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), 2)
                tokensArray = tokensArray.view(batch_size, -1)
                backIndexArray = backVector.unsqueeze(2).repeat(1, 1, self.vocab_size).view(batch_size, -1)

                topProbs, topIdx = probs.topk(beam_size, dim=1)
                ProbSums = topProbs
                beamTokensTable[:, :, i] = tokensArray.gather(1, topIdx)
                backIndices[:, :, i] = backIndexArray.gather(1, topIdx)

            aliveVector = beamTokensTable[:, :, i:i + 1].ne(2)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = i
            if aliveBeams == 0:
                break

        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        RECOVER_TOP_BEAM_ONLY = True
        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while tokenIdx >= 0:
            tokens.append(beamTokensTable[:, :, tokenIdx].gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beam_size, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLen = tokens.ne(2).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            tokens = tokens[:, 0]
            seqLen = seqLen[:, 0]
            
        return Variable(tokens)        
    
class VideoEncoder(nn.Module):
    def __init__(self, dim, dim_ff, n_head, n_block, dropout):
        super(VideoEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class VideoLSTM(nn.Module):
    def __init__(self, n_emb, n_hidden, dropout, num_layers=2):
        super(VideoLSTM, self).__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size = n_emb,
                            hidden_size = n_hidden,
                            num_layers = num_layers,
                            batch_first = True)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.n_hidden).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.n_hidden).cuda())
        hidden = (h0, c0)
        lstm_out, (h, c) = self.lstm(x, hidden)
        x = lstm_out[:,-1,:]
        return x

class VideoAttention(nn.Module):
    def __init__(self, dim, dim_ff, n_head, out_dim, dropout, n_block=1):
        super(VideoAttention, self).__init__()
        self.layers = nn.ModuleList([AttnBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.linear = nn.Linear(dim, out_dim)
        self.norm = LayerNorm(out_dim)

    def forward(self, x):
        t = x.size(1) // 2
        img_t = x[:,t,:]
        for layer in self.layers:
            x = layer(img_t.unsqueeze(1), x)
        x = self.linear(x.squeeze(1))
        return self.norm(x)

class TopicAttention(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout, n_block=1):
        super(TopicAttention, self).__init__()
        self.layers = nn.ModuleList([AttnBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x, m):
        for layer in self.layers:
            x = layer(x, m)
        return x

class TextAttention(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout, n_block=4):
        super(TextAttention, self).__init__()
        self.layers = nn.ModuleList([AttnBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, x)
        return x

class MultiModalAttention(nn.Module):
    def __init__(self, dim, dropout):
        super(MultiModalAttention, self).__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, v):
        weights = torch.bmm(q, v.transpose(1, 2))
        attn_weights = F.softmax(weights.squeeze(1), dim=1)
        output = torch.bmm(attn_weights.unsqueeze(1), v)
        return output.squeeze(1)

class AttnBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(AttnBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x, m):
        x = self.sublayer[0](x, lambda x: self.attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward)


class TextLSTM(nn.Module):
    def __init__(self, n_emb, n_hidden, dropout, num_layers=2):
        super(TextLSTM, self).__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size = n_emb,
                            hidden_size = n_hidden,
                            num_layers = num_layers,
                            batch_first = True)
    def forward(self, x):
        batch_size = x.size(0)
        time = x.size(1)

        x = x.view(-1, x.size(-2), x.size(-1))

        # initilize hidden state
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.n_hidden).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.n_hidden).cuda())
        hidden = (h0, c0)
        lstm_out, (h, c) = self.lstm(x, hidden)
        x = lstm_out[:,-1,:]
        x = x.view(batch_size, time, -1, x.size(-1))
        return x

class CommentDecoder(nn.Module):
    def __init__(self, dim, dim_ff, n_head, n_block, dropout):
        super(CommentDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x, tv, mask):
        for layer in self.layers:
            x = layer(x, tv, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(dim, n_head, dropout)
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(3)])

    def forward(self, x, tv, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, lambda x: self.attn(x, tv, tv))
        return self.sublayer[2](x, self.feed_forward)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert dim % n_head == 0
        self.d_k = dim // n_head
        self.n_head = n_head
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        weights = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(weights, dim=-1)

        if dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)
        return self.linears[-1](x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, dim)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dim_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def subsequent_mask(batch, size):
    # mask out subsequent positions
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   