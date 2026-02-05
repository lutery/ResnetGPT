import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        '''
        这个可以去除
        
        :param self: Description
        :param vocab_size: 词表大小 todo
        :param d_model: 采样的维度 todo
        '''
        super().__init__()
        self.d_model = d_model
        self.embed = Embedder2(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        x = self.dropout(x)
        return x


class Embedder2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        '''
        这里是一个嵌入层的实现，类似于PyTorch自带的nn.Embedding
        但是增加了一些自定义的初始化逻辑，在训练的时候可以使用预训练的词向量
        也可以随机初始化词向量
        
        :param self: Description
        :param num_embeddings: 词表大小 todo
        :param embedding_dim: 采样的维度 todo
        :param padding_idx: todo
        :param max_norm: todo
        :param norm_type: todo
        :param scale_grad_by_freq: todo
        :param sparse: todo
        :param _weight: todo
        '''
        super(Embedder2, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            # 防御性编程，检查padding_idx是否在范围内
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            # 随机初始化权重
            np.random.seed(1)
            np数 = np.random.uniform(0, 1, (num_embeddings, embedding_dim))
            self.weight = nn.Parameter(torch.Tensor(np数))
            # self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            #self.reset_parameters()
        else:
            # 外部传入权重
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)
        self.sparse = sparse 
        a = 0

    def reset_parameters(self):
        # 重置权重
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)