import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        '''
        在标准归一化的基础上增加了可学习的参数alpha和bias，使得归一化的结果更加适合模型训练
        
        :param self: Description
        :param d_model: 特征提取的维度 todo
        :param eps: 防止除零错误
        '''

        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        # 创建两个可学习的参数来校准归一化
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        # 归一化的流程
        # x.mean(dim=-1, keepdim=True)：计算最后一个维度的均值，保持维度不变
        # x.std(dim=-1, keepdim=True)：计算最后一个维度的标准差，保持维度不变
        # x - x.mean(dim=-1, keepdim=True)：对x进行均值归一化
        # x.std(dim=-1, keepdim=True) + self.eps：计算标准差并加上一个小的常数以防止除零错误
        # (x - x.mean(dim=-1, keepdim=True)) \ (x.std(dim=-1, keepdim=True) + self.eps) ：进行标准化
        # self.alpha * ... + self.bias：应用可学习的缩放和平移参数实现更加适合的归一化
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    '''
    
    :param q: 多头注意力机制中的查询矩阵
    :param k: 多头注意力机制中的键矩阵
    :param v: 多头注意力机制中的值矩阵
    :param d_k: 每个注意力头的维度
    :param mask: 掩码，用于屏蔽某些位置的注意力，在本代码中是否用到了mask还不清楚 todo
    :param dropout: 随机失活层
    '''
    
    # torch.matmul(q, k.transpose(-2, -1)) ： 计算查询矩阵和键矩阵的点积，得到注意力分数
    # /  math.sqrt(d_k)：对点积结果进行缩放，防止数值过大
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None: # 如果有传入掩码
        mask = mask.unsqueeze(1)
        # 对注意力分数进行掩码处理，将掩码位置的分数设为一个很小的值，防止其在softmax中被选中
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 对注意力分数进行softmax归一化
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None: # 如果传入了dropout层，则应用dropout
        scores = dropout(scores)
        
    # 将分数与值矩阵相乘，得到最终的注意力输出
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        '''
        Docstring for __init__
        
        :param self: Description
        :param heads: 注意力头数 todo
        :param d_model: 特征提取的维度 todo
        :param dropout: 随机失活比率
        '''
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads # 每个头的维度
        self.h = heads # 头的数量
        
        # 定义线性变换层，提取q、k、v
        self.q_linear = 全连接层(d_model, d_model)
        self.v_linear = 全连接层(d_model, d_model)
        self.k_linear = 全连接层(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = 全连接层(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        '''
        传入的qkv从哪里来的？是不是直接复用上一层的输出？todo
        
        :param self: Description
        :param q: 多头注意力机制中的查询矩阵
        :param k: 多头注意力机制中的键矩阵
        :param v: 多头注意力机制中的值矩阵
        :param mask: 掩码，用于屏蔽某些位置的注意力
        '''
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        # 对输入的q、k、v进行线性变换，并分割成多个头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        # 计算注意力分数，明确该助理哪些部分内容是重点
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        # 将多个注意力头的输出拼接起来，并通过最终的线性层
        # concat shape is (bs, sl, d_model)，其中sl是序列长度 todo 这里的序列长度是什么？
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        # 进一步特征提取
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = 全连接层(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = 全连接层(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
class 全连接层(nn.Module):
    def __init__(self,输入_接口, 输出_接口):
        super().__init__()
        np.random.seed(1)
        self.weight = nn.Parameter(torch.FloatTensor(np.random.uniform(-1/np.sqrt(输入_接口), 1/np.sqrt(输入_接口), (输入_接口, 输出_接口))))
        self.bias = nn.Parameter(torch.FloatTensor(np.random.uniform(-1/np.sqrt(输入_接口), 1/np.sqrt(输入_接口), 输出_接口)))


    def forward(self, x):
        输出=torch.matmul(x,self.weight)
        输出=输出+self.bias
        return 输出