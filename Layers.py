import torch
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        '''
        Docstring for __init__
        
        :param self: Description
        :param d_model: 特征提取的维度 todo
        :param heads: 注意力头数 todo
        :param dropout: 随机失活比率
        '''
        super().__init__()
        # 一上来就是三个归一化层
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        # 构建三个dropout层
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        # 多头注意力机制和前馈神经网络
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x,  trg_mask):
        '''
        Docstring for forward
        
        :param self: Description
        :param x:输入特征
        :param trg_mask: 目标序列的掩码 todo
        :return: 输出特征 shape(batch_size, seq_len, d_model)
        '''

        x2 = self.norm_1(x) # 进行归一化
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask)) # 自注意力机制后在进行残差连接和dropout
        x2 = self.norm_3(x) # 进行归一化
        x2 = self.ff(x2) # 前馈神经网络
        x = x + self.dropout_3(x2) # 残差连接和dropout
        return x