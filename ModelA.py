import torch
import torch.nn as nn 
from Layers import  DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm, 全连接层
import copy
import os.path
import torchvision


def get_clones(module, N):
    # 返回N个相同的模块的列表，用于堆叠多层
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, 最大长度=1024):
        '''
        
        :param self: Description
        :param vocab_size: 词表大小 todo
        :param d_model: 采样的维度 todo
        :param N: 层数
        :param heads: 注意力头数 todo
        :param dropout: dropout比率
        :param 最大长度: 最大序列长度 todo 这是干嘛的
        '''
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model) # 对游戏操作序列进行嵌入编码 
        self.embedP = Embedder(最大长度, d_model) # 这个嵌入向量是对位置进行编码的
       # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model) # 归一化层
    def forward(self,图向量,操作 ,trg_mask):
        '''
        Docstring for forward
        
        :param self: Description
        :param 图向量: 输入的图像特征向量 todo shape是啥？
        :param 操作: 输入的操作序列 todo 这个操作序列是啥？
        :param trg_mask: 目标序列的掩码 todo
        :return: 输出特征 shape(batch_size, seq_len, d_model)
        '''

        # 简单的序列位置编码
        position = torch.arange(0, 图向量.size(1), dtype=torch.long,
                                    device=图向量.device)


        # 对位置进行嵌入编码，并与图像特征和操作特征相加
        x = 图向量+self.embedP(position)+self.embed(操作)*0


        # 通过N层解码器层
        for i in range(self.N):
            x = self.layers[i](x,  trg_mask)
        # 最终进行一次归一化
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self,  trg_vocab, d_model, N, heads, dropout,图向量尺寸=6*6*2048):
        '''
        Transformer模型，只有解码器部分
        
        :param self: Description
        :param trg_vocab: 词表大小 todo 是啥？
        :param d_model: 采样的维度 todo
        :param N:  层数
        :param heads: 注意力头数 todo
        :param dropout: dropout比率
        :param 图向量尺寸: 图向量尺寸，默认6*6*2048，是resnet50最后一层卷积特征的尺寸 经过卷积的特征提取后，输入到本模型中，todo 在哪用
        '''
        super().__init__()
        self.图转= 全连接层(图向量尺寸,d_model) # 这里只是构建了一个线性变换层去接收resnet的输出



        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout) # transformer的解码器部分，看来没有使用编码器
        self.out = 全连接层(d_model, trg_vocab) # 前链接层，将解码器的输出映射到词表大小的维度上，进行分类预测 todo 这个最终输出是啥？

    def forward(self, 图向量 ,操作, trg_mask):
        图向量=self.图转(图向量)

        d_output = self.decoder(图向量,操作 , trg_mask)
        output = self.out(d_output)
        return output

class RESNET_Transformer(nn.Module):
    def __init__(self,  trg_vocab, d_model, N, heads, dropout,图向量尺寸=1000):
        super().__init__()
        self.图转= 全连接层(图向量尺寸,d_model)

        self.resnet = torchvision.models.resnet18(pretrained=False).eval().requires_grad_(True)

        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = 全连接层(d_model, trg_vocab)

    def forward(self, 图向量 , trg_mask):
        x=self.resnet(图向量).unsqueeze(0)
        图向量=self.图转(x)

        d_output = self.decoder(图向量,  trg_mask)
        output = self.out(d_output)
        output=output[:,-1,:]
        return output
def get_model(opt,  trg_vocab,model_weights='model_weights'):
    '''
    opt: 配置参数
    trg_vocab: 目标词表大小 todo 这个词表是啥？
    model_weights: 模型权重文件名称
    '''
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer( trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights is not None and os.path.isfile(opt.load_weights+'/'+model_weights):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/'+model_weights))
    else:
        量 = 0
        for p in model.parameters():
            if p.dim() > 1:
                #nn.init.xavier_uniform_(p)
                a=0
            长 = len(p.shape)
            点数 = 1
            for j in range(长):
                点数 = p.shape[j] * 点数

            量 += 点数
        print('使用参数:{}百万'.format(量/1000000))
    return model


def get_modelB(opt, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = RESNET_Transformer(trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)

    if opt.load_weights is not None and os.path.isfile(opt.load_weights + '/model_weightsB'):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weightsB'))
    else:
        量 = 0
        for p in model.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                a = 0
            长 = len(p.shape)
            点数 = 1
            for j in range(长):
                点数 = p.shape[j] * 点数

            量 += 点数
        print('使用参数:{}百万'.format(量 / 1000000))
    return model