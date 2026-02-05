import torch
from torchtext import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, device):
    '''
    本方法用于创建防止模型看到未来信息的掩码矩阵

    :param size: 操作序列的长度
    :param device: 设备
    '''

    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8') # 创建一个上三角矩阵，主对角线以上的元素为1，其余为0
    variable = Variable
    np_mask = variable(torch.from_numpy(np_mask) == 0) # 转换为布尔类型的tensor，1变为True，0变为False
    np_mask = np_mask.cuda(device)
    return np_mask

def create_masks(src, trg, device):
    '''
    本方法中只创建了目标序列的掩码，因为模型中只有解码器部分
    
    :param src: 源序列
    :param trg: 目标序列
    :param device: 设备
    '''
    
    src_mask = (src != -1).unsqueeze(-2) # 源序列的掩码，-1表示填充位置

    if trg is not None:
        trg_mask = (trg != -1).unsqueeze(-2) # 目标序列的掩码，-1表示填充位置，我记得第一次操作序列的默认值时-1，估计是将无效位置填充为-1
        trg_mask.cuda(device)
        size = trg.size(1) # get seq_len for matrix 获取操作序列的长度
        np_mask = nopeak_mask(size, device) # 防止模型看到未来信息的掩码
        trg_mask = trg_mask & np_mask # 将两个掩码进行与操作，得到最终的目标序列掩码，防止模型看到未来信息且屏蔽填充位置
    else:
        trg_mask = None
    return src_mask, trg_mask

# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
