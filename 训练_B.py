import torch
import torchvision
from PIL import Image
import numpy as np
import time
import json
from config import GPT2Config, TransformerConfig
from Batch import create_masks
from ModelA import get_model
import torch.nn.functional as F
from 取训练数据 import *
from 杂项 import *
import os
import random

训练数据保存目录='../训练数据样本'
if not os.path.exists(训练数据保存目录):
   os.makedirs(训练数据保存目录)
for root, dirs, files in os.walk('../训练数据样本'):
    if len(dirs)>0:
        break

词数词典路径="./json/词_数表.json"
数_词表路径="./json/数_词表.json"
if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
    词_数表, 数_词表 = 读出引索(词数词典路径, 数_词表路径)
with open(词数词典路径, encoding='utf8') as f:
    词数词典=json.load(f)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#
#
config = TransformerConfig()

# 构建模型
# 这里的130是指操作的词表大小 ，比如上移_推塔，下移_加点技能之类的操作指令
model = get_model(config,  130)
模型路径 = 'weights/model_weights'
model = model.cuda(device)
optimizer = torch.optim.Adam(model.parameters(), lr=6.25e-5, betas=(0.9, 0.98), eps=1e-9)


# 后续标记这个的作用 todo
分块大小=25 
游标大小=23
树枝=10

计数=0
time_start=time.time()
for j in range(100):
    random.shuffle(dirs)
    for 号 in dirs:
        预处理数据 = '../训练数据样本/'+号+'/图片_操作预处理数据2.npz'
        if os.path.isfile(预处理数据):
            npz文件 = np.load(预处理数据, allow_pickle=True) # 这里的数据是在采集后预处理好的数据
            图片张量np, 操作序列 = npz文件["图片张量np"], npz文件["操作序列"]
            # todo 后续标记这些的作用
            循环=True
            游标=0
            操作序列=np.insert(操作序列,0,128) # 在操作序列开头插入开始标记128，表示啥都不做

            操作_分_表 = [] # 操作序列分块列表
            目标输出_分_表 = [] # 目标输出序列分块列表，即操作序列向后移动一位，作为模型的目标输出计算损失
            图片_分_表 = [] # 图片序列分块列表

            # 以下训练可以看出，显示将图片特征进行提取，然后将操作序列进行分块（就是输入序列），最后进行模型训练
            while 循环:
                if 游标 + 分块大小 < 操作序列.shape[0]: # 如果当前游标加上分块大小还没有超过操作序列长度

                    操作_分 = 操作序列[游标:游标 + 分块大小]
                    目标输出_分 = 操作序列[游标 + 1:游标 + 1 + 分块大小]
                    图片_分 = 图片张量np[游标:游标 + 分块大小, :]
                    操作_分_表.append(操作_分)
                    目标输出_分_表.append(目标输出_分)
                    图片_分_表.append(图片_分)
                    游标 = 游标 + 游标大小
                else: # 如果当前游标加上分块大小超过了操作序列长度，则取最后一块数据，并退出循环
                    操作_分 = 操作序列[-分块大小 - 1:-1]
                    目标输出_分 = 操作序列[-分块大小:]

                    图片_分 = 图片张量np[-分块大小:, :]
                    操作_分_表.append(操作_分)
                    目标输出_分_表.append(目标输出_分)
                    图片_分_表.append(图片_分)
                    循环 = False

            循环=True
            i=0
            while 循环:
                # 这边是提取树枝个batch的序列数据进行训练
                if (i+1)*树枝<len(操作_分_表):
                    操作_分_枝=np.array(操作_分_表[i*树枝:(i+1)*树枝])
                    图片_分_枝 = np.array(图片_分_表[i * 树枝:(i + 1) * 树枝])
                    目标输出_分_枝 = np.array(目标输出_分_表[i * 树枝:(i + 1) * 树枝])
                else:
                    操作_分_枝 = np.array(操作_分_表[i * 树枝:len(操作_分_表)])
                    图片_分_枝 = np.array(图片_分_表[i * 树枝:len(图片_分_表)],dtype=np.float32)
                    目标输出_分_枝 = np.array(目标输出_分_表[i * 树枝:len(目标输出_分_表)])
                    循环 = False

                操作_分_torch=torch.from_numpy(操作_分_枝).cuda(device)
                图片_分_torch = torch.from_numpy(图片_分_枝).cuda(device)
                目标输出_分_torch = torch.from_numpy(目标输出_分_枝).cuda(device)

                # 创建掩码矩阵，避免看到未来信息
                src_mask, trg_mask = create_masks(操作_分_torch, 操作_分_torch, device)
                if 图片_分_torch.shape[0]!=操作_分_torch.shape[0]: # 防御性编程，理论不该出现
                    continue
                输出_实际_A = model(图片_分_torch,操作_分_torch ,trg_mask) # 得到预测的动作
                lin = 输出_实际_A.view(-1, 输出_实际_A.size(-1))
                optimizer.zero_grad()
                loss = F.cross_entropy(lin, 目标输出_分_torch.contiguous().view(-1), ignore_index=-1)
                if 计数 % 1 == 0:
                    print(loss)





                    time_end = time.time()
                    用时 = time_end - time_start

                    _, 抽样 = torch.topk(输出_实际_A, k=1, dim=-1)
                    抽样np = 抽样.cpu().numpy()
                    打印抽样数据(数_词表, 抽样np[0:1,:,:], 目标输出_分_torch[0,:])
                    print("用时{} 第{}轮 第{}张 号{}".format(用时, j, 计数, 号))
                if 计数 % 45060 == 0:
                    print('888')

                loss.backward()

                optimizer.step()
                计数=计数+1
                i=i+1
    torch.save(model.state_dict(), 'weights/model_weights')
    torch.save(model.state_dict(), 'weights/model_weights_P{}'.format(str(j)))





