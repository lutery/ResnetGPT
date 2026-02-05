import torch
import torch.nn as nn
import torch.nn.functional as F

class myResnet(nn.Module):
    '''
    通过主动调用resnet的各个层，来获取中间特征
    这样可以同时获得全局特征和局部特征
    '''

    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=6):
        x = img

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0) # todo 这个的作用
        
        return fc, att

