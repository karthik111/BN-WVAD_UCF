import torch
import torch.nn as nn

class NormalHead(nn.Module):
    def __init__(self, in_channel=512, ratios=[16, 32], kernel_sizes=[1, 1, 1]):
        super(NormalHead, self).__init__()
        self.ratios = ratios
        self.kernel_sizes = kernel_sizes

        self.build_layers(in_channel)
        
    def build_layers(self, in_channel):
        ratio_1, ratio_2 = self.ratios
        self.conv1 = nn.Conv1d(in_channel, in_channel // ratio_1, 
                               self.kernel_sizes[0], 1, self.kernel_sizes[0] // 2)
        self.bn1 = nn.BatchNorm1d(in_channel // ratio_1)
        self.conv2 = nn.Conv1d(in_channel // ratio_1, in_channel // ratio_2, 
                               self.kernel_sizes[1], 1, self.kernel_sizes[1] // 2)
        self.bn2 = nn.BatchNorm1d(in_channel // ratio_2)
        self.conv3 = nn.Conv1d(in_channel // ratio_2, 1, 
                               self.kernel_sizes[2], 1, self.kernel_sizes[2] // 2)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.bns = [self.bn1, self.bn2]

    def forward(self, x):   # ratio = [16, 32], kernel_size = [1, 1, 1], input          (128, 512, 200)
        '''
        x: BN * C * T
        return BN * C // 64 * T and BN * 1 * T
        '''
        outputs = []
        ## SLS, SBS 적용하는 Conv1d 부분, BatchNorm1d(channel)이므로 channel에 딸린 시간 차원 200에 대하여 정규화
        x = self.conv1(x)                                               # select 32     (128, 32, 200)
        outputs.append(x)           
        x = self.conv2(self.act(self.bn1(x)))                           # select 16     (128, 16, 200)
        outputs.append(x)
        x = self.sigmoid(self.conv3(self.act(self.bn2(x))))             # select 1      (128, 1, 200)
        outputs.append(x)
        return outputs               # outputs [(128, 32, 200), (128, 16, 200), (128, 1, 200)]
