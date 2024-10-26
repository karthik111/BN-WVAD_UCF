import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from .normal_head import NormalHead
from .translayer import Transformer

class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

class WSAD(Module):
    def __init__(self, input_size, flag, args):
        super().__init__()
        self.flag = flag
        self.args = args
        
        self.ratio_sample = args.ratio_sample
        self.ratio_batch = args.ratio_batch
        
        self.ratios = args.ratios
        self.kernel_sizes = args.kernel_sizes

        # ratio = [16, 32], kernel_size = [1, 1, 1]
        self.normal_head = NormalHead(in_channel=512, ratios=args.ratios, kernel_sizes=args.kernel_sizes)
        self.embedding = Temporal(input_size,512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout = 0)
        self.step = 0
    
    def get_normal_scores(self, x, ncrops=None):    # input             (128, 200, 512)
        new_x  = x.permute(0, 2, 1)   # b x c x t   # permute           (128, 512, 200)
        
        outputs = self.normal_head(new_x)           # NormalHead
        # NormalHead outputs:  [(128, 32, 200), (128, 16, 200), (128, 1, 200)]
        normal_scores = outputs[-1]                 # normal score      (128, 1, 200)
        xhs = outputs[:-1]                          # X^h (BLS, SBS)    [(128, 32, 200), (128, 16, 200)]
        
        if ncrops: # n=1도 가능, ncrops=5(xd), =10(ucf) when normal score: (128, 1, 200)
            b = normal_scores.shape[0] // ncrops    # b = 128//1 = 128
            normal_scores = normal_scores.view(b, ncrops, -1).mean(1) # (128, 1, 200)->(128, 200)
        
        return xhs, normal_scores   # X^h [(128, 32, 200), (128, 16, 200)], n_score (128, 200)
    
    def get_mahalanobis_distance(self, feats, anchor, var, ncrops = None):
        # feats: (128, 32/16, 200)
        # anchor: ( ,32/16, )  시간차원에 대한 채널 32/16개의 평균
        # var: ( ,32/16, )     시간차원에 대한 채널 32/16개의 평균 
        # ncrop=1 or 5(xd), 10(ucf)
        
        
        # distance      (1280, 200) 논문에 나왔던 B x T 배열 여기서 뽑는구나 
        # batchnorm은 시간차원 정규화, mahalanobis는 피처차원 평균 및 분산을 고려하여 시간 차원의 크기의 합
        distance = torch.sqrt(torch.sum((feats - anchor[None, :, None]) ** 2 / var[None, :, None], dim=1))
        if ncrops: # n=1도 가능, ncrops=5(xd), =10(ucf)
            bs = distance.shape[0] // ncrops
            # b x t
            distance = distance.view(bs, ncrops, -1).mean(1)    # (128, 200)
        return distance                                         # (128, 200)
    
    def pos_neg_select(self, feats, distance, ncrops):
        # feats (128, 32/16, 200)
        # distance (128, 200)
        # ncrops=1 or =5(xd), =10(ucf)
        batch_select_ratio = self.ratio_batch   # 0.2
        sample_select_ratio = self.ratio_sample # 0.1
        bs, c, t = feats.shape                  # 128, 32/16, 200
        select_num_sample = int(t * sample_select_ratio)            # 200 * 0.1 = 20
        select_num_batch = int((bs // ncrops) // 2 * t * batch_select_ratio)    # (128 // 1) // 2 * 200 * 0.2 = 2560 
        feats = feats.view(bs//ncrops, ncrops, c, t).mean(1) # b x c x t    # feats (128, 32/16, 200)
        nor_distance = distance[:(bs // ncrops) // 2] # b x t                   # normal_distance (64, 200)
        nor_feats = feats[:(bs // ncrops) // 2].permute(0, 2, 1) # b x t x c    # normal_feats (64, 32/16, 200)
        abn_distance = distance[(bs // ncrops) // 2:] # b x t                   # abnormal_distance (64, 200)
        abn_feats = feats[(bs // ncrops) // 2:].permute(0, 2, 1) # b x t x c    # abnormal_feats (64, 32/16, 200)
        abn_distance_flatten = abn_distance.reshape(-1)             # abnormal_distance (12800, )
        abn_feats_flatten = abn_feats.reshape(-1, c)                # abnormal_feats (12800, 32/16)
        
        # abnormal distance (64, 200)에서 sample level 0.1 topk 비율 추출
        mask_select_abnormal_sample = torch.zeros_like(abn_distance, dtype=torch.bool)  # (64, )
        topk_abnormal_sample = torch.topk(abn_distance, select_num_sample, dim=-1)[1]
        mask_select_abnormal_sample.scatter_(1, topk_abnormal_sample, True)
        
        # 배치 레벨에서 전체 (12800, )에서 batch level 0.2 topk 비율 추출
        mask_select_abnormal_batch = torch.zeros_like(abn_distance_flatten, dtype=torch.bool) #(12800, )
        topk_abnormal_batch = torch.topk(abn_distance_flatten, select_num_batch, dim=-1)[1]
        mask_select_abnormal_batch.scatter_(0, topk_abnormal_batch, True)
        
        # sample level과 batch level 합치기
        mask_select_abnormal = mask_select_abnormal_batch | mask_select_abnormal_sample.reshape(-1)
        select_abn_feats = abn_feats_flatten[mask_select_abnormal]  # (num_select, 32/16)
        
        # 총 뽑은 개수
        num_select_abnormal = torch.sum(mask_select_abnormal)
        
        k_nor = int(num_select_abnormal / ((bs  // ncrops) // 2)) + 1
        topk_normal_sample = torch.topk(nor_distance, k_nor, dim=-1)[1]    # 총 뽑은 개수//batch_size 해서 batch 별로 추출
        select_nor_feats = torch.gather(nor_feats, 1, topk_normal_sample[..., None].expand(-1, -1, c)) #(64, 200, 32/16)
        select_nor_feats = select_nor_feats.permute(1, 0, 2).reshape(-1, c) # (12800, 32/16)
        select_nor_feats = select_nor_feats[:num_select_abnormal]           # (num_select, 32/16)
        
        return select_nor_feats, select_abn_feats       # (num_select, 32/16), (num_select, 32/16)

    def forward(self, x):               #                           (128, 200, 1024)
        #print(x.size())
        if len(x.size()) == 4:          # crop concat               (128, 10, 200, 1024)
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)  # crop concat               (1280, 200, 1024) 
        else:
            b, t, d = x.size()          # 128, 200, 1024                           
            n = 1                       # n=1
        ## temporal embedding
        x = self.embedding(x)           # 512 embedding             (128, 200, 512)
        ## transformer
        x = self.selfatt(x)             # feat enhfance(attn+ff)    (128, 200, 512)             
        
        # ncrop한 데이터의 경우 n=5, 10을 넣으므로 가능(이미 되있긴함) (보통은 n=1로 되어있을 거임)
        normal_feats, normal_scores = self.get_normal_scores(x, n)
        # normal_feats: [(128, 32, 200), (128, 16, 200)]
        # normal_score: (128, 200)
        
        # BLS 및 SBS 시 running_mean과 running_var 추출 각 2개
        anchors = [bn.running_mean for bn in self.normal_head.bns]  # [running_mean1, running_mean2]
        variances = [bn.running_var for bn in self.normal_head.bns] # [running_var1, running_var2]

        # distances     [(128, 200), (128, 200)]
        distances = [self.get_mahalanobis_distance(normal_feat, anchor, var, ncrops=n) for normal_feat, anchor, var in zip(normal_feats, anchors, variances)]
        #output_distance = distances.copy()

        if self.flag == "Train":
            
            select_normals = []
            select_abnormals = []
            for feat, distance in zip(normal_feats, distances):     # (128, 32/16, 200), (128, 200)
                # (num_select, 32/16), (num_select, 32/16)
                select_feat_normal, select_feat_abnormal = self.pos_neg_select(feat, distance, n)
                # 둘 다 [(num_select, 32), (num_select, 16)]
                select_normals.append(select_feat_normal[..., None])
                select_abnormals.append(select_feat_abnormal[..., None])

            bn_resutls = dict(
                anchors = anchors,
                variances = variances,
                select_normals = select_normals,
                select_abnormals = select_abnormals, 
            )

            return {
                    'pre_normal_scores': normal_scores[0:b // 2],
                    'bn_results': bn_resutls,
                }
        else:

            distance_sum = sum(distances)       # (128, )

            return distance_sum * normal_scores # (128, ) * (128, 200) = (128, 200)