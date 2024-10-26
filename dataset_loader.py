import torch
import torch.utils.data as data
import os
import numpy as np
import utils 


"""_summary_

Returns:
    _type_: _description_
"""
class XDVideo(data.DataLoader):
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        ## video feature를 가져올 path 설정
        if seed >= 0:
            utils.set_seed(seed)
        # 여기도 seed 해제
        self.data_path=root_dir
        self.mode=mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        
        self.feature_path = self.data_path
        split_path = os.path.join("list",'XD_{}.list'.format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        
        ## train에서 normal과 abnormal video feature의 분리
        ## 저장된 list의 순서에 따라서 분리하므로 순서가 바뀌면 slice 범위도 변경해줘야 함
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[9525:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:9525]
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        ## 해당하는 index의 video label과 video feature을 반환
        ## get_data 함수에서 (I3D_temporal, C (feature dim)) -> 을 (T(=num_segments, 200), C)로 변환하여 반환
        data,label = self.get_data(index)
        return data, label


    """_summary_
        해당 index의 video label과 video feature 반환
        해당 video feature을 self.num_segments 만큼의 temporal 차원으로 분리하여 video feature을 반환
    """
    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        if "_label_A" not in vid_name:
            label=1  
        video_feature = np.load(os.path.join(self.feature_path, vid_name )).astype(np.float32)          # (64, t, 1024)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)            # (64, 200, 1024)

            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)

            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature     # (64, 200, 1024)
        return video_feature, label    
    

class UCFVideo(data.DataLoader):
    def __init__(self, root_dir, mode, num_segments, len_feature, seed=-1, is_normal=None):
        ## video feature를 가져올 path 설정
        # if seed >= 0:
        #     utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        
        self.feature_path = self.data_path
        split_path = os.path.join("list",'UCF_{}_V1.list'.format(self.mode))
        #split_path = os.path.join("list", "new50-{}.list".format(self.mode))
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        
        for line in split_file:
            self.vid_list.append(line.split('\n'))
        split_file.close()
        
        self.is_normal = is_normal
        
        ## train에서 normal과 abnormal video feature의 분리
        ## 저장된 list의 순서에 따라서 분리하므로 순서가 바뀌면 slice 범위도 변경해줘야 함
        
        ### !!!! 여기 부분 list 넣은 후 수정
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[8100:]
                
            elif is_normal is False:
                self.vid_list = self.vid_list[:8100]
                
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        ## 해당하는 index의 video label과 video feature을 반환
        ## get_data 함수에서 (I3D_temporal, C (feature dim)) -> 을 (T(=num_segments, 200), C)로 변환하여 반환
        data,label = self.get_data(index)
        return data, label


    """_summary_
        해당 index의 video label과 video feature 반환
        해당 video feature을 self.num_segments 만큼의 temporal 차원으로 분리하여 video feature을 반환
    """
    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        #if "_label_A" not in vid_name:
        if "Normal" not in vid_name: ##self.is_normal==False랑 다르다!!
            label=1  
        # if self.is_normal == False:
        #     label=1  
        video_feature = np.load(os.path.join(self.feature_path, vid_name)).astype(np.float32)          # (64, t, 1024)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments, self.len_feature)).astype(np.float32)            # (64, 200, 1024)

            sample_index = np.linspace(0, video_feature.shape[0], self.num_segments+1, dtype=np.uint16)

            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature     # (64, 200, 1024)
        return video_feature, label    

