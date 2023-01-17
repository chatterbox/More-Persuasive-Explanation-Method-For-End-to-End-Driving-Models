#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division
from pickle import APPEND
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import os
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
# defining customized Dataset class for Udacity

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import random
import pandas.io.common
from tqdm import tqdm

class Normalize_zhang(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class UdacityDataset(Dataset):
    def __init__(self, csv_file_path_list, root_dir, train_loader_flag, seq_len = 2, shuffle = True):
        self.each_video_frame_num_list = []
        self.train_loader_flag = train_loader_flag
        all_camera_csv = []
        for csv_file_path in csv_file_path_list:    
            camera_csv = pd.read_csv(csv_file_path, engine="python")
            all_camera_csv.append(camera_csv)
            each_video_frame_num = len(camera_csv)
            self.each_video_frame_num_list.append(each_video_frame_num)
        
        self.camera_csv = pd.concat(all_camera_csv, axis=0, ignore_index=True)
        
        
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.shuffle = shuffle
        
        self.all_sample_list = []
        self.sequence_sample_index_maker()
        # print("img_num",len(self.camera_csv))
        
        
    def label_data_analyzer(self):
        forward_counter = 0
        left_counter = 0
        right_counter = 0
        all_counter = 0
        for idx in tqdm(range(len(self.camera_csv))):
            Forward_action = self.camera_csv['Forward_action'].iloc[idx]
            Left_action = self.camera_csv['Left_action'].iloc[idx]
            Right_action = self.camera_csv['Right_action'].iloc[idx]
            all_counter = all_counter + 1
            if Forward_action == 1:
                forward_counter = forward_counter + 1
            if Left_action == 1:
                left_counter = left_counter + 1
            if Right_action == 1:
                right_counter = right_counter + 1
        print(forward_counter, right_counter, left_counter, all_counter)
        print(forward_counter/all_counter, right_counter/all_counter, left_counter/all_counter, all_counter)


        
    def get_each_video_frame_num_list(self):
        return self.each_video_frame_num_list
    
    def video_start_frame_index_list_maker(self):
        video_start_frame_index_list = []
        
        index_flag = 0
        for i in  self.each_video_frame_num_list:
            index_flag = index_flag + i
            video_start_frame_index_list.append(index_flag)
        
        video_start_frame_index_list.pop()
        
        self.video_start_frame_index_list = video_start_frame_index_list

    def __len__(self):
        
        return len(self.all_sample_list)
    
    def sequence_sample_index_maker(self):
        # self.video_start_frame_index_list_maker()
        video_now_length_counter = 0
        start_indices = []
        for video_index in range(len(self.each_video_frame_num_list)):
            assert self.seq_len <= self.each_video_frame_num_list[video_index], print("self.seq_len > self.each_video_frame_num_list[video_index]")
            for index in range(self.seq_len - 1, self.each_video_frame_num_list[video_index]):
                
                index = index + video_now_length_counter
                start_indices.append(index)
            video_now_length_counter = video_now_length_counter + self.each_video_frame_num_list[video_index]

        
        if self.shuffle:
            random.shuffle(start_indices)
        # print("start_indices", start_indices)
        all_sample_list = []
        for ind in start_indices:
            single_sample = []
            single_sample.extend(list(range(ind-self.seq_len+1, ind+1)))
            all_sample_list.append(single_sample)
        self.all_sample_list = all_sample_list
        # self.all_sample_list = all_sample_list[:int(len(all_sample_list)/2)]
            
            
        

    def read_data_single(self, idx):
        # print("camera_csv", len(self.camera_csv), idx)
        img_name = self.camera_csv['img_name'].iloc[idx]
        # print(img_name)
        video_name_split_list = self.camera_csv['img_name'].iloc[idx].split("-")[0:2]
        video_name = video_name_split_list[0] + "-" + video_name_split_list[1]
        
        path = os.path.join(self.root_dir, video_name, self.camera_csv['img_name'].iloc[idx])
        
        # image = io.imread(path)
        # image = cv2.resize(image, (320, 120), interpolation=cv2.INTER_CUBIC)
        image = Image.open(path)

        # color_jitter = transforms.ColorJitter(
        #         brightness=0.5,
        #         contrast=0.5,
        #         saturation=0.5,
        #         hue=0.5,
        #     )

        # transform_random_persepective = transforms.RandomPerspective(distortion_scale=0.5, p=1, fill=(0, 0, 255))

        # scale=(0.8, 1.0)
        # ratio=(0.75, 1.0)
        # transform_RR = transforms.RandomResizedCrop(size=(120, 320), scale=scale, ratio=ratio)

        # random_tranform = transforms.RandomApply(torch.nn.ModuleList([color_jitter, transform_RR, transform_random_persepective]), p=0.5)
        
        normalize_transform = Normalize_zhang(
                mean=[102.9801, 115.9465, 122.7717],
                std=[1., 1., 1.],
                to_bgr255=True
            )
        # normalize_transform = transforms.Normalize(
        #         mean=[102.9801, 115.9465, 122.7717],
        #         std=[1., 1., 1.],
        #         # to_bgr255=True
        #     )
        # normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        transform_method = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize_transform])
        
        # transform_method = transforms.Compose([transforms.ToPILImage(),transforms.Resize((120, 320)), transforms.ToTensor(), color_jitter, normalize_transform])
        image_transformed = transform_method(image)
        del image
        image = image_transformed

        Forward_action = self.camera_csv['Forward_action'].iloc[idx]
        Left_action = self.camera_csv['Left_action'].iloc[idx]
        Right_action = self.camera_csv['Right_action'].iloc[idx]

        Forward_action_t = torch.tensor(Forward_action)
        Left_action_t = torch.tensor(Left_action)
        Right_action_t = torch.tensor(Right_action)
        All_action =  torch.tensor([Forward_action_t, Left_action_t, Right_action_t])
        del Forward_action, Left_action, Right_action
        
        return image, All_action, img_name
    
    def read_data(self, idx):
        if isinstance(idx, list):
            data = None
            for i in idx:
                new_data = self.read_data(i)
                if data is None:
                    data = [[] for _ in range(len(new_data))] # data = [[img], [timestamp], [frame_id], [angle], [torque], [speed]]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data
            # len(data) = 6 6种信息, len(data[0]) = 5 每个信息保存5张图片
            
            '''
            test0 = torch.stack(data[3])  ===   test1 = torch.tensor(data[3])   这两种操作方法结果相同，将一个list的tensor转化成tensor的list，tensor的list其实就是tensor
            但是在高维的情况下：会发生这样的问题 ValueError: only one element tensors can be converted to Python scalars
            torch.tensor(data[stack_idx]) 只能处理单个tensor，将多个tensor单独值 变成一个 tensor的一维 list
            不能将一维的tensor list转化成二维的tensor list，这还是需要我们的 torch.stack
            '''
            
            # print(data[3]) # [tensor(0.0768, dtype=torch.float64), tensor(0.0750, dtype=torch.float64), tensor(0.0716, dtype=torch.float64), tensor(0.0690, dtype=torch.float64), tensor(0.0681, dtype=torch.float64)]
            for stack_idx in [0, 1]: # we don't stack timestamp and frame_id since those are string data
                data[stack_idx] = torch.stack(data[stack_idx])
                # data[stack_idx] = torch.tensor(data[stack_idx])
                # print("data[stack_idx]", data[stack_idx].shape)
                ''''
                有时候是
                data[stack_idx] torch.Size([5, 3, 480, 640])
                data[stack_idx] torch.Size([5])
                当到了外围的维度的时候：
                data[stack_idx] torch.Size([5, 5, 3, 480, 640])
                data[stack_idx] torch.Size([5, 5])

                注意data本身始终是包含6个元素的， 这六个元素是信息，这里进行的信息的整合是针对每个种类的信息的
                比如说， 将5*5 张图片的信息整合到一起，为了达成这一目的，首先把5个1放在一起（使用stack），变成[5, 3, 480, 640]
                然后再把5个5放在一起，变成[5, 5, 3, 480, 640]
                '''
                
            
            # print(data[3]) # tensor([0.0768, 0.0750, 0.0716, 0.0690, 0.0681], dtype=torch.float64)
            # print("len(data)", len(data))
            # exit()
            return data
        
        else:
            return self.read_data_single(idx)
    
    def __getitem__(self, idx):
        
        read_data_index = self.all_sample_list[idx]
        data = self.read_data(read_data_index)  
        # print(len(data), data[0].shape, data[3].shape)
        # exit()
        '''
        这个函数之所以要使用迭代的形式，就是因为他给的不是一个list，而是一个二维的list，使用迭代的结构可以保证
        无论给他的是多少维度的list，都可以不停的迭代        
        '''
        
        sample = {'image': data[0],
                  'Action': data[1],
                  'image_name': data[2]}
        
        del data
        # print(sample["timestamp"]) # 5*5 的列表
        # print(sample["angle"].shape) # 5*5 的tensor torch.Size([5, 5])
        
        return sample

