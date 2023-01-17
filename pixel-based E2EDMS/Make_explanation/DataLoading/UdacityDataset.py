#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division
import pandas as pd
import os
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
# defining customized Dataset class for Udacity
import torch
import random
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
    def __init__(self, csv_file_path_list, root_dir, seq_len = 2, shuffle = True):
        self.each_video_frame_num_list = []
        
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
        print("img_num",len(self.camera_csv))
        
        
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
        video_name_split_list = self.camera_csv['img_name'].iloc[idx].split("-")[0:2]
        video_name = video_name_split_list[0] + "-" + video_name_split_list[1]
        
        path = os.path.join(self.root_dir, video_name, self.camera_csv['img_name'].iloc[idx])
        
        
        image = Image.open(path)
        
        normalize_transform = Normalize_zhang(
                mean=[102.9801, 115.9465, 122.7717],
                std=[1., 1., 1.],
                to_bgr255=True
            )
        
        transform_method = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize_transform])
        
        
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
            
            for stack_idx in [0, 1]: # we don't stack timestamp and frame_id since those are string data
                data[stack_idx] = torch.stack(data[stack_idx])
                
            return data
        
        else:
            return self.read_data_single(idx)
    
    def __getitem__(self, idx):
        
        read_data_index = self.all_sample_list[idx]
        data = self.read_data(read_data_index)  
        
        
        sample = {'image': data[0],
                  'Action': data[1],
                  'image_name': data[2]}
        
        del data
        
        
        return sample

