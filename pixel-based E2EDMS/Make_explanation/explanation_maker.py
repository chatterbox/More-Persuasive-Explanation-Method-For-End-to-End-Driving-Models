#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from DataLoading import UdacityDataset_test as UD
from DataLoading import UdacityDataset_zhang_for_obejct_explanation as UD2

from model import Convolution3D_LSTM_transfer as CNN3D_LSTM
from model import Convolution2D_LSTM_transfer as CNN2D_LSTM
from model import Convolution3D_only_transfer as CNN3D

import cv2
import numpy as np
# get_ipython().run_line_magic('run', 'Visualization.ipynb')
import json

if torch.cuda.is_available() == True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")

class gradual_img_for_experiment_drawer():
    def __init__(self, original_img_name, target_heatmap_folder, img_folder_dir_path, objects_alpha, lane_flag_list, objects_position_list, folder_name):
        self.original_img_path_list = []
        self.original_img_name = original_img_name
        time_serial_list = ["-0000051.jpg", "-0000052.jpg"]
        self.target_heatmap_folder = target_heatmap_folder

        self.objects_alpha = objects_alpha

        # this object is a lane or not
        self.lane_flag_list = lane_flag_list
        self.objects_position_list = objects_position_list
        
        for time_serial in time_serial_list:
            img_name = original_img_name + time_serial
            img_path = os.path.join(img_folder_dir_path, folder_name, img_name)
            self.original_img_path_list.append(img_path)
        self.step_number = 4
        self.start_divide_num = 20
        
    def decsending_order_for_alpha_and_info_list_for_3(self):
        self.index_list = np.argsort(self.objects_alpha)[::-1]

        self.objects_alpha = np.array(self.objects_alpha)
        self.objects_position_list = np.array(self.objects_position_list)
        self.lane_flag_list = np.array(self.lane_flag_list)
        
        self.objects_position_list = self.objects_position_list[self.index_list]
        self.objects_alpha = self.objects_alpha[self.index_list]
        self.lane_flag_list = self.lane_flag_list[self.index_list]

    def main_drawer(self):
        # sort the object information by the importance of objects
        self.decsending_order_for_alpha_and_info_list_for_3()

        # Here, we make a mask as big as the original image, each pixel in this mask is the importance of this pixel (based on each object importance and object position info)
        # The reason we need this heatmap, In case when we made partially shown image based on the object position, but the object number is less then 4, then we cannot make enough partially shown image
        # Therefore, if the object number is 3, then the forth grid is calculated based on the pixel importance sum in th grid (which is also the method used in pixel-level explanation data)
        # Note, the situation that "the object number in a image is less than 4" is very rarely, The heatmap here is a safe check
        single_heat_map_drawer_for_gradual = heat_map_drawer(None, None, self.objects_alpha, self.lane_flag_list, self.objects_position_list)
        single_heat_map_drawer_for_gradual.make_attention_map()
        heatmap = single_heat_map_drawer_for_gradual.attention_map
    
        # we divide the image into 8*8 grid, we calculate the most important 4 grid (First based on object, if not enough, use heatmap)
        grid_object = puzzle_simulation(heatmap, self.objects_alpha, self.objects_position_list, self.lane_flag_list, self.original_img_name)
        grid_box_size = grid_object.get_grid_box_size()
        grid_coordinate_list = grid_object.find_match()
        
        gradual_counter = 0
        
        # Make the simulation data (Objective perisuasibility eavalution data)
        for serial_num in range(len(grid_coordinate_list)):
            filter_map = filter_map_translator(grid_coordinate_list, grid_box_size, serial_num)
            time_counter = 0
            gradual_counter = gradual_counter + 1
            for original_img_path in self.original_img_path_list:
                time_counter = time_counter + 1
                original_image = cv2.imread(original_img_path, 3)
                filtered_original_image = filter_original_img(original_image, filter_map)
                target_img_name_for_save = self.original_img_name + "_" + str(gradual_counter) + "_" + str(
                    time_counter) + ".jpg"
                target_img_path = os.path.join(self.target_heatmap_folder, target_img_name_for_save)
                cv2.imwrite(target_img_path, filtered_original_image)

def filter_original_img(original_image, filter_map):
    height, width, channels = original_image.shape
    filtered_original_image = original_image.copy()
    # color = [128, 128, 128]
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                if filter_map[i][j] == 0:
                    filtered_original_image[:, :, channel][i][j] = 128
    return filtered_original_image

class puzzle_simulation():
    def __init__(self, heatmap, objects_alpha, objects_info, lane_flag_list, target_img_name, puzzle_top_num = 4):
        self.top_k = puzzle_top_num
        self.target_img_name = target_img_name
        self.heatmap = heatmap
        self.objects_alpha = objects_alpha
        self.objects_info = objects_info
        self.lane_flag_list = lane_flag_list


        self.object_center_point_tuple_list = self.object_center_point_calculator()

        self.puzzle_top_num = puzzle_top_num
        self.image_shape = (1280, 720)
        self.grid_shape_divide_num = (8, 8)

        self.grid_box_size = (
            int(self.image_shape[0] / self.grid_shape_divide_num[0]), int(self.image_shape[1] / self.grid_shape_divide_num[1]))

        self.grid_list = self.grid_maker()

    def find_match(self):
        grid_match_list = []
        # We first calculate  the grid based on object center point
        for each_object_center_point_tuple in self.object_center_point_tuple_list:
            for each_grid in self.grid_list:
                x_coordinate = each_object_center_point_tuple[0]
                y_coordinate = each_object_center_point_tuple[1]

                each_grid_x = each_grid[0]
                each_grid_y = each_grid[1]
                box_size_x = self.grid_box_size[0]
                box_size_y = self.grid_box_size[1]
                if x_coordinate >= each_grid_x and x_coordinate <= each_grid_x + box_size_x and y_coordinate >= each_grid_y and y_coordinate <= each_grid_y + box_size_y:
                    if each_grid not in grid_match_list and len(grid_match_list) < 4:
                        grid_match_list.append(each_grid)

        # If less then 4, use pixel importantance to full fill the 4 partially shown images
        if len(grid_match_list) < self.top_k:
            grid_coordinate_list = self.find_top_k_grid()
            for grid_coordinate in grid_coordinate_list:
                grid_coordinate_tuple = (grid_coordinate[0], grid_coordinate[1])
                if grid_coordinate_tuple not in grid_match_list:
                    grid_match_list.append(grid_coordinate_tuple)
                if len(grid_match_list) >= self.top_k:
                    break
        assert len(grid_match_list) == self.top_k, ("len(grid_match_list) != self.top_k", self.target_img_name, grid_match_list)

        return grid_match_list




    def object_center_point_calculator(self):
        # Note, the object center point calculation method for lane is different for BBox 
        # For BBox, the center point is obvious, but for lane, we think the nearest grid(point) is better
        object_center_point_tuple_list = []
        for i in range(len(self.objects_info)):
            object_box_info = self.objects_info[i]
            lane_flag = self.lane_flag_list[i]

            if lane_flag == True:
                x1 = int(object_box_info[0])
                x2 = int(object_box_info[2])
                y1 = int(object_box_info[1])
                y2 = int(object_box_info[3])
                if y1 < y2:
                    nearest_point = (x2, y2)
                else:
                    nearest_point = (x1, y1)
                object_center_point_tuple_list.append( nearest_point )
            else:
                x1 = int(object_box_info[0])
                x2 = int(object_box_info[2])
                y1 = int(object_box_info[1])
                y2 = int(object_box_info[3])
                center_point_x = int((x1 + x2) / 2)
                center_point_y = int((y1 + y2) / 2)
                object_center_point_tuple_list.append( (center_point_x, center_point_y) )
        return object_center_point_tuple_list

    def get_grid_box_size(self):
        return self.grid_box_size

    def grid_maker(self):
        grid_list = []

        window_slide_counter_left_up_y = 0
        grid_box_size = self.grid_box_size

        for height_grid in range(1, self.grid_shape_divide_num[0] + 1):
            window_slide_counter_left_up_x = 0
            for width_grid in range(1, self.grid_shape_divide_num[1] + 1):
                window_slide_counter_left_up_point = (window_slide_counter_left_up_x, window_slide_counter_left_up_y)
                grid_list.append( window_slide_counter_left_up_point )

                window_slide_counter_left_up_x = int(grid_box_size[0] * width_grid)

            window_slide_counter_left_up_y = int(grid_box_size[1] * height_grid)

        # grid_list[0] = ((0, 0)) (left_up_point_coordinate)
        return grid_list

    def decsending_order_for_alpha_and_info_list(self, alpha, info):
        index_list = np.argsort(alpha)[::-1]
        alpha = np.array(alpha)
        info = np.array(info)
        sorted_alpha = alpha[index_list]
        sorted_info = info[index_list]
        return sorted_alpha, sorted_info

    def find_top_k_grid(self):
        grid_importance_list = []
        for each_grid in self.grid_list:
            grid_importance_list.append( self.grid_importance_calculator(each_grid) )
        sorted_grid_importance_list, sorted_grid_list = self.decsending_order_for_alpha_and_info_list(grid_importance_list, self.grid_list)
        return sorted_grid_list[:self.top_k]
    def grid_importance_calculator(self, grid_coordinate):
        importance_counter = 0
        counter_flag = 0

        for i in range(grid_coordinate[0], grid_coordinate[0] + self.grid_box_size[0]):
            for j in range(grid_coordinate[1], grid_coordinate[1] + self.grid_box_size[1]):
                counter_flag = counter_flag + 1
                importance_counter = importance_counter + self.heatmap[j][i]

        return importance_counter / counter_flag

def filter_map_translator(grid_coordinate_list, grid_box_size, top_range):
    filter_map = np.zeros((720, 1280))
    for serial_num in range(top_range + 1):
        grid_coordinate = grid_coordinate_list[serial_num]
        for i in range(grid_coordinate[0], grid_coordinate[0] + grid_box_size[0]):
            for j in range(grid_coordinate[1], grid_coordinate[1] + grid_box_size[1]):
                filter_map[j][i] = 1

    return filter_map

        
class heat_map_drawer():
    def __init__(self, original_img_path, target_heatmap_path, objects_alpha, lane_flag_list, objects_position_list):
        self.original_img_path = original_img_path
        self.original_img = cv2.imread(original_img_path)
        self.target_heatmap_path = target_heatmap_path
        self.objects_alpha = objects_alpha
        self.lane_flag_list = lane_flag_list
        self.objects_position_list = objects_position_list
        self.attention_map = np.zeros((720, 1280))
        self.ratio_heatmap = 0.3
        
    def acsending_order_for_alpha_and_info_list(self):
        index_list = np.argsort(self.objects_alpha)
        self.objects_alpha = np.array(self.objects_alpha)
        self.objects_position_list = np.array(self.objects_position_list)
        self.lane_flag_list = np.array(self.lane_flag_list)

        self.objects_alpha = self.objects_alpha[index_list]
        self.objects_position_list = self.objects_position_list[index_list]
        self.lane_flag_list = self.lane_flag_list[index_list]
    def make_attention_with_each_box_info(self, serial_num):
        horizontal_line_start = int(self.objects_position_list[serial_num][0])
        horizontal_line_end = int(self.objects_position_list[serial_num][2])

        longitudinal_line_start = int(self.objects_position_list[serial_num][1])
        longitudinal_line_end = int(self.objects_position_list[serial_num][3])
        
        for horizontal_line in range(horizontal_line_start, horizontal_line_end):
            for longitudinal_line in range(longitudinal_line_start, longitudinal_line_end):
                self.attention_map[longitudinal_line, horizontal_line] = self.objects_alpha[serial_num]
    def make_attention_with_each_start_end_point(self, serial_num):
        blank_img = np.zeros(self.attention_map.shape)
        (x1, y1) = int(self.objects_position_list[serial_num][0]), int(self.objects_position_list[serial_num][1])
        (x2, y2) = int(self.objects_position_list[serial_num][2]), int(self.objects_position_list[serial_num][3])
        color = (255)
        thickness = 30
        lineType = 4
        cv2.line(blank_img, (x1, y1), (x2, y2), color,thickness, lineType)
        line_pixel_list = []
        for i in range(blank_img.shape[0]):
            for j in range(blank_img.shape[1]):
                if blank_img[i][j] == 255:
                    line_pixel_list.append((i,j))
        for i in line_pixel_list:
            self.attention_map[i[0],i[1]] = self.objects_alpha[serial_num]

    def make_attention_map(self):
        assert len(self.objects_alpha) == len(self.objects_position_list), ("len(objects_alpha) != len(objects_box_info)")
        self.acsending_order_for_alpha_and_info_list()
        for serial_num in range(len(self.objects_alpha)):
            if self.lane_flag_list[serial_num] == False:
                self.make_attention_with_each_box_info(serial_num)
            else:
                self.make_attention_with_each_start_end_point(serial_num)
    
        

def save_original_action_label(cnn3d_model, target_csv_file_path, cross_validation_counter, img_folder_dir_path):
    csv_folder_path = "/home0/zhangc/Dataset/BDD_100K_FLR/cross_validation_500"


    each_csv_folder_path_list = []
    for i in range(5):
        each_csv_folder_path = os.path.join(csv_folder_path, "regularized_csv_folder_500_" + str(i))
        each_csv_folder_path_list.append(each_csv_folder_path)

    test_csv_folder_path = [each_csv_folder_path_list[cross_validation_counter]]

    test_csv_file_path_list = []

    Batch_size = 1
    Seq_len = 2 
    
    for each_csv_folder in test_csv_folder_path:
        for i in os.listdir(each_csv_folder):
            csv_file_path = os.path.join(each_csv_folder, i)
            test_csv_file_path_list.append(csv_file_path)
    
    
    explanation_dataset = UD.UdacityDataset(test_csv_file_path_list, img_folder_dir_path, train_loader_flag = True, seq_len = Seq_len, shuffle = True)
    
    
    loader_3dcnn = DataLoader(explanation_dataset, batch_size = Batch_size)

    
    image_label_Dict = dict()

    with torch.no_grad():  
        cnn3d_model.eval()
        for testing_sample in tqdm(loader_3dcnn):
            # print("testing_sample.keys()", testing_sample.keys())
            testing_sample['image'] = testing_sample['image'].permute(0,2,1,3,4)
            
            param_values = [v for v in testing_sample.values()]
            
            angle = param_values[1]
            labels = angle.to(device)
            labels = labels[:,-1,:,]
            labels = labels.float()
            labels = labels.cpu().data.numpy()

            image = param_values[0]

            image = image.to(device)   
            
            predict_action = cnn3d_model(image)

            predict_action = predict_action.cpu().data.numpy()

            
            del param_values, image

            predict_action = predict_action.reshape(3)
            image_label_Dict[testing_sample['image_name'][-1][0]] = predict_action

    save_action_output_dict_as_csv(target_csv_file_path, image_label_Dict)
    return image_label_Dict

def read_action_label_dict(csv_file_path):
    camera_csv = pd.read_csv(csv_file_path, engine="python")
    image_label_Dict = dict()
    for idx in range(len(camera_csv)):
        original_image_name = camera_csv['image_name'].iloc[idx]
        Forward_label = camera_csv['Forward'].iloc[idx]
        Left_label = camera_csv['Left'].iloc[idx]
        Right_label = camera_csv['Right'].iloc[idx]
        image_label_Dict[original_image_name] = [Forward_label, Left_label, Right_label]
    return image_label_Dict

def save_action_output_dict_as_csv(target_csv_file_path, dict):
    each_line_list = []
    
    for key in dict:
        each_line = [key, dict[key][0], dict[key][1], dict[key][2]]
        each_line_list.append(each_line)

    df = pd.DataFrame(each_line_list, columns=["image_name", "Forward", "Left", "Right"])
    df.to_csv(target_csv_file_path, index = None)

def save_dict_as_csv(target_csv_file_path, dict):
    each_line_list = []
    
    for key in dict:
        each_line = [key, dict[key][0], dict[key][1], dict[key][2][0][0], dict[key][2][0][1], dict[key][2][0][2], dict[key][2][0][3], dict[key][2][1][0], dict[key][2][1][1], dict[key][2][1][2], dict[key][2][1][3]]
        each_line_list.append(each_line)

    df = pd.DataFrame(each_line_list, columns=["image_name", "loss_info", "lane_flag", "time_step_1_x1", "time_step_1_y1", "time_step_1_x2", "time_step_1_y2", "time_step_2_x1", "time_step_2_y1", "time_step_2_x2", "time_step_2_y2"])
    df.to_csv(target_csv_file_path, index = None)

def diff_calcu(list1, list2):
    assert len(list1) == len(list2), (len(list1) != len(list2))
    loss_counter = 0
    for i in range(len(list1)):
        
        loss_counter = loss_counter + abs(list1[i] - list2[i])
    return loss_counter

def make_save_loss_object_info_dict(cnn3d_model, loss_object_info_dict_path, original_action_label_dict):
    Batch_size = 1
    Seq_len = 2
    target_img_folder_path =  "/home0/zhangc/Dataset/BDD_100K_FLR/explanation/explanation_500__gray_out_img_folder"
    target_csv_folder_for_each_object_path =  "/home0/zhangc/Dataset/BDD_100K_FLR/explanation/explanation_500__gray_out_csv_folder"

    csv_file_path_list = []

    for i in os.listdir(target_csv_folder_for_each_object_path):
        csv_file_path = os.path.join(target_csv_folder_for_each_object_path, i)
        csv_file_path_list.append(csv_file_path)

    explanation_dataset = UD2.UdacityDataset(csv_file_path_list, target_img_folder_path, seq_len = Seq_len, shuffle = True)

    loader_3dcnn = DataLoader(explanation_dataset, batch_size = Batch_size)

    
    counter = 0
    gray_out_img_dict_for_loss_and_info = dict()

    cnn3d_model.eval()
    for testing_sample in tqdm(loader_3dcnn):
        
        testing_sample['image'] = testing_sample['image'].permute(0,2,1,3,4)
        
        counter = counter + 1
        param_values = [v for v in testing_sample.values()]

        
        image = param_values[0]

        image = image.to(device)   
        
        predict_action = cnn3d_model(image)
        predict_action = predict_action.cpu().data.numpy()
        del param_values, image
        predict_action = predict_action.reshape(3)
        
        gray_out_img_name = os.path.split(testing_sample['image_name'][-1][0])
        original_img_name_first_part = gray_out_img_name[-1]
        original_img_name_second_part = original_img_name_first_part.split("_")[0]
        original_img_name = original_img_name_second_part + "-0000052" + ".jpg"

        gray_out_img_name_for_dict = gray_out_img_name[0]
        
        if original_img_name in original_action_label_dict:
            original_action_label = original_action_label_dict[original_img_name]
        else:
            continue
        loss = diff_calcu(predict_action, original_action_label)
        
  
        position_info_list_list = []

        if testing_sample['correspond_object_info'][-1][0][0] == "{":
            lane_flag = False
            temp_str_image_1 = testing_sample['correspond_object_info'][0][0].replace('\'',"\"")
            temp_str_image_2 = testing_sample['correspond_object_info'][-1][0].replace('\'',"\"")
            convert_dict_image_1 = json.loads(temp_str_image_1)
            convert_dict_image_2 = json.loads(temp_str_image_2)

            position_info_list = [convert_dict_image_1["x1"], convert_dict_image_1["y1"], convert_dict_image_1["x2"], convert_dict_image_1["y2"]]
            position_info_list_list.append(position_info_list)
            position_info_list = [convert_dict_image_2["x1"], convert_dict_image_2["y1"], convert_dict_image_2["x2"], convert_dict_image_2["y2"]]
            position_info_list_list.append(position_info_list)

        else:
            lane_flag = True

            get_in_num_list_image_1 = testing_sample['correspond_object_info'][0][0][1:-1].split(",")
            get_in_num_list_image_2 = testing_sample['correspond_object_info'][-1][0][1:-1].split(",")
            num1_image_1 = int(get_in_num_list_image_1[0][1:])
            num2_image_1 = int(get_in_num_list_image_1[1][:-1])
            num3_image_1 = int(get_in_num_list_image_1[2][2:])
            num4_image_1 = int(get_in_num_list_image_1[3][:-1])

            num1_image_2 = int(get_in_num_list_image_2[0][1:])
            num2_image_2 = int(get_in_num_list_image_2[1][:-1])
            num3_image_2 = int(get_in_num_list_image_2[2][2:])
            num4_image_2 = int(get_in_num_list_image_2[3][:-1])

            
            position_info_list_list = [[num1_image_1, num2_image_1, num3_image_1, num4_image_1], [num1_image_2, num2_image_2, num3_image_2, num4_image_2]]       

        gray_out_img_dict_for_loss_and_info[gray_out_img_name_for_dict] = [loss, lane_flag, position_info_list_list]

    save_dict_as_csv(loss_object_info_dict_path, gray_out_img_dict_for_loss_and_info)
    


def make_attention_list_by_loss_object_info_dict(image_object_loss_dict_for_500, loss_object_info_dict_path):
    camera_csv = pd.read_csv(loss_object_info_dict_path, engine="python")
    loss_object_info_list = []
    # image_object_loss_dict_for_500 = dict()
    image_name_list = []
    for idx in range(len(camera_csv)):
        original_image_name = camera_csv['image_name'].iloc[idx].split("_")[:-1][0]
        loss_info = camera_csv['loss_info'].iloc[idx]
        lane_flag = camera_csv['lane_flag'].iloc[idx]
        
        time_step_1_x1 = camera_csv['time_step_1_x1'].iloc[idx]
        time_step_1_y1 = camera_csv['time_step_1_y1'].iloc[idx]
        time_step_1_x2 = camera_csv['time_step_1_x2'].iloc[idx]
        time_step_1_y2 = camera_csv['time_step_1_y2'].iloc[idx]

        time_step_2_x1 = camera_csv['time_step_2_x1'].iloc[idx]
        time_step_2_y1 = camera_csv['time_step_2_y1'].iloc[idx]
        time_step_2_x2 = camera_csv['time_step_2_x2'].iloc[idx]
        time_step_2_y2 = camera_csv['time_step_2_y2'].iloc[idx]
        
        if original_image_name in image_object_loss_dict_for_500:
            temp_list = image_object_loss_dict_for_500[original_image_name]
            temp_list.append( [loss_info, lane_flag, [time_step_1_x1, time_step_1_y1, time_step_1_x2, time_step_1_y2], [time_step_2_x1, time_step_2_y1, time_step_2_x2, time_step_2_y2]] )
        else:
            image_object_loss_dict_for_500[original_image_name] = [[loss_info, lane_flag, [time_step_1_x1, time_step_1_y1, time_step_1_x2, time_step_1_y2], [time_step_2_x1, time_step_2_y1, time_step_2_x2, time_step_2_y2]]]
    return image_object_loss_dict_for_500
    



def float_list_to_int_list(float_list):
    int_list = []
    for i in float_list:
        int_list.append(int(i))
    return int_list




def norm_loss_gap_to_attention_against_max(loss_list_for_single_img):
    norm = [float(i)/max(loss_list_for_single_img) for i in loss_list_for_single_img]
    return norm



def fake_main():
    if torch.cuda.is_available() == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    img_folder_dir_path = '/home0/zhangc/Dataset/BDD_100K_FLR/imgs_folder_500'
    
    CNN2D_LSTM_model = CNN2D_LSTM.MyEnsemble()
    CNN3D_LSTM_model = CNN3D_LSTM.MyEnsemble()
    CNN3D_model = CNN3D.MyEnsemble()
    
    cnn3d_model_list = [CNN2D_LSTM_model]
    model_category_list = ["CNN2D_LSTM"]
    target_model_serial_num_list = [[0, 0, 0, 0, 0]]

    # cnn3d_model_list = [CNN2D_LSTM_model, CNN3D_LSTM_model, CNN3D_model]
    # model_category_list = ["CNN2D_LSTM", "CNN3D_LSTM", "CNN3D"]
    # target_model_serial_num_list = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    # the model that we want to explain is saved here
    saved_model_for_explanation_path = os.path.join(os.path.split(os.getcwd())[0], "Train_E2EDM","cross_validation_saver") 
    
    
    # we saved the predition results of E2EDMs for origina images
    trained_model_original_action_label_csv_folder_path = "trained_model_for_explanation/trained_model_original_action_label_csv"

    # we saved the predition results of E2EDMs for occluded images, each image is occluded for a single object
    gray_out_img_dict_for_loss_and_info_folder_path = "trained_model_for_explanation/gray_out_img_dict_for_loss_and_info"
    image_object_loss_dict_for_500_model_category_list = []         
    for index in range(len(cnn3d_model_list)):
        image_object_loss_dict_for_500 = {}
        for cross_validation_index in range(5):
            cnn3d_model = cnn3d_model_list[index]
            target_model_serial_num = target_model_serial_num_list[index][cross_validation_index]
            model_category = model_category_list[index]
            print(model_category, cross_validation_index)
            cross_validation_name = "cross_validation-" + str(cross_validation_index)
            target_model_name = str(target_model_serial_num) + ".pt"
            model_path = os.path.join(saved_model_for_explanation_path, model_category, cross_validation_name, target_model_name) 

            cnn3d_model.load_state_dict(torch.load(model_path))
            cnn3d_model = cnn3d_model.to(device)
            target_loss_object_info_dict_path = os.path.join(gray_out_img_dict_for_loss_and_info_folder_path, "gray_out_img_dict_for_loss_and_info_" + model_category + "_" + cross_validation_name + "_epoch-" + str(target_model_serial_num) + ".csv")
            
            original_action_label_csv_path = os.path.join(trained_model_original_action_label_csv_folder_path, "original_action_label_dict_" + model_category + "_" + cross_validation_name + "_epoch" + str(target_model_serial_num) + ".csv") 
            

            if os.path.exists(original_action_label_csv_path) == False:
                original_action_label_dict = save_original_action_label(cnn3d_model,original_action_label_csv_path, cross_validation_index, img_folder_dir_path)

            original_action_label_dict = read_action_label_dict(original_action_label_csv_path)

            if os.path.exists(target_loss_object_info_dict_path) == False:
                make_save_loss_object_info_dict(cnn3d_model, target_loss_object_info_dict_path, original_action_label_dict)

            # Use the predition result for original image and prediction result for occluded image, we calculate the object importance
            image_object_loss_dict_for_500 = make_attention_list_by_loss_object_info_dict(image_object_loss_dict_for_500, target_loss_object_info_dict_path)    
        
        assert len(image_object_loss_dict_for_500) == 500, (image_object_loss_dict_for_500, "image_object_loss_dict_for_500 length is error, not 500")
        image_object_loss_dict_for_500_model_category_list.append(image_object_loss_dict_for_500)
    
    for index in range(len(cnn3d_model_list)):
        model_category = model_category_list[index]
        image_object_loss_dict_for_500 = image_object_loss_dict_for_500_model_category_list[index]
        for key in tqdm(image_object_loss_dict_for_500):
            folder_name = key
                
            loss_list_for_single_img_list = []
            lane_flag_for_single_img_list = []
            position_for_first_img_list = []
            position_for_second_img_list = []
            for each_object in image_object_loss_dict_for_500[key]:
                loss_list_for_single_img_list.append(each_object[0])
                lane_flag_for_single_img_list.append(each_object[1])
                position_for_first_img_list.append(each_object[2])
                position_for_second_img_list.append(each_object[3])

            normed_loss_list = norm_loss_gap_to_attention_against_max(loss_list_for_single_img_list)
            
            
            target_heatmap_folder = os.path.join("gradual_explanation_img(object_simulation)/", model_category)
            os.makedirs(target_heatmap_folder, exist_ok=True)
            single_gradual_img_drawer = gradual_img_for_experiment_drawer(key, target_heatmap_folder, img_folder_dir_path, normed_loss_list, lane_flag_for_single_img_list, position_for_second_img_list, folder_name)
            single_gradual_img_drawer.main_drawer()

    return image_object_loss_dict_for_500_model_category_list    
            
            
    

if __name__== '__main__':
    # Note, this code is not efficient
    fake_main()
    
            
            
    

# %%
