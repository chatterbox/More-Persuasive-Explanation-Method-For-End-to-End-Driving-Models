from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from DataLoading import UdacityDataset_zhang as UD
from DataLoading import UdacityDataset_zhang_for_obejct_explanation as UD2

from model import Convolution3D_LSTM_transfer as CNN3D_LSTM
from model import Convolution2D_LSTM_transfer as CNN2D_LSTM
from model import Convolution3D_only_transfer as CNN3D
import cv2
import numpy as np

import json
ratio_heatmap = 0.3
if torch.cuda.is_available() == True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")


        
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
        
    def draw_heat_map(self):
        
        rescaled_act_img_j = self.attention_map - np.amin(self.attention_map)
        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
        heatmap = np.uint8(255 * rescaled_act_img_j)
        
        rescaled_act_img_j = self.attention_map - np.amin(self.attention_map)
        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        attentioned_original_img = ((1-ratio_heatmap) * self.original_img) + (self.ratio_heatmap * heatmap * 255)
        cv2.imwrite( self.target_heatmap_path , attentioned_original_img)
        
        

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
    target_img_folder_path =  "/home/zhangc/zhangc/dataset/BDD_100K_FLR/explanation/explanation_500__gray_out_img_folder"
    target_csv_folder_for_each_object_path =  "/home/zhangc/zhangc/dataset/BDD_100K_FLR/explanation/explanation_500__gray_out_csv_folder"

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
        
        
        original_action_label = original_action_label_dict[original_img_name]
        loss = diff_calcu(predict_action, original_action_label)
        
        # print(testing_sample['correspond_object_info'][-1][0], testing_sample['image_name'][-1][0])
        
        
        position_info_list_list = []
        # print(testing_sample['correspond_object_info'][-1][0], type(testing_sample['correspond_object_info'][-1][0]))
        # exit()
        if testing_sample['correspond_object_info'][-1][0][0] == "{":
            lane_flag = False
            temp_str_image_1 = testing_sample['correspond_object_info'][0][0].replace('\'',"\"")
            temp_str_image_2 = testing_sample['correspond_object_info'][-1][0].replace('\'',"\"")
            convert_dict_image_1 = json.loads(temp_str_image_1)
            convert_dict_image_2 = json.loads(temp_str_image_2)
            # print(convert_dict)
            position_info_list = [convert_dict_image_1["x1"], convert_dict_image_1["y1"], convert_dict_image_1["x2"], convert_dict_image_1["y2"]]
            position_info_list_list.append(position_info_list)
            position_info_list = [convert_dict_image_2["x1"], convert_dict_image_2["y1"], convert_dict_image_2["x2"], convert_dict_image_2["y2"]]
            position_info_list_list.append(position_info_list)

        else:
            lane_flag = True
            # print(type(testing_sample['correspond_object_info'][-1][0]), "lane")
            # print(testing_sample['correspond_object_info'][-1][0])
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
        # exit()
    

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


    for model_categoy_index in range(len(cnn3d_model_list)):
        model_category = model_category_list[model_categoy_index]
        target_heatmap_folder = os.path.join("Object_based_explanation(heatmap)/", model_category) 
        os.makedirs(target_heatmap_folder, exist_ok=True)
        image_object_loss_dict_for_500 = image_object_loss_dict_for_500_model_category_list[model_categoy_index]
        for key in tqdm(image_object_loss_dict_for_500):
            folder_name = key

            loss_list_for_single_img_list = []
            lane_flag_for_single_img_list = []
            position_for_img_list = []
            position_for_img_list_time_step_1 = []
            position_for_img_list_time_step_2 = []
            for each_object in image_object_loss_dict_for_500[key]:
                loss_list_for_single_img_list.append(each_object[0])
                lane_flag_for_single_img_list.append(each_object[1])
                position_for_img_list_time_step_1.append(each_object[2])
                position_for_img_list_time_step_2.append(each_object[3])
                
                
            position_for_img_list = [position_for_img_list_time_step_1, position_for_img_list_time_step_2]
            
            normed_loss_list = norm_loss_gap_to_attention_against_max(loss_list_for_single_img_list)

            original_image_postfix_list = ["-0000051.jpg", "-0000052.jpg"]
            
            for time_index in range(len(original_image_postfix_list)):
                
                img_name = key + original_image_postfix_list[time_index]
            
                target_heatmap_name = key + "_" + str(time_index+1)  + ".jpg"
                target_heatmap_path = os.path.join(target_heatmap_folder, target_heatmap_name)
                
                img_path = os.path.join(img_folder_dir_path, folder_name, img_name)
                    
                single_heat_map_drawer = heat_map_drawer(img_path, target_heatmap_path, normed_loss_list, lane_flag_for_single_img_list, position_for_img_list[time_index])
                single_heat_map_drawer.make_attention_map()
                single_heat_map_drawer.draw_heat_map()                


if __name__== '__main__':
    # Note, this code is not efficient
    fake_main()      
    