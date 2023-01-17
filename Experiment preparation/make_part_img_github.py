import csv
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
def serial_threshold_calculator(image, top_percent):
    flattened_img = image.flatten()
    sorted_flattened_img = sorted(flattened_img, reverse=True)
    flattened_img_length = len(sorted_flattened_img)
    threshold_index = int(top_percent * flattened_img_length)
    if top_percent == 1:
        threshold_index = threshold_index - 1
    threshold = sorted_flattened_img[threshold_index]
    # print("pixel_number", int(top_percent * flattened_img_length)/921600, "threshold", threshold, "top_percent", top_percent)
    return threshold

def top_percent_filter(image, top_percent):

    threshold_value = serial_threshold_calculator(image, top_percent)
    filter_map = np.zeros(image.shape)

    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            if image[height][width] >= threshold_value:
                filter_map[height][width] = 1
    return filter_map



def filter_original_img(original_image, filter_map):
    height, width, channels = original_image.shape
    filtered_original_image = original_image.copy()
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                if filter_map[i][j] == 0:
                    filtered_original_image[:, :, channel][i][j] = 128
    return filtered_original_image

def img_name_extracter(csv_path):
    img_name_list = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            img_name  = line[0]
            img_name_list.append(img_name)
        next(reader, None)  # jump to the next line
    return  img_name_list

class puzzle_simulation():
    def __init__(self, heatmap_image, puzzle_top_num = 4):
        self.top_k = puzzle_top_num
        self.heatmap = heatmap_image
        self.puzzle_top_num = puzzle_top_num
        self.image_shape = (1280, 720)
        self.grid_shape_divide_num = (8, 8)

        self.grid_box_size = (
            int(self.image_shape[0] / self.grid_shape_divide_num[0]), int(self.image_shape[1] / self.grid_shape_divide_num[1]))

        self.grid_list = self.grid_maker()

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
    def find_top_k_grid(self):
        grid_importance_list = []
        for each_grid in self.grid_list:
            grid_importance_list.append( self.grid_importance_calculator(each_grid) )
        sorted_grid_importance_list, sorted_grid_list = decsending_order_for_alpha_and_info_list(grid_importance_list, self.grid_list)
        return sorted_grid_list[:self.top_k]
    def grid_importance_calculator(self, grid_coordinate):
        importance_counter = 0
        counter_flag = 0

        for i in range(grid_coordinate[0], grid_coordinate[0] + self.grid_box_size[0]):
            for j in range(grid_coordinate[1], grid_coordinate[1] + self.grid_box_size[1]):
                counter_flag = counter_flag + 1
                importance_counter = importance_counter + self.heatmap[j][i]

        return importance_counter / counter_flag
def decsending_order_for_alpha_and_info_list(alpha, info):
    index_list = np.argsort(alpha)[::-1]
    alpha = np.array(alpha)
    info = np.array(info)
    sorted_alpha = alpha[index_list]
    sorted_info = info[index_list]
    return sorted_alpha, sorted_info
def filter_map_translator(grid_coordinate_list, grid_box_size, top_range):
    filter_map = np.zeros((720, 1280))
    for serial_num in range(top_range + 1):
        grid_coordinate = grid_coordinate_list[serial_num]
        for i in range(grid_coordinate[0], grid_coordinate[0] + grid_box_size[0]):
            for j in range(grid_coordinate[1], grid_coordinate[1] + grid_box_size[1]):
                filter_map[j][i] = 1

    return filter_map

if __name__ == '__main__':
    DP_category_list = ["CNN2D_LSTM", "CNN3D_LSTM", "CNN3D"]
    img_name_list = img_name_extracter("zhang_pesudo_img.csv")

    original_img_folder_path = "./original_image"
    heatmap_img_folder_path = "./heatmap_mask"
    target_folder_path = "./unmixed_simulation/DP-pixel"
    heatmap_img_postfix = "_2.jpg"
    target_folder_path_post_name = "DP-pixel-"
    original_img_postfix = ".jpg"
    original_img_num_list = [1, 2]

    for DP_category in DP_category_list:
        subtarget_folder_path = target_folder_path + "-" + DP_category
        subheatmap_img_folder_path = os.path.join(heatmap_img_folder_path, DP_category)
        os.makedirs(subtarget_folder_path, exist_ok = True)
        for target_img_name in tqdm(img_name_list):
            gradual_counter = 0
            original_img_path_list = []
            for original_img_num in original_img_num_list:
                target_original_img_name = target_img_name + "_" + str(original_img_num) + \
                                           original_img_postfix
                original_img_path_list.append(
                    os.path.join(original_img_folder_path, target_original_img_name))
            heatmap_img_path = os.path.join(subheatmap_img_folder_path, target_img_name + heatmap_img_postfix)
            assert os.path.exists(heatmap_img_path), (heatmap_img_path, target_img_name, heatmap_img_postfix)
            heatmap_image = cv2.imread(heatmap_img_path, 2)

            grid_object = puzzle_simulation(heatmap_image)
            grid_box_size = grid_object.get_grid_box_size()
            grid_coordinate_list = grid_object.find_top_k_grid()

            for serial_num in range(len(grid_coordinate_list)):
                filter_map = filter_map_translator(grid_coordinate_list, grid_box_size, serial_num)
                time_counter = 0
                gradual_counter = gradual_counter + 1
                for original_img_path in original_img_path_list:
                    assert os.path.exists(original_img_path)
                    time_counter = time_counter + 1
                    original_image = cv2.imread(original_img_path, 3)
                    filtered_original_image = filter_original_img(original_image, filter_map)
                    target_img_name_for_save = target_img_name + "_" + str(gradual_counter) + "_" + str(
                        time_counter) + ".jpg"
                    target_img_path = os.path.join(subtarget_folder_path, target_img_name_for_save)
                    cv2.imwrite(target_img_path, filtered_original_image)







