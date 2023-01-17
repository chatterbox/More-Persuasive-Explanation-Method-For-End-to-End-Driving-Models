import os
import json
import cv2
from PIL import Image
import numpy
import csv

import numpy as np
import numpy.matlib
import math

def moveable_object_info_translate(moveable_objects_info, final_image_only_flag = False):
    save_for_future_moveable_position_list = []
    final_image_only_flag_moveable_position_list = []
    for each_moveable_object in moveable_objects_info:

        previous_each_moveable_object = each_moveable_object[1]
        each_moveable_object = each_moveable_object[0]

        position_info_x1 = each_moveable_object[1] - each_moveable_object[3] / 2
        position_info_y1 = each_moveable_object[2] - each_moveable_object[4] / 2
        position_info_x2 = each_moveable_object[1] + each_moveable_object[3] / 2
        position_info_y2 = each_moveable_object[2] + each_moveable_object[4] / 2

        position_info_dict = {'x1': position_info_x1, 'y1': position_info_y1, 'x2': position_info_x2,
                              'y2': position_info_y2}
        final_image_only_flag_moveable_position_list.append(position_info_dict)

        previous_position_info_dict = {'x1': previous_each_moveable_object[0], 'y1': previous_each_moveable_object[1], 'x2': previous_each_moveable_object[2],
                              'y2': previous_each_moveable_object[3]}

        save_for_future_moveable_position_list.append([position_info_dict, previous_position_info_dict])

    if final_image_only_flag == True:
        return final_image_only_flag_moveable_position_list
    else:
        return save_for_future_moveable_position_list

def traffic_light_info_translate(traffic_light_info):
    traffic_light_position_info_list = []
    for each_traffic_light in traffic_light_info:
        position_info_x1 = each_traffic_light[1] - each_traffic_light[3] / 2
        position_info_y1 = each_traffic_light[2] - each_traffic_light[4] / 2
        position_info_x2 = each_traffic_light[1] + each_traffic_light[3] / 2
        position_info_y2 = each_traffic_light[2] + each_traffic_light[4] / 2

        position_info_dict = {'x1': position_info_x1, 'y1': position_info_y1, 'x2': position_info_x2, 'y2': position_info_y2}
        traffic_light_position_info_list.append(position_info_dict)

    return traffic_light_position_info_list

def superimpose_picture(original_img_j, attention_img):
    # print(attention_img.shape)
    rescaled_act_img_j = attention_img - np.amin(attention_img)
    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    overlayed_original_img_j = 0.7 * original_img_j + 0.3 * heatmap * 255


    return overlayed_original_img_j


def make_attention_with_each_box_info(object_alpha, object_box_info, attention_map, time_counter = None):

    if isinstance(object_box_info, dict) == False and len(object_box_info) > 1 and time_counter != None :
        # print(object_box_info, len(object_box_info))
        # print("its ok ")
        object_box_info = object_box_info[time_counter - 1]

    horizontal_line_start = int(object_box_info["x1"])
    horizontal_line_end = int(object_box_info["x2"])

    longitudinal_line_start = int(object_box_info["y1"])
    longitudinal_line_end = int(object_box_info["y2"])

    for horizontal_line in range(horizontal_line_start, horizontal_line_end):
        for longitudinal_line in range(longitudinal_line_start, longitudinal_line_end):
            attention_map[longitudinal_line, horizontal_line] = object_alpha

    return attention_map

def float_list_to_int_list(float_list):
    int_list = []
    for i in float_list:
        int_list.append(int(i))
    return int_list

def make_attention_with_each_start_end_point(object_alpha, object_box_info, attention_map):

    (x1, y1) = object_box_info[1], object_box_info[2]
    (x2, y2) = object_box_info[3], object_box_info[4]
    gap_length = 10
    line_length = math.sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) )
    gap_number = int( line_length / gap_length )
    if gap_number % 2 == 0:
        gap_number = gap_number + 1

    # gap_number = 5 # 至少5个, 只能为奇数 2n + 1
    x_gap_length = (x2 - x1) / gap_number
    y_gap_length = (y2 - y1) / gap_number
    each_line_list = []
    for i in range(gap_number):
        if i % 2 == 0:
            each_line = []
            # print(i)
            x_temp = x1 + i * x_gap_length
            y_temp = y1 + i * y_gap_length

            point = []
            point.append(x_temp)
            point.append(y_temp)
            point = float_list_to_int_list(point)
            each_line.append(point)

            point = []
            point.append(x_temp + x_gap_length)
            point.append(y_temp + y_gap_length)
            point = float_list_to_int_list(point)
            each_line.append(point)

            each_line_list.append(each_line)
    # print(each_line_list)
    # exit()
    for each_point in each_line_list:
        # print(each_point)
        center_point_x = int((each_point[0][0] + each_point[1][0]) / 2)
        center_point_y = int((each_point[0][1] + each_point[1][1]) / 2)

        box_length = abs(each_point[0][0] - each_point[1][0])
        box_height = abs(each_point[0][1] - each_point[1][1])

        x1 = center_point_x - int(box_length / 2)
        x2 = center_point_x + int(box_length / 2)
        y1 = center_point_y - int(box_height / 2)
        y2 = center_point_y + int(box_height / 2)

        object_box_info = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        # print(object_box_info)
        # exit()
        attention_map = make_attention_with_each_box_info(object_alpha, object_box_info, attention_map)
    return attention_map


def make_attention_with_box_info(objects_alpha, objects_box_info, attention_map, time_counter = None):

    if len(objects_alpha) != len(objects_box_info):
        print("len(objects_alpha) != len(objects_box_info)")
        exit()
    for serial_num in range(len(objects_alpha)):
        if time_counter != None:

            attention_map = make_attention_with_each_box_info(objects_alpha[serial_num], objects_box_info[serial_num][time_counter - 1], attention_map)
        else:
            attention_map = make_attention_with_each_box_info(objects_alpha[serial_num], objects_box_info[serial_num],
                                                              attention_map)
    return attention_map


def line_pixel_coordinate_calculator(image, object_box_info):
    blank_img = np.zeros(image.shape)
    (x1, y1) = object_box_info[1], object_box_info[2]
    (x2, y2) = object_box_info[3], object_box_info[4]
    color = (255)
    thickness = 30
    lineType = 4

    cv2.line(blank_img, (x1, y1), (x2, y2), color,
             thickness, lineType)
    line_pixel_list = []
    for i in range(blank_img.shape[0]):
        for j in range(blank_img.shape[1]):
            if blank_img[i][j] == 255:
                line_pixel_list.append((i,j))
    return line_pixel_list
def make_attention_with_each_start_end_point_upgrade(object_alpha, object_box_info, attention_map):
    line_pixel_list = line_pixel_coordinate_calculator(attention_map, object_box_info)
    for i in line_pixel_list:
        attention_map[i[0],i[1]] = object_alpha
    return attention_map
def make_attention_with_start_end_point_upgrade(objects_alpha, objects_box_info, attention_map):
    if len(objects_alpha) != len(objects_box_info):
        print("len(objects_alpha start_end_point) != len(objects_box_info start_end_point)")
        exit()
    for serial_num in range(len(objects_alpha)):
        attention_map = make_attention_with_each_start_end_point_upgrade(objects_alpha[serial_num], objects_box_info[serial_num], attention_map)
    # exit()
    return attention_map

def acsending_order_for_alpha_and_info_list(alpha, info):

    index_list = np.argsort(alpha)

    alpha = np.array(alpha)
    info = np.array(info)
    sorted_alpha = alpha[index_list]
    sorted_info = info[index_list]
    return sorted_alpha, sorted_info

def decsending_order_for_alpha_and_info_list_for_3(alpha, info, flag_list):

    index_list = np.argsort(alpha)[::-1]

    alpha = np.array(alpha)
    info = np.array(info)
    flag_list = np.array(flag_list)

    sorted_alpha = alpha[index_list]
    sorted_info = info[index_list]
    sorted_flag_list = flag_list[index_list]
    return sorted_alpha, sorted_info, sorted_flag_list

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
        if len(grid_match_list) < self.top_k:
            grid_coordinate_list = self.find_top_k_grid()
            for grid_coordinate in grid_coordinate_list:
                grid_coordinate_tuple = (grid_coordinate[0], grid_coordinate[1])
                if grid_coordinate_tuple not in grid_match_list:
                    if len(grid_match_list) >= 4:
                        break
                    grid_match_list.append(grid_coordinate_tuple)

        assert len(grid_match_list) == self.top_k, ("len(grid_match_list) != self.top_k", self.target_img_name, grid_match_list)

        return grid_match_list




    def object_center_point_calculator(self):
        object_center_point_tuple_list = []
        for i in range(len(self.objects_info)):
            object_box_info = self.objects_info[i]
            lane_flag = self.lane_flag_list[i]

            if lane_flag == True:
                (x1, y1) = object_box_info[1], object_box_info[2]
                (x2, y2) = object_box_info[3], object_box_info[4]
                if y1 < y2:
                    nearest_point = (x2, y2)
                else:
                    nearest_point = (x1, y1)
                object_center_point_tuple_list.append( nearest_point )
            else:
                x1 = int(object_box_info["x1"])
                x2 = int(object_box_info["x2"])

                y1 = int(object_box_info["y1"])
                y2 = int(object_box_info["y2"])
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

def gradual_image_based_on_explanation(all_object_info_and_attention, target_img_name, simulation_gradual_image_folder):
    original_img_folder_path = "./original_image"

    original_img_postfix = ".jpg"
    original_img_num_list = [1, 2]
    original_img_path_list = []
    for original_img_num in original_img_num_list:
        target_original_img_name = target_img_name + "_" + str(original_img_num) + \
                                   original_img_postfix
        target_original_img_path = os.path.join(original_img_folder_path, target_original_img_name)
        assert os.path.exists(target_original_img_path), target_original_img_path
        original_img_path_list.append(target_original_img_path)


    moveable_objects_info = all_object_info_and_attention[0][0]
    moveable_objects_alpha = all_object_info_and_attention[1][0]

    lane_info = all_object_info_and_attention[0][1]
    lane_alpha = all_object_info_and_attention[1][1]

    traffic_light_info = all_object_info_and_attention[0][2]
    traffic_light_alpha = all_object_info_and_attention[1][2]

    moveable_objects_box_info = moveable_object_info_translate(moveable_objects_info, True)
    traffic_light_box_info = traffic_light_info_translate(traffic_light_info)

    objects_alpha = moveable_objects_alpha + lane_alpha + traffic_light_alpha
    objects_info = moveable_objects_box_info + lane_info + traffic_light_box_info
    lane_flag_list = []
    for i in moveable_objects_alpha:
        lane_flag_list.append(False)
    for i in lane_alpha:
        lane_flag_list.append(True)
    for i in traffic_light_alpha:
        lane_flag_list.append(False)

    objects_alpha, objects_info, lane_flag_list = decsending_order_for_alpha_and_info_list_for_3(objects_alpha,objects_info,lane_flag_list)

    heatmap = np.zeros((720, 1280))
    heatmap = make_attention_with_start_end_point_upgrade(lane_alpha, lane_info, heatmap)
    heatmap = make_attention_with_box_info(moveable_objects_alpha, moveable_objects_box_info, heatmap)
    heatmap = make_attention_with_box_info(traffic_light_alpha, traffic_light_box_info, heatmap)


    grid_object = puzzle_simulation(heatmap, objects_alpha, objects_info, lane_flag_list, target_img_name)
    grid_box_size = grid_object.get_grid_box_size()
    grid_coordinate_list = grid_object.find_match()
    # print(grid_coordinate_list)
    # exit()
    gradual_counter = 0
    for serial_num in range(len(grid_coordinate_list)):
        filter_map = filter_map_translator(grid_coordinate_list, grid_box_size, serial_num)
        time_counter = 0
        gradual_counter = gradual_counter + 1
        for original_img_path in original_img_path_list:
            time_counter = time_counter + 1
            original_image = cv2.imread(original_img_path, 3)
            filtered_original_image = filter_original_img(original_image, filter_map)
            target_img_name_for_save = target_img_name + "_" + str(gradual_counter) + "_" + str(
                time_counter) + ".jpg"
            target_img_path = os.path.join(simulation_gradual_image_folder, target_img_name_for_save)
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



def make_attention_mask_interface(all_object_info_and_attention, target_img_name, target_img_folder_path, heatmap_mask_folder_path):
    original_img_folder_path = "./original_image"

    original_img_postfix = ".jpg"
    original_img_num_list = [1, 2]
    original_img_path_list = []
    for original_img_num in original_img_num_list:
        target_original_img_name = target_img_name + "_" + str(original_img_num) + \
                                   original_img_postfix
        target_original_img_path = os.path.join(original_img_folder_path, target_original_img_name)
        assert os.path.exists(target_original_img_path), target_original_img_path
        original_img_path_list.append(target_original_img_path)


    moveable_objects_info = all_object_info_and_attention[0][0]
    moveable_objects_alpha = all_object_info_and_attention[1][0]

    lane_info = all_object_info_and_attention[0][1]
    lane_alpha = all_object_info_and_attention[1][1]

    traffic_light_info = all_object_info_and_attention[0][2]
    traffic_light_alpha = all_object_info_and_attention[1][2]


    moveable_objects_box_info = moveable_object_info_translate(moveable_objects_info)
    traffic_light_box_info = traffic_light_info_translate(traffic_light_info)

    lane_alpha, lane_info = acsending_order_for_alpha_and_info_list(lane_alpha, lane_info)
    moveable_objects_alpha, moveable_objects_box_info = acsending_order_for_alpha_and_info_list(moveable_objects_alpha, moveable_objects_box_info)
    traffic_light_alpha, traffic_light_box_info = acsending_order_for_alpha_and_info_list(traffic_light_alpha, traffic_light_box_info)

    time_counter = 3
    image_name_counter = 0
    for original_img_path in original_img_path_list:
        attention_map = np.zeros((720, 1280))
        time_counter = time_counter - 1
        image_name_counter = image_name_counter + 1
        original_image = cv2.imread(original_img_path, 3)
        attention_map = make_attention_with_start_end_point_upgrade(lane_alpha, lane_info, attention_map)
        attention_map = make_attention_with_box_info(moveable_objects_alpha, moveable_objects_box_info, attention_map, time_counter)
        attention_map = make_attention_with_box_info(traffic_light_alpha, traffic_light_box_info, attention_map)

        attentioned_img_name = target_img_name + "_" + str(image_name_counter) + ".jpg"


        rescaled_act_img_j = attention_map - np.amin(attention_map)
        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
        heatmap = np.uint8(255 * rescaled_act_img_j)
        attention_map = np.float32(heatmap)

        # heatmap_mask_path = os.path.join(heatmap_mask_folder_path, attentioned_img_name)
        # cv2.imwrite(heatmap_mask_path, attention_map)
        attentioned_original_img = superimpose_picture(original_image, attention_map)
        attentioned_img_path = os.path.join(target_img_folder_path, attentioned_img_name)
        cv2.imwrite(attentioned_img_path, attentioned_original_img)

