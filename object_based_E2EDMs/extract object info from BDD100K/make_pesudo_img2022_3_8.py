import os
import json
import cv2
from PIL import Image
import numpy
import csv
import math
from tqdm import tqdm
image_size = (1280, 720)


def read_environment_name_json_file(video_environment_name_folder_path):

    with open(video_environment_name_folder_path, 'r') as jsonfile:
        json_string = json.load(jsonfile)

    name_list = []
    environment_name_list = []
    for i in json_string:
        name_list.append(i['name'])
        environment_name_list.append(i['attributes']['scene'])
        if i["timestamp"] != 10000:
            print("something wrong", i['name'])
            exit()

    return json_string, environment_name_list

def filter_for_required_environment(tracking_file_name_list, name_list, environment_name_list):
    required_name_list = []
    for tracking_file_name in tracking_file_name_list:
        index_num = name_list.index(tracking_file_name + ".jpg")
        if environment_name_list[index_num] == "residential" or environment_name_list[index_num] == "city street":
            required_name_list.append(tracking_file_name)


    return required_name_list

def get_position_info_for_moveable_objects(tracking_info_file_path, serial_num_list):

    with open(tracking_info_file_path, 'r') as jsonfile:
        json_string = json.load(jsonfile)

    category_list = []
    moveable_id_list_list = []
    occluded_boole_list = []
    truncated_boole_list = []
    for serial_num in serial_num_list:
        pseuedo_info = json_string[serial_num]

        moveable_id_list = []
        for moveable_object in pseuedo_info['labels']:
            moveable_id_list.append(moveable_object["id"])

        moveable_id_list_list.append(moveable_id_list)


    interactive_union = set(moveable_id_list_list[0])
    for data in moveable_id_list_list[1:]:
        interactive_union &= set(data)

    pseuedo_info = json_string[serial_num_list[0]]
    for each_valid_object_id in interactive_union:
        for moveable_object in pseuedo_info['labels']:
            if moveable_object["id"] == each_valid_object_id:
                category_list.append(moveable_object["category"])
                occluded_boole_list.append(moveable_object["attributes"]["occluded"])
                truncated_boole_list.append(moveable_object["attributes"]["truncated"])

    moveable_position_list_list = []

    for serial_num in serial_num_list:
        pseuedo_info = json_string[serial_num]
        moveable_position_list = []
        for each_valid_object_id in interactive_union:
            for moveable_object in pseuedo_info['labels']:
                if moveable_object["id"] == each_valid_object_id:
                    moveable_position_list.append(moveable_object["box2d"])

        moveable_position_list_list.append(moveable_position_list)

    return category_list, moveable_position_list_list, occluded_boole_list


def draw_pseudo_img_boundingbox(position_list):

    center_point_x_list = []
    center_point_y_list = []
    x_length_list = []
    y_length_list = []

    for serial_num in range(len(position_list)):
        x1 = int(position_list[serial_num]["x1"])
        x2 = int(position_list[serial_num]["x2"])
        y1 = int(position_list[serial_num]["y1"])
        y2 = int(position_list[serial_num]["y2"])

        measure_of_rectangle_area, center_point_x, center_point_y, x_length, y_length = rectangle_area_and_center_point(x1,x2,y1,y2)

        center_point_x_list.append(center_point_x)
        center_point_y_list.append(center_point_y)
        x_length_list.append(x_length)
        y_length_list.append(y_length)


    return center_point_x_list, center_point_y_list, x_length_list, y_length_list


def rectangle_area_and_center_point(x1, x2, y1, y2):
    return ( (x2-x1) * (y2 - y1) ), (x2-x1)/2 + x1 , (y2 - y1) / 2 + y1, x2-x1, y2 - y1

def move_direction_calcu(center_point_x, center_point_y, x_length, y_length,threhold_value_for_lateral,threhold_value_for_vertical):

    lateral_list = [] # 0 是无， 1是向右边，-1是向左
    vertical_list = [] # 0 是无， 1是向前，-1是向后

    for i in range(len(center_point_x) - 1):
        x_diff = center_point_x[i + 1] - center_point_x[i]
        if abs(x_diff) >= threhold_value_for_lateral:
            if x_diff > 0:
                lateral_list.append(1)
            else:
                lateral_list.append(-1)
        else:
            lateral_list.append(0)
        # y_diff = center_point_y[i + 1] - center_point_y[i]
        area_diff = ( x_length[i + 1] * y_length[i + 1] ) - ( y_length[i] * x_length[i] )

        if abs(area_diff) >= threhold_value_for_vertical:
            if area_diff < 0:
                vertical_list.append(1)
            else:
                vertical_list.append(-1)
        else:
            vertical_list.append(0)

    final_lateral_decision = 0
    final_vertical_decision = 0
    if lateral_list.count(1) > lateral_list.count(0) and lateral_list.count(1) > lateral_list.count(-1):
        final_lateral_decision = 1
    elif lateral_list.count(-1) > lateral_list.count(0) and lateral_list.count(-1) > lateral_list.count(0):
        final_lateral_decision = -1

    if vertical_list.count(1) > vertical_list.count(0) and vertical_list.count(1) > vertical_list.count(-1):
        final_vertical_decision = 1
    elif vertical_list.count(-1) > vertical_list.count(0) and vertical_list.count(-1) > vertical_list.count(0):
        final_vertical_decision = -1
    return final_lateral_decision, final_vertical_decision

def move_speed_calcu(center_point_x, center_point_y, x_length, y_length):

    constant_area_to_y_axis = 5
    area_diff = float( (x_length[-1] * y_length[-1]) / (y_length[0] * x_length[0]) )
    x_diff = center_point_x[-1] - center_point_x[0]

    y_diff = (center_point_y[-1] - center_point_y[0]) + constant_area_to_y_axis * area_diff

    return x_diff, y_diff


def get_all_object_info(json_string, wanted_test_name):

    category_list = []
    position_info_list = []
    lane_line_style = ["solid", "dashed", "orange solid", "orange dashed"]

    # all object: ['person', 'train', 'traffic sign', 'car', 'bike', 'drivable area', 'traffic light', 'motor', 'rider', 'lane', 'truck', 'bus']
    avoid_list = ['person', 'train', 'traffic sign', 'car', 'bike', 'drivable area', 'motor', 'rider', 'truck', 'bus']
    for each_img in json_string:
        if each_img["name"] == wanted_test_name + ".jpg":
            object_info_list = each_img["labels"]
            flag_lane = False
            for each_object in object_info_list:

                if each_object["category"] in avoid_list:
                    continue
                if each_object["category"] == "lane":

                    if each_object["attributes"]["laneType"] == "crosswalk":
                        category_list.append(each_object["category"] + " " + lane_line_style[1])
                        position_info_list.append(each_object["poly2d"][0]["vertices"])
                        continue
                    if each_object["attributes"]["laneType"] == "road curb":
                        flag_lane = True
                        category_list.append(each_object["category"] + " " + lane_line_style[0])
                        position_info_list.append(each_object["poly2d"][0]["vertices"])
                        continue
                    if each_object["attributes"]["laneStyle"] == "solid":
                        if each_object["attributes"]["laneDirection"] == "vertical":
                            continue
                        category_list.append(each_object["category"] + " " + lane_line_style[2])
                        position_info_list.append(each_object["poly2d"][0]["vertices"])
                        continue
                    if each_object["attributes"]["laneStyle"] == "dashed":
                        if each_object["attributes"]["laneDirection"] == "vertical":
                            continue
                        category_list.append(each_object["category"] + " " + lane_line_style[3])
                        position_info_list.append(each_object["poly2d"][0]["vertices"])
                        continue
                if each_object["category"] == "traffic light":

                    if each_object["attributes"]["trafficLightColor"] == "none" or each_object["attributes"]["influence"] == False:
                        pass
                    else:
                        category_list.append(each_object["category"] + each_object["attributes"]["trafficLightColor"] + " " + each_object["attributes"]["shape"])
                        position_info_list.append(each_object["box2d"])
                else:
                    category_list.append(each_object["category"])
                    position_info_list.append(each_object["box2d"])

            break # wanted picture has been found
    return category_list, position_info_list

def float_list_to_int_list(float_list):
    int_list = []
    for i in float_list:
        int_list.append(int(i))
    return int_list


def calculate_motion_direction_by_tracking_speed_value(moveable_position_list_list, target_img_path, category_list_move, serial_num_list):
    center_point_x_object_time_list = []
    center_point_y_object_time_list = []
    x_length_object_time_list = []
    y_length_object_time_list = []

    center_point_x_list_list = []
    center_point_y_list_list = []
    x_length_list_list = []
    y_length_list_list = []
    save_img_name_list = []


    save_for_future_moveable_position_list = moveable_position_list_list[-1]
    save_for_future_category_list_move = category_list_move
    save_for_future_serial_num = serial_num_list[-1] + 1


    for time_serial_number in range(len(moveable_position_list_list)):

        center_point_x_list, center_point_y_list, x_length_list, y_length_list = draw_pseudo_img_boundingbox(moveable_position_list_list[time_serial_number])
        save_img_name = str(serial_num_list[time_serial_number] + 1)

        center_point_x_list_list.append(center_point_x_list)
        center_point_y_list_list.append(center_point_y_list)
        x_length_list_list.append(x_length_list)
        y_length_list_list.append(y_length_list)
        save_img_name_list.append(save_img_name)



    for object_serial_num in range(len(center_point_x_list_list[0])):
        center_point_x_time_list = []
        center_point_y_time_list = []
        x_length_time_list = []
        y_length_time_list = []
        for time_serial_number in range(len(center_point_x_list_list)):
            center_point_x_time_list.append(center_point_x_list_list[time_serial_number][object_serial_num])
            center_point_y_time_list.append(center_point_y_list_list[time_serial_number][object_serial_num])
            x_length_time_list.append(x_length_list_list[time_serial_number][object_serial_num])
            y_length_time_list.append(y_length_list_list[time_serial_number][object_serial_num])

        center_point_x_object_time_list.append(center_point_x_time_list)
        center_point_y_object_time_list.append(center_point_y_time_list)
        x_length_object_time_list.append(x_length_time_list)
        y_length_object_time_list.append(y_length_time_list)


    final_lateral_decision_list = []
    final_vertical_decision_list = []

    last_img_center_point_x_list = []
    last_img_center_point_y_list = []
    last_x_length_list = []
    last_y_length_list = []

    for each_object in range(len(center_point_x_object_time_list)):
        final_lateral_decision, final_vertical_decision = move_speed_calcu(
            center_point_x_object_time_list[each_object], center_point_y_object_time_list[each_object],
            x_length_object_time_list[each_object],
            y_length_object_time_list[each_object])

        final_lateral_decision_list.append(final_lateral_decision)
        final_vertical_decision_list.append(final_vertical_decision)
        last_img_center_point_x_list.append(center_point_x_object_time_list[each_object][-1])
        last_img_center_point_y_list.append(center_point_y_object_time_list[each_object][-1])
        last_x_length_list.append(x_length_object_time_list[each_object][-1])
        last_y_length_list.append(y_length_object_time_list[each_object][-1])

    return last_img_center_point_x_list, last_img_center_point_y_list, target_img_path, final_lateral_decision_list, final_vertical_decision_list, save_for_future_moveable_position_list, save_for_future_category_list_move, save_for_future_serial_num



def create_csv(pesudo_csv_folder_name, wanted_test_name, new_unmoveable_object_info_list, new_moveable_object_info_list, self_speed):
    path = os.path.join(pesudo_csv_folder_name, wanted_test_name + ".csv")
    # list_save = []
    with open(path, 'a', newline="", encoding = "utf-8") as f:
        writer = csv.writer(f, delimiter = ",")
        # csv_head = wanted_test_name
        # list_save.append(csv_head)
        # writer.writerow(list_save)
        writer.writerow(self_speed)
        for unmoveable_object_info in new_unmoveable_object_info_list:
            writer.writerow(unmoveable_object_info)
        for moveable_object_info in new_moveable_object_info_list:

            writer.writerow(moveable_object_info)


def get_middle_point_list_by_two_point(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    distance_between_two_point = math.sqrt( (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) )
    gap_length = 10
    gap_number = int(distance_between_two_point // gap_length)

    if gap_number % 2 == 0:
        gap_number = gap_number + 1

    gap_number = 5 # 至少5个, 只能为奇数 2n + 1
    x_gap_length = (x2 - x1) / gap_number
    y_gap_length = (y2 - y1) / gap_number
    each_point_list = []
    for i in range(gap_number):
        if i % 2 == 0:

            x_temp = x1 + i * x_gap_length
            y_temp = y1 + i * y_gap_length

            point = []
            point.append(x_temp)
            point.append(y_temp)
            point = float_list_to_int_list(point)
            each_point_list.append(point)

            point = []
            point.append(x_temp + x_gap_length)
            point.append(y_temp + y_gap_length)
            point = float_list_to_int_list(point)
            each_point_list.append(point)

    return each_point_list

def make_grid_info_by_single_point(single_point):

    img_width = 1280
    img_height = 720
    ratio_width = 16
    ratio_height = 9
    grid_size = int(img_width / ratio_width)
    single_point_x = single_point[0]
    single_point_y = single_point[1]
    width_num = float(single_point_x / grid_size)
    height_num = float(single_point_y / grid_size)

    if width_num % 1 != 0:  # 这个的目的是为了解决当一个像素点出现在网格中间这条线上的情况下我们应该算他左边还是算他右边，我们现在给它规范了那就是默认算左边。同样的如果一个像素点出现在横向的网格中间的线上我们就默认算它上面, 除非左边没有格子了
        grid_box_serial_num_x = int(width_num + 1)
    else:
        grid_box_serial_num_x = int(width_num)
        if width_num == 0:
            grid_box_serial_num_x = int(width_num + 1)
    if height_num % 1 != 0:  # 同样的如果一个像素点出现在横向的网格中间的线上我们就默认算它上面， 除非上面没有格子了
        grid_box_serial_num_y = int(height_num + 1)
    else:
        grid_box_serial_num_y = int(height_num)
        if height_num == 0:
            grid_box_serial_num_y = int(height_num + 1)
    grid_box_serial_num = (grid_box_serial_num_y - 1) * ratio_width + grid_box_serial_num_x
    return grid_box_serial_num

def make_grid_info_by_discret_points(point_list):
    grid_box_serial_num_list = []
    for single_point in point_list:
        grid_box_serial_num = make_grid_info_by_single_point(single_point)
        grid_box_serial_num_list.append(grid_box_serial_num)

    return grid_box_serial_num_list

def translate_point_line_lane_to_grid_info(position_info):

    each_point_list = []
    each_line_list = []

    grid_info_list_whole_line = []
    for each_point in position_info:
        each_point = float_list_to_int_list(each_point)
        each_point_list.append(each_point)

    for each_point_serial_num in range(len(each_point_list) - 1) :
        each_line = []
        each_line.append(each_point_list[each_point_serial_num])
        each_line.append(each_point_list[each_point_serial_num + 1])
        each_line_list.append(each_line)
    for each_line in each_line_list:
        start_point = (each_line[0][0], each_line[0][1])
        end_point = (each_line[1][0], each_line[1][1])


        point_list = get_middle_point_list_by_two_point(start_point, end_point)

        grid_info_list = make_grid_info_by_discret_points(point_list)

        grid_info_list = list(set(grid_info_list))

        grid_info_list_whole_line = grid_info_list_whole_line + grid_info_list

    grid_info_list_whole_line = list(set(grid_info_list_whole_line))
    return grid_info_list_whole_line
    # exit()

def translate_point_line_lane_to_start_end_point_info(position_info):

    each_point_list = []
    each_line_list = []

    for each_point in position_info:
        each_point = float_list_to_int_list(each_point)
        each_point_list.append(each_point)

    for each_point_serial_num in range(len(each_point_list) - 1) :
        each_line = [each_point_list[each_point_serial_num][0], each_point_list[each_point_serial_num][1], each_point_list[each_point_serial_num + 1][0], each_point_list[each_point_serial_num + 1][1]]
        each_line_list.append(each_line)

    return each_line_list

def csv_dataset_maker_for_unmoveable_object(category_list, position_info_list):
    new_object_info_list = []

    for serial_num in range(len(position_info_list)):
        i = position_info_list[serial_num]
        if isinstance(i,dict):
            center_point_x = int( (i["x2"] + i["x1"]) / 2 )
            center_point_y = int((i["y2"] + i["y1"]) / 2)
            box_length = int( i["x2"] - i["x1"] )
            box_heigth = int( i["y2"] - i["y1"] )
            object_info = [category_list[serial_num], center_point_x, center_point_y, box_length, box_heigth]
            new_object_info_list.append(object_info)
        else:
            # grid_info_list = translate_point_line_lane_to_grid_info(i)
            start_end_point_info = translate_point_line_lane_to_start_end_point_info(i)

            # object_info = [category_list[serial_num], grid_info_list]
            object_info = [category_list[serial_num], start_end_point_info]
            new_object_info_list.append(object_info)
    return new_object_info_list

def csv_dataset_maker_for_moveable_object(category_list_move, moveable_position_list_list, final_lateral_decision_list, final_vertical_decision_list):
    # new_moveable_position_info_list = []
    new_object_info_list = []
    moveable_position_list = moveable_position_list_list[-1]
    previous_moveable_position_list = moveable_position_list_list[0]

    for serial_num in range(len(moveable_position_list)):
        i = moveable_position_list[serial_num]
        if isinstance(i, dict):
            center_point_x = int((i["x2"] + i["x1"]) / 2)
            center_point_y = int((i["y2"] + i["y1"]) / 2)
            box_length = int(i["x2"] - i["x1"])
            box_heigth = int(i["y2"] - i["y1"])
            final_lateral_decision = int(final_lateral_decision_list[serial_num])
            final_vertical_decision = int(final_vertical_decision_list[serial_num])

            previous_x1 = int(previous_moveable_position_list[serial_num]["x1"])
            previous_y1 = int(previous_moveable_position_list[serial_num]["y1"])
            previous_x2 = int(previous_moveable_position_list[serial_num]["x2"])
            previous_y2 = int(previous_moveable_position_list[serial_num]["y2"])

            object_info = [category_list_move[serial_num], center_point_x, center_point_y, box_length, box_heigth, final_lateral_decision, final_vertical_decision, previous_x1, previous_y1, previous_x2, previous_y2]
            # object_info = [category_list_move[serial_num], center_point_x, center_point_y, box_length, box_heigth, final_lateral_decision, final_vertical_decision]
            new_object_info_list.append(object_info)


    return new_object_info_list


def make_a_pesudo_img(end_frame_serial_num, wanted_test_name, target_img_path, calculation_time_interval, tracking_info_folder_path, json_string, pesudo_csv_folder_name):
    serial_num_list = []
    for i in range(calculation_time_interval):
        serial_num_list.append(end_frame_serial_num - calculation_time_interval + i)

    tracking_info_file_path = os.path.join(tracking_info_folder_path,
                                           wanted_test_name + ".json")
    category_list_move, moveable_position_list_list, occluded_boole_list = get_position_info_for_moveable_objects(
        tracking_info_file_path, serial_num_list)


    category_list, position_info_list = get_all_object_info(json_string, wanted_test_name)


    last_img_center_point_x_list, last_img_center_point_y_list, target_img_path, final_lateral_decision_list, final_vertical_decision_list, save_for_future_moveable_position_list, save_for_future_category_list_move, save_for_future_serial_num = calculate_motion_direction_by_tracking_speed_value(
        moveable_position_list_list, target_img_path, category_list_move, serial_num_list)

    new_unmoveable_object_info_list = csv_dataset_maker_for_unmoveable_object(category_list, position_info_list)

    new_moveable_object_info_list = csv_dataset_maker_for_moveable_object(category_list_move, moveable_position_list_list, final_lateral_decision_list, final_vertical_decision_list)

    self_speed = [0, 0]

    create_csv(pesudo_csv_folder_name, wanted_test_name, new_unmoveable_object_info_list, new_moveable_object_info_list, self_speed)

def read_csv_file(file_path, skip_flag = False):
    samples = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        if skip_flag:
            csv_reader.__next__()
        for row in csv_reader:
            # samples.append(row[0] + "_2.jpg")
            samples.append(row[0])
    return samples

if __name__ == "__main__":

    pesudo_csv_folder_name = "object_position_info"

    tracking_info_folder_path = "./BDD_object_position_info/train"

    target_img_path = "pesudo_img_folder_626"

    tracking_file_name_list = os.listdir(tracking_info_folder_path)
    video_environment_name_folder_path = "./BDD_object_position_info/bdd100k_labels_images_train_zhang625.json"

    json_string, environment_name_list = read_environment_name_json_file(video_environment_name_folder_path)
    # required_name_list = filter_for_required_environment(tracking_file_name_list, name_list, environment_name_list) # not useful, only reference

    required_name_list = read_csv_file("zhang_pesudo_img.csv")


    end_frame_serial_num = 52
    calculation_time_interval = 2


    for required_name in tqdm(required_name_list) :
        make_a_pesudo_img(end_frame_serial_num, required_name, target_img_path, calculation_time_interval,
                          tracking_info_folder_path, json_string, pesudo_csv_folder_name)


