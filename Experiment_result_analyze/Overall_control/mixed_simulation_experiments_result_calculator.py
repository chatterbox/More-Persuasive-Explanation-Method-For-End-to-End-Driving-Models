import sys
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import csv
import pandas as pd
from sklearn.metrics import f1_score
import torch
import numpy as np
import matplotlib.pyplot as plt
gradual_step_num = 4
time_step_num = 2

def read_csv_file(file_path, skip_flag = False):
    samples = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        if skip_flag:
            csv_reader.__next__()
        for row in csv_reader:
            samples.append(row)
    return samples

def mixed_samples_seperater_and_f1score_calculator(gradual_image_prediction_samples, groud_truth_dict):
    result_dict = {}
    category_list = []
    samples_list_for_each_category = []
    for sample in gradual_image_prediction_samples:

        category_name = sample[0].split("_")[0]
        if category_name in category_list:
            category_index_num = category_list.index(category_name)
            samples_list_for_each_category[category_index_num].append(sample)
        else:
            samples_list_for_each_category.append([sample])
            category_list.append(category_name)
    target_sort_list = []
    all_info_list = []
    for index_for_each_category in range(len(samples_list_for_each_category)):
        category_name, each_driving_action_list_list, groud_truth_list = action_list_maker(category_list[index_for_each_category], samples_list_for_each_category[index_for_each_category], groud_truth_dict)
        f1_score_list, counter_list = f1_score_calculator(each_driving_action_list_list, groud_truth_list)
        target_sort_list.append(sum(f1_score_list)/len(f1_score_list))
        # target_sort_list.append(sum(counter_list)/len(counter_list))
        # all_info_list.append([category_name, f1_score_list, round(sum(f1_score_list)/len(f1_score_list), 2), counter_list, sum(counter_list)/len(counter_list)])
        result_dict[category_name] = round(sum(f1_score_list) / len(f1_score_list), 3)


    return result_dict

def action_list_maker(category_name, gradual_image_prediction_samples, groud_truth_dict):
    driving_action_dict = {}
    each_driving_action_list_list = []
    groud_truth_list = []

    for sample in gradual_image_prediction_samples:
        sample_name = sample[0].split("_")[1]

        float_action = list(map(float, sample[1:-2]))

        if sample_name in driving_action_dict:
            temp_list = driving_action_dict[sample_name] + [float_action]
            driving_action_dict[sample_name] = temp_list
        else:
            driving_action_dict[sample_name] = [float_action]

    for key in driving_action_dict:
        action_length = len(driving_action_dict[key])
        break

    for key in driving_action_dict:
        groud_truth_list.append(groud_truth_dict[key])


    for gradual_action_serial_number in range(action_length):
        each_driving_action_list = []
        for key in driving_action_dict:

            each_driving_action_list.append(driving_action_dict[key][gradual_action_serial_number])

        each_driving_action_list_list.append(each_driving_action_list)

    return category_name, each_driving_action_list_list, groud_truth_list

def int_str_to_boole(boole_str):
    if boole_str == "1":
        return True
    else:
        return False
def driving_action_dict_maker(model_prediction_samples, read_info_type = "string", post_name_del = None):
    driving_action_dict = {}
    for i in model_prediction_samples:
        if read_info_type == "string":
            predict_action = list(map(boole_str_to_float, i[1:]))
        elif read_info_type == "int":
            predict_action = list(map(int_str_to_boole, i[1:-2]))

        elif read_info_type == "float":
            action_list = list(map(float, i[1:]))
            prediction = np.array(action_list)
            prediction = torch.from_numpy(prediction)
            predict_action = torch.sigmoid(prediction) > 0.5
            predict_action = predict_action.tolist()
        else:
            print("Unclear type")
            exit()


        if post_name_del != None:
            img_name_list = i[0][:-post_name_del].split("-")
        else:
            img_name_list = i[0].split("-")

        img_name = img_name_list[0] + "-" + img_name_list[1]
        driving_action_dict[img_name] = predict_action
    return driving_action_dict

def f1_score_calculator(driving_action_list1, groud_truth_list):
    ground_truth_action_list = groud_truth_list
    f1_score_list = []
    counter_list = []
    for driving_action_list in driving_action_list1:



        driving_action_list = np.array(driving_action_list)
        ground_truth_action_list = np.array(groud_truth_list)



        f1_score_list.append(round(f1_score(driving_action_list, ground_truth_action_list, average='macro'), 3))

        counter = 0
        for j in range(len(driving_action_list)):
            for k in range(3):
                if (driving_action_list[j][k] == ground_truth_action_list[j][k]):
                    counter = counter + 1
        counter_list.append(counter)


    return f1_score_list, counter_list




def fake_main_2(result_dict, sub_data_folder_name_path, temp_result_30dict_list):


    groud_truth_label_csv_path = os.path.join(sub_data_folder_name_path, "original_image.csv")
    groud_truth_samples = read_csv_file(groud_truth_label_csv_path)
    groud_truth_dict = driving_action_dict_maker(groud_truth_samples, read_info_type = "int")


    file_name = "simulation.csv"
    file_path = os.path.join(sub_data_folder_name_path, file_name)
    gradual_image_prediction_samples = read_csv_file(file_path)

    temp_result_dict = mixed_samples_seperater_and_f1score_calculator(gradual_image_prediction_samples, groud_truth_dict)
    temp_result_30dict_list.append(temp_result_dict)

    # exit()
    for key in temp_result_dict:
        if key in result_dict:
            result_dict[key] = result_dict[key] + temp_result_dict[key]
        else:
            result_dict[key] = temp_result_dict[key]

    return result_dict, temp_result_30dict_list, temp_result_dict


def fake_main(sub_data_folder_name_path, temp_result_30dict_list, data_dict_list):
    data_folder_path = sub_data_folder_name_path

    sub_data_folder_name_list = os.listdir(data_folder_path)


    result_dict = {}
    for sub_data_folder_name in sub_data_folder_name_list:
        sub_data_folder_name_path = os.path.join(data_folder_path, sub_data_folder_name)

        result_dict, temp_result_30dict_list, temp_result_dict = fake_main_2(result_dict, sub_data_folder_name_path, temp_result_30dict_list)
        data_dict_list[int(sub_data_folder_name)].append(temp_result_dict)

    return len(sub_data_folder_name_list), result_dict, temp_result_30dict_list, data_dict_list

if __name__=="__main__":
    # We now analyze the experimental results, the experimental results are saved in a strange way. Plz see the current data arange example
    # in this research we gathered different participants experimental results. Each participant have done several subset of experimental results (I divide the all experimental data into 10 subsets).
    # This code could evaluate all the results, every experimental subset of every participant it's considered.
    # The CNN2D-LSTM is LRCN-18, CNN3D-LSTM is LRCN-50, CNN3D is CNN3D in the paper.
    temp_result_30dict_list = []
    data_dict_list = []  # 每个元素是三个人做出的三个dict，每个dict里有九个模型的分数
    for i in range(10):
        data_dict_list.append([])


    # get the names of the folders which are the same level with the "Overall_control", this py file is in this "Overall_control" folder
    base_img_path = os.path.abspath('.')
    data_folder_path = os.path.split(base_img_path)[0]
    self_folder_name = os.path.split(base_img_path)[1]
    sub_data_folder_name_list = os.listdir(data_folder_path)
    sub_data_folder_name_list.remove(self_folder_name)
    # sub_data_folder_name_list represents all participants


    # the key of this dict is the explanation name, e.g. DP-pixel-CNN3D is the pixel-level explanation from the pixel-based E2EDMs CNN3D.
    # This dict save the objective persuasibility score of all participants for each subset
    overall_result_dict = {}
    overall_sub_data_folder_num = 0
    # "sub_data_folder_name" represents a participant, each folder inside "sub_data_folder_name" is a subset of experimental results, which contains three csv files
    # every loop is to add up every participant
    for sub_data_folder_name in sub_data_folder_name_list:
        sub_data_folder_name_path = os.path.join(data_folder_path, sub_data_folder_name)

        # "result_dict" save the objective persuasibility score of current participants for each subset
        # "temp_result_30dict_list" is a list, every element is a dict, the key of every dict is the explanation name, the dict saves objective persuasibility score
        # "sub_data_folder_num" is the subsets number that the current participant have done
        # "data_dict_list" is is a list, every element of this list is a dict, the key of every dict is the subset name, the dict saves objective persuasibility score
        # we only use the "sub_data_folder_num"
        sub_data_folder_num, result_dict, temp_result_30dict_list, data_dict_list = fake_main(sub_data_folder_name_path, temp_result_30dict_list, data_dict_list)


        # count how many subsets this current participant have done, in order to calculate the average score.
        overall_sub_data_folder_num = overall_sub_data_folder_num + sub_data_folder_num


        for key in result_dict:
            if key in overall_result_dict:
                overall_result_dict[key] = overall_result_dict[key] + result_dict[key]
            else:
                overall_result_dict[key] = result_dict[key]


    for key in overall_result_dict:
        overall_result_dict[key] = overall_result_dict[key] / overall_sub_data_folder_num

    temp_tuple = zip(overall_result_dict.values(), overall_result_dict.keys())
    result_list = list(sorted(temp_tuple))
    for i in result_list[::-1]:
        print(i)
