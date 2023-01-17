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
import math
import matplotlib.pyplot as plt
gradual_step_num = 4
time_step_num = 2

def read_csv_file(file_path, skip_flag = False):
    samples = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)  #
        if skip_flag:
            csv_reader.__next__()
        for row in csv_reader:  # Ai first we let the 1 be the best, at last, we decide let the 5 be the best, so we change it
            if row[1] == "1":
                row[1] = "5"
            elif row[1] == "2":
                row[1] = "4"
            elif row[1] == "4":
                row[1] = "2"
            elif row[1] == "5":
                row[1] = "1"
            samples.append(row)
    return samples


def subjective_average_score_calculator(samples_for_each_category):
    score = 0
    for sample in samples_for_each_category:
        score = score + int(sample[1])
    return score/len(samples_for_each_category)

def mixed_samples_seperater_and_f1score_calculator_for_change(gradual_image_prediction_samples):
    human_label_dict_dict = {}
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

    for category_index_num in range(len(category_list)):
        each_category = category_list[category_index_num]
        human_label_dict_dict[each_category] = samples_list_for_each_category[category_index_num]

    target_sort_list = []
    all_info_list = []

    for key in human_label_dict_dict:
        category_name = key

        samples_for_each_category = human_label_dict_dict[key]

        subjective_average_score = subjective_average_score_calculator(samples_for_each_category)

        target_sort_list.append(subjective_average_score)
        # all_info_list.append([category_name, ])
        result_dict[category_name] = round(subjective_average_score, 2)



    return result_dict


def fake_main_2(result_dict, sub_data_folder_name_path, temp_result_30dict_list):


    file_path = os.path.join(sub_data_folder_name_path, "heatmap.csv")
    image_prediction_samples = read_csv_file(file_path)


    temp_result_dict = mixed_samples_seperater_and_f1score_calculator_for_change(image_prediction_samples)
    temp_result_30dict_list.append(temp_result_dict)
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
    temp_result_30dict_list = []

    data_dict_list = []
    for i in range(10):
        data_dict_list.append([])

    base_img_path = os.path.abspath('.')
    data_folder_path = os.path.split(base_img_path)[0]
    self_folder_name = os.path.split(base_img_path)[1]
    sub_data_folder_name_list = os.listdir(data_folder_path)
    sub_data_folder_name_list.remove(self_folder_name)

    overall_result_dict = {}
    overall_sub_data_folder_num = 0
    for sub_data_folder_name in sub_data_folder_name_list:
        sub_data_folder_name_path = os.path.join(data_folder_path, sub_data_folder_name)

        sub_data_folder_num, result_dict, temp_result_30dict_list, data_dict_list = fake_main(sub_data_folder_name_path, temp_result_30dict_list, data_dict_list)

        overall_sub_data_folder_num = overall_sub_data_folder_num + sub_data_folder_num

        for key in result_dict:
            if key in overall_result_dict:
                overall_result_dict[key] = overall_result_dict[key] + result_dict[key]
            else:
                overall_result_dict[key] = result_dict[key]



    print(overall_sub_data_folder_num, overall_result_dict)
    for key in overall_result_dict:
        overall_result_dict[key] = overall_result_dict[key] / overall_sub_data_folder_num
    temp_tuple = zip(overall_result_dict.values(), overall_result_dict.keys())
    result_list = list(sorted(temp_tuple))
    for i in result_list[::-1]:
        print(i)

