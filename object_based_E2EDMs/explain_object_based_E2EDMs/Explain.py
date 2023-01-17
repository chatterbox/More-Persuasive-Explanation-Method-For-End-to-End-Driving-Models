# coding: UTF-8
from sklearn.metrics import f1_score
import torch
import numpy as np
import os
import csv
from scipy.cluster.vq import vq
import build_attention_mask
from config import Config
import K_means_zhang
from tqdm import tqdm
import sklearn.ensemble
import lime.lime_tabular
import pickle

import warnings
warnings.filterwarnings("ignore")
config = Config()


def adjust_feature_serial_num_in_all_feature(feature_serial_num, each_object_category_num):
    if each_object_category_num < config.moveable_object_num:
        feature_serial_num_in_all_feature = each_object_category_num * config.moveable_object_cluster_num + feature_serial_num

    if config.moveable_object_num <= each_object_category_num < config.moveable_object_num + config.lane_object_num:
        feature_serial_num_in_all_feature = config.moveable_object_num * config.moveable_object_cluster_num + (
                    each_object_category_num - config.moveable_object_num) * config.lane_object_cluster_num + feature_serial_num
        # print((each_object_category_num - config.moveable_object_num) * config.lane_object_cluster_num)

    if config.moveable_object_num + config.lane_object_num <= each_object_category_num < config.moveable_object_num + config.lane_object_num + config.traffic_light_object_num:
        feature_serial_num_in_all_feature = config.moveable_object_num * config.moveable_object_cluster_num + config.lane_object_num * config.lane_object_cluster_num + (
                    each_object_category_num - config.moveable_object_num - config.lane_object_num) * config.traffic_light_object_cluster_num + feature_serial_num
    return feature_serial_num_in_all_feature


def cal_moveableobject_feature(each_object_info, codebook):
    """
    Calculate the features of a single image given the codebook(vocabulary) generated by clustering method (kmeans), each column is the center of the cluster.
    :param img:
    :param codebook:
    :return:
    """
    object_position_info = each_object_info[0:2]
    object_size_info = each_object_info[2:4]
    object_motion_info = each_object_info[4:6]

    object_position_info_codebook = codebook[0]
    object_size_info_codebook = codebook[1]
    object_motion_info_codebook = codebook[2]

    position_info_features = np.zeros((object_position_info_codebook.shape[0]))
    size_info_features = np.zeros((object_size_info_codebook.shape[0]))
    motion_info_features = np.zeros((object_motion_info_codebook.shape[0]))

    object_position_info = object_position_info[np.newaxis, :]
    object_size_info = object_size_info[np.newaxis, :]
    object_motion_info = object_motion_info[np.newaxis, :]

    code, _ = vq(object_position_info, object_position_info_codebook)
    position_info_features[code] += 1

    code, _ = vq(object_size_info, object_size_info_codebook)
    size_info_features[code] += 1

    code, _ = vq(object_motion_info, object_motion_info_codebook)
    motion_info_features[code] += 1

    result_feature = merge_3_kmeans_to_1_kmeans(position_info_features, size_info_features, motion_info_features)

    return result_feature


def cal_lane_object_feature(each_object_info, codebook):
    """
    Calculate the features of a single image given the codebook(vocabulary) generated by clustering method (kmeans), each column is the center of the cluster.
    :param img:
    :param codebook:
    :return:
    """
    object_start_point_info = each_object_info[0:2]
    object_end_point_info = each_object_info[2:4]

    object_start_point_info_codebook = codebook[0]
    object_end_point_info_codebook = codebook[1]

    object_start_point_info_features = np.zeros((object_start_point_info_codebook.shape[0]))
    object_end_point_info_features = np.zeros((object_end_point_info_codebook.shape[0]))

    object_start_point_info = object_start_point_info[np.newaxis, :]
    object_end_point_info = object_end_point_info[np.newaxis, :]

    code, _ = vq(object_start_point_info, object_start_point_info_codebook)
    object_start_point_info_features[code] += 1

    code, _ = vq(object_end_point_info, object_end_point_info_codebook)
    object_end_point_info_features[code] += 1

    result_feature = merge_2_kmeans_to_1_kmeans(object_start_point_info_features, object_end_point_info_features)
    return result_feature


def cal_trafficlight_object_feature(each_object_info, codebook):
    """
    Calculate the features of a single image given the codebook(vocabulary) generated by clustering method (kmeans), each column is the center of the cluster.
    :param img:
    :param codebook:
    :return:
    """
    object_position_info = each_object_info[0:2]
    object_size_info = each_object_info[2:4]

    object_position_info_codebook = codebook[0]
    object_size_info_codebook = codebook[1]

    position_info_features = np.zeros((object_position_info_codebook.shape[0]))
    size_info_features = np.zeros((object_size_info_codebook.shape[0]))

    object_position_info = object_position_info[np.newaxis, :]
    object_size_info = object_size_info[np.newaxis, :]

    code, _ = vq(object_position_info, object_position_info_codebook)
    position_info_features[code] += 1

    code, _ = vq(object_size_info, object_size_info_codebook)
    size_info_features[code] += 1
    result_feature = merge_2_kmeans_to_1_kmeans(position_info_features, size_info_features)
    return result_feature

def find_serial_num_infeature(feature):
    for i in range(len(feature)):
        if feature[i] == 1:
            return i

def merge_2_kmeans_to_1_kmeans(feature_1, feature_2):
    feature_length_list = [len(feature_1), len(feature_2)]

    feature_1_cluster_serial_num = find_serial_num_infeature(feature_1)
    feature_2_cluster_serial_num = find_serial_num_infeature(feature_2)

    result_feature_length = feature_length_list[0] * feature_length_list[1]
    first_group_serial_num = feature_1_cluster_serial_num * feature_length_list[1]
    second_group_serial_num = feature_2_cluster_serial_num

    result_feature_serial_num = first_group_serial_num + second_group_serial_num
    result_feature = [0] * result_feature_length
    result_feature[result_feature_serial_num] = 1

    return result_feature


def merge_3_kmeans_to_1_kmeans(feature_1, feature_2, feature_3):
    feature_length_list = [len(feature_1), len(feature_2), len(feature_3)]

    feature_1_cluster_serial_num = find_serial_num_infeature(feature_1)
    feature_2_cluster_serial_num = find_serial_num_infeature(feature_2)
    feature_3_cluster_serial_num = find_serial_num_infeature(feature_3)

    result_feature_length = feature_length_list[0] * feature_length_list[1] * feature_length_list[2]
    first_group_serial_num = feature_1_cluster_serial_num * feature_length_list[1] * feature_length_list[2]
    second_group_serial_num = feature_2_cluster_serial_num * feature_length_list[2]
    third_group_serial_num = feature_3_cluster_serial_num
    result_feature_serial_num = first_group_serial_num + second_group_serial_num + third_group_serial_num
    result_feature = [0] * result_feature_length
    result_feature[result_feature_serial_num] = 1

    return result_feature

def local_explanation(objects_info_list, one_img_features, batch_names, codebook_list, network_parameters,
                       object_settings_path, causal_flag):



    intermediate_feature = one_img_features * network_parameters
    intermediate_feature = torch.Tensor(intermediate_feature).to("cpu")

    if causal_flag == "all_important":
        intermediate_feature = np.absolute(intermediate_feature)
        object_importance, sort_ind = intermediate_feature.sort(dim=0, descending=True)
    else:
        object_importance, sort_ind = intermediate_feature.sort(dim=0, descending=False)

    object_dict = {}
    with open(object_settings_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # intense_loss_line.append(float(line[0]))
            object_dict[line[0]] = line[1]
        next(reader, None)  # jump to the next line
    top_num = 1
    # len(intermediate_feature)
    all_object_attention = [[], [], []]
    objects_info_list_for_attention = [[], [], []]

    for each_big_category_serial_num in range(len(objects_info_list)):
        for each_object in objects_info_list[each_big_category_serial_num]:
            for i in range(len(intermediate_feature)):
                if each_big_category_serial_num == 0:
                    temp_each_object = each_object[0]
                else:
                    temp_each_object = each_object

                code_serial_num = sort_ind[i]
                each_object_category_num = int(temp_each_object[0]) - 1
                each_object_info = np.array(temp_each_object[1:])

                if each_object_category_num <= 4:
                    this_object_feature = cal_moveableobject_feature(each_object_info,
                                                                     codebook_list[each_object_category_num])

                if 4 < each_object_category_num <= 8:
                    this_object_feature = cal_lane_object_feature(each_object_info,
                                                                  codebook_list[each_object_category_num])

                if 8 < each_object_category_num <= 10:
                    this_object_feature = cal_trafficlight_object_feature(each_object_info,
                                                                          codebook_list[
                                                                              each_object_category_num])
                for k in range(len(this_object_feature)):
                    if this_object_feature[k] == 1:
                        feature_serial_num = k
                feature_serial_num_in_all_feature = adjust_feature_serial_num_in_all_feature(feature_serial_num,
                                                                                             each_object_category_num)
                if feature_serial_num_in_all_feature == code_serial_num:
                    all_object_attention[each_big_category_serial_num].append(object_importance[i].item())
                    objects_info_list_for_attention[each_big_category_serial_num].append(each_object)

    # print("all_object_attention", all_object_attention)
    if causal_flag == "all_important":
        pass
    else:
        for i in range(len(all_object_attention)):
            for j in range(len(all_object_attention[i])):
                if all_object_attention[i][j] > 0:
                    all_object_attention[i][j] = 0
                else:
                    all_object_attention[i][j] = -all_object_attention[i][j]



    for i in range(len(objects_info_list)):
        if len(objects_info_list[i]) != len(objects_info_list_for_attention[i]):
            print("Wrong")
            exit()


    return objects_info_list_for_attention, all_object_attention

def list_normalization(pixel_list):
    list_min = min(pixel_list)
    list_max = max(pixel_list)
    list_ranges = list_max - list_min
    normed_list = []
    # print(pixel_list)
    if list_ranges == 0:
        return pixel_list
    for i in pixel_list:
        normed_value = (i - list_min) / list_ranges
        normed_list.append(normed_value)
    return normed_list


def merge_all_action_explanation_independent(all_object_attention_list):

    each_category_num_list = []

    for i in all_object_attention_list[0]:
        each_category_num_list.append(len(i))
    # print(each_category_num_list)
    all_object_in_one_list_for_norm_list = []

    for object_attention_list in all_object_attention_list:
        all_object_in_one_list_for_norm = []
        for each_category_attention in object_attention_list:
            all_object_in_one_list_for_norm = all_object_in_one_list_for_norm + each_category_attention
        # print(all_object_in_one_list_for_norm)
        all_object_in_one_list_for_norm_list.append(all_object_in_one_list_for_norm)


    norm_object_attention_list = []
    for object_attention_list in all_object_in_one_list_for_norm_list:
        if len(object_attention_list) != 0:
            norm_object_attention_list.append(list_normalization(object_attention_list))
        else:
            norm_object_attention_list.append([])

    normed_for_each_action_object_attention_list = []

    for normed_objects_attention in norm_object_attention_list:
        normed_all_object_attention_3list = [[], [], []]
        for each_object_attention_serial_num in range(len(normed_objects_attention)):
            if each_object_attention_serial_num < each_category_num_list[0]:
                normed_all_object_attention_3list[0].append(normed_objects_attention[each_object_attention_serial_num])
            if each_category_num_list[0] <= each_object_attention_serial_num < each_category_num_list[0] + \
                    each_category_num_list[1]:
                normed_all_object_attention_3list[1].append(normed_objects_attention[each_object_attention_serial_num])
            if each_category_num_list[0] + each_category_num_list[1] <= each_object_attention_serial_num:
                normed_all_object_attention_3list[2].append(normed_objects_attention[each_object_attention_serial_num])

        normed_for_each_action_object_attention_list.append(normed_all_object_attention_3list)

    for serial_num in range(len(all_object_attention_list)):
        for serial_num_1 in range(len(all_object_attention_list[serial_num])):
            # print(len(all_object_attention_list[serial_num][serial_num_1]), len(normed_for_each_action_object_attention_list[serial_num][serial_num_1]), serial_num_1)
            assert len(all_object_attention_list[0][serial_num_1]) == len(normed_for_each_action_object_attention_list[0][serial_num_1]), ("length dont match" )



    all_object_attention_list = normed_for_each_action_object_attention_list

    np_object_attention_list = np.array(all_object_attention_list[0])

    for serial_num in range(1, len(all_object_attention_list)):
        for each_category_attention_list_serial_num in range(len(all_object_attention_list[serial_num])):
            np_object_attention_list[each_category_attention_list_serial_num] = np_object_attention_list[
                                                                                    each_category_attention_list_serial_num] + np.array(
                all_object_attention_list[serial_num][each_category_attention_list_serial_num])

    each_category_num_list = []
    all_object_in_one_list_for_norm = []
    normed_all_object_attention_3list = [[], [], []]
    for i in np_object_attention_list:
        all_object_in_one_list_for_norm = np.concatenate((all_object_in_one_list_for_norm, i))
        each_category_num_list.append(len(i))
    # print("all_object_in_one_list_for_norm", all_object_in_one_list_for_norm)
    normed_objects_attention = list_normalization(all_object_in_one_list_for_norm)

    # print(len(normed_objects_attention), each_category_num_list)
    for each_object_attention_serial_num in range(len(normed_objects_attention)):
        if each_object_attention_serial_num < each_category_num_list[0]:
            normed_all_object_attention_3list[0].append(normed_objects_attention[each_object_attention_serial_num])
        if each_category_num_list[0] <= each_object_attention_serial_num < each_category_num_list[0] + \
                each_category_num_list[1]:
            normed_all_object_attention_3list[1].append(normed_objects_attention[each_object_attention_serial_num])
        if each_category_num_list[0] + each_category_num_list[1] <= each_object_attention_serial_num:
            normed_all_object_attention_3list[2].append(normed_objects_attention[each_object_attention_serial_num])

    for i in normed_objects_attention:
        assert i >= 0, ("attention smaller than 0", i, normed_objects_attention)

    return normed_all_object_attention_3list


def exolanation_lime_list_transform_for_show(lime_explanation):
    explanation_dict = {}
    for single_explanation in lime_explanation:
        explanation_dict[single_explanation[0].split("=")[0]] = single_explanation[1]
    ouput_explanation_list = []
    for i in range(168):
        ouput_explanation_list.append(explanation_dict[str(i)])
    return ouput_explanation_list




class custom_corss_validation:
    def __init__(self, cross_each_cross_validation_iter_list):
        self.cross_each_cross_validation_iter_list = cross_each_cross_validation_iter_list
        input_list, _ = read_info_outof_cross_validation_dataset(self.cross_each_cross_validation_iter_list)
        self.data_length = len(input_list)

        self.each_cross_valdiation_part_num = self.data_length / len(cross_each_cross_validation_iter_list)

        assert self.data_length % len(cross_each_cross_validation_iter_list) == 0, (self.each_cross_valdiation_part_num, "self.each_cross_valdiation_part_num is not a int, check cross validation k fold number")

    def get_dataset_part_fold_by_serial_num(self, cross_validation_serial_num):
        self.test_dataset_part = self.cross_each_cross_validation_iter_list[cross_validation_serial_num]
        if cross_validation_serial_num >= ( len(self.cross_each_cross_validation_iter_list) - 1 ):
            vali_serial_num = 0
        else:
            vali_serial_num = cross_validation_serial_num + 1
        self.validation_dataset_part = self.cross_each_cross_validation_iter_list[vali_serial_num]
        self.train_dataset_part = []
        for index in range(len(self.cross_each_cross_validation_iter_list)):
            if index == cross_validation_serial_num or index == vali_serial_num:
                continue
            self.train_dataset_part.append(self.cross_each_cross_validation_iter_list[index])

        return self.train_dataset_part, self.validation_dataset_part, self.test_dataset_part
    def get_cv_split_fold_by_serial_num(self, cross_validation_serial_num):
        test_index_start_flag = cross_validation_serial_num * self.each_cross_valdiation_part_num
        test_index_end_flag = (cross_validation_serial_num + 1) * self.each_cross_valdiation_part_num
        test_np_array = np.arange(test_index_start_flag, test_index_end_flag, 1, int)

        if cross_validation_serial_num >= ( len(self.cross_each_cross_validation_iter_list) - 1 ):
            vali_serial_num = 0
        else:
            vali_serial_num = cross_validation_serial_num + 1

        vali_index_start_flag = vali_serial_num * self.each_cross_valdiation_part_num
        vali_index_end_flag = (vali_serial_num + 1) * self.each_cross_valdiation_part_num
        vali_np_array = np.arange(vali_index_start_flag, vali_index_end_flag, 1, int)
        train_np_array = []
        for index in range(self.data_length):
            if index in test_np_array or index in vali_np_array:
                continue
            train_np_array.append(index)
        train_np_array = np.array(train_np_array)
        return train_np_array, vali_np_array, test_np_array

def read_info_outof_cross_validation_dataset(dataset_part):
    if isinstance(dataset_part, list) == True:
        all_train_input_list = []
        all_train_output_list = []
        for each_element_in_list in dataset_part:
            train_input_list, train_output_list = read_info_outof_cross_validation_dataset(each_element_in_list)

            all_train_input_list = all_train_input_list + train_input_list
            all_train_output_list = all_train_output_list + train_output_list
        all_train_input_list = np.array(all_train_input_list)
        all_train_output_list = np.array(all_train_output_list)

        return  all_train_input_list, all_train_output_list
    else:
        train_input_list = []
        train_output_list = []
        for (batch_object_features, batch_labels, batch_names, batch_self_speed_info_list,
             batch_all_objects_info_list) in dataset_part:
            for batch_serial_num in range(len(batch_object_features)):
                train_input_list.append(batch_object_features[batch_serial_num])
                train_output_list.append(batch_labels[batch_serial_num])

        return train_input_list, train_output_list


class custom_corss_validation:
    def __init__(self, cross_each_cross_validation_iter_list):
        self.cross_each_cross_validation_iter_list = cross_each_cross_validation_iter_list
        input_list, _ = read_info_outof_cross_validation_dataset(self.cross_each_cross_validation_iter_list)
        self.data_length = len(input_list)

        self.each_cross_valdiation_part_num = self.data_length / len(cross_each_cross_validation_iter_list)

        assert self.data_length % len(cross_each_cross_validation_iter_list) == 0, (self.each_cross_valdiation_part_num, "self.each_cross_valdiation_part_num is not a int, check cross validation k fold number")

    def get_dataset_part_fold_by_serial_num(self, cross_validation_serial_num):
        self.test_dataset_part = self.cross_each_cross_validation_iter_list[cross_validation_serial_num]
        if cross_validation_serial_num >= ( len(self.cross_each_cross_validation_iter_list) - 1 ):
            vali_serial_num = 0
        else:
            vali_serial_num = cross_validation_serial_num + 1
        self.validation_dataset_part = self.cross_each_cross_validation_iter_list[vali_serial_num]
        self.train_dataset_part = []
        for index in range(len(self.cross_each_cross_validation_iter_list)):
            if index == cross_validation_serial_num or index == vali_serial_num:
                continue
            self.train_dataset_part.append(self.cross_each_cross_validation_iter_list[index])

        return self.train_dataset_part, self.validation_dataset_part, self.test_dataset_part
    def get_cv_split_fold_by_serial_num(self, cross_validation_serial_num):
        test_index_start_flag = cross_validation_serial_num * self.each_cross_valdiation_part_num
        test_index_end_flag = (cross_validation_serial_num + 1) * self.each_cross_valdiation_part_num
        test_np_array = np.arange(test_index_start_flag, test_index_end_flag, 1, int)

        if cross_validation_serial_num >= ( len(self.cross_each_cross_validation_iter_list) - 1 ):
            vali_serial_num = 0
        else:
            vali_serial_num = cross_validation_serial_num + 1

        vali_index_start_flag = vali_serial_num * self.each_cross_valdiation_part_num
        vali_index_end_flag = (vali_serial_num + 1) * self.each_cross_valdiation_part_num
        vali_np_array = np.arange(vali_index_start_flag, vali_index_end_flag, 1, int)
        train_np_array = []
        for index in range(self.data_length):
            if index in test_np_array or index in vali_np_array:
                continue
            train_np_array.append(index)
        train_np_array = np.array(train_np_array)
        return train_np_array, vali_np_array, test_np_array


def make_subjective_and_objective_explanation_experiment_data(target_object_based_E2EDMs_name):

    # prepare the dataset
    cross_each_cross_validation_iter_list, test_iter_list, all_object_codebook = K_means_zhang.K_means_zhang_train_iter()

    # each_model saved position
    save_model_target_path = "cross_validation_saved_model/" + target_object_based_E2EDMs_name

    model_name_list = ["Forward", "Left", "Right"]

    original_image_label_Dict = {}

    all_score1_all = 0

    # cross validation, explain the object-based model for each test dataset
    for index in range(5):
        custom_corss_validation_instance = custom_corss_validation(cross_each_cross_validation_iter_list)

        train_dataset_part, validation_dataset_part, test_dataset_part = custom_corss_validation_instance.get_dataset_part_fold_by_serial_num(index)
        test_data_input, test_data_ouput = read_info_outof_cross_validation_dataset(
            test_dataset_part)
        train_data_input, train_data_ouput = read_info_outof_cross_validation_dataset(
            train_dataset_part)

        with open(os.path.join(save_model_target_path, model_name_list[0] + "_" + str(index) + ".pkl"), 'rb') as f:
            RF_forward = pickle.load(f)
        with open(os.path.join(save_model_target_path, model_name_list[1] + "_" + str(index) + ".pkl"), 'rb') as f:
            RF_left = pickle.load(f)
        with open(os.path.join(save_model_target_path, model_name_list[2] + "_" + str(index) + ".pkl"), 'rb') as f:
            RF_right = pickle.load(f)

        pred_forward = RF_forward.predict(test_data_input)
        pred_left = RF_left.predict(test_data_input)
        pred_right = RF_right.predict(test_data_input)
        test_data_ouput = np.array(test_data_ouput)
        test_forward_data = test_data_ouput[:, 0]
        test_left_data = test_data_ouput[:, 1]
        test_right_data = test_data_ouput[:, 2]

        score_forward = sklearn.metrics.f1_score(test_forward_data, pred_forward, average='binary')
        score1_left = sklearn.metrics.f1_score(test_left_data, pred_left, average='binary')
        score1_right = sklearn.metrics.f1_score(test_right_data, pred_right, average='binary')

        all_score1 = (score_forward + score1_left + score1_right) / 3
        all_score1_all = all_score1_all + all_score1
        # print(all_score1)
        # continue
        test_iter = test_iter_list[index]

        categorical_features = range(168)
        explainer = lime.lime_tabular.LimeTabularExplainer(train_data_input,
                                                           categorical_features=categorical_features)

        # explain the data from the test dataset
        for (batch_object_features, batch_labels, batch_names, batch_self_speed_info_list,
             batch_all_objects_info_list) in test_iter:
            for batch_serial_num in tqdm(range(len(batch_object_features))):

                each_img_local_folder_name = batch_names[batch_serial_num][:-4]
                one_img_features = batch_object_features[batch_serial_num]

                # print("one_img_features",one_img_features)
                # one_img_features_reshaped: input
                one_img_features_reshaped = one_img_features.reshape(1, -1)

                forward_label = RF_forward.predict(one_img_features_reshaped)
                left_label = RF_left.predict(one_img_features_reshaped)
                right_label = RF_right.predict(one_img_features_reshaped)
                original_action_label = [forward_label[0], left_label[0], right_label[0]]
                original_image_label_Dict[each_img_local_folder_name] = original_action_label

                forward_exp = explainer.explain_instance(one_img_features_reshaped[0], RF_forward.predict_proba,
                                                         num_features=168)
                left_exp = explainer.explain_instance(one_img_features_reshaped[0], RF_left.predict_proba,
                                                      num_features=168)
                right_exp = explainer.explain_instance(one_img_features_reshaped[0], RF_right.predict_proba,
                                                       num_features=168)
                forward_exp = exolanation_lime_list_transform_for_show(forward_exp.as_list())
                left_exp = exolanation_lime_list_transform_for_show(left_exp.as_list())
                right_exp = exolanation_lime_list_transform_for_show(right_exp.as_list())

                # the object importance could also be extracted from the coefficient from the Logistics regression model
                # forward_exp = RF_forward.coef_.squeeze().tolist()
                # left_exp = RF_left.coef_.squeeze().tolist()
                # right_exp = RF_right.coef_.squeeze().tolist()

                network_parameters_actions = [forward_exp, left_exp, right_exp]

                one_img_label = batch_labels[batch_serial_num]
                all_object_attention_list = []

                for i in range(3):
                    objects_info_list, all_object_attention = local_explanation(
                        batch_all_objects_info_list[batch_serial_num], one_img_features,
                        batch_names[batch_serial_num], all_object_codebook,
                        network_parameters_actions[i], config.object_settings_path_reverse, "all_important")

                    all_object_attention_list.append(all_object_attention)

                # merge the explanations of three E2EDMs (forward, left, right driving models) to a single explanation
                sum_normed_all_object_attention = merge_all_action_explanation_independent(
                    all_object_attention_list)

                all_object_info_and_attention = [objects_info_list, sum_normed_all_object_attention]

                if len(all_object_attention_list) != 0:
                    # make heatmap for subjective evaluation
                    build_attention_mask.make_attention_mask_interface(all_object_info_and_attention,
                                                                       each_img_local_folder_name,
                                                                       os.path.join(config.original_image_heatmap, target_object_based_E2EDMs_name),
                                                                       os.path.join(config.heatmap_mask, target_object_based_E2EDMs_name))
                    # make partially shown images for objective evaluation
                    build_attention_mask.gradual_image_based_on_explanation(all_object_info_and_attention,
                                                                            each_img_local_folder_name,
                                                                            os.path.join(config.simulation_gradual_image,
                                                                                         target_object_based_E2EDMs_name))


if __name__ == '__main__':

    object_based_model_name_list = ["MLP", "RF", "LR"]
    for object_based_model_name in object_based_model_name_list:
        make_subjective_and_objective_explanation_experiment_data(object_based_model_name)


