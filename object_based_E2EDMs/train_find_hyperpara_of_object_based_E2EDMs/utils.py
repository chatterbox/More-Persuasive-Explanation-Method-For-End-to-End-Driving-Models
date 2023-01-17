# coding: UTF-8
import os
import torch
import numpy as np

import csv



def build_dataset(cross_validation_folder_path_pre, object_settings_path, action_label_csv_path):

    def float_list__to_int_list(float_list):
        new_list = []
        for i in float_list:
            new_list.append(int(i))
        return new_list


    def load_dataset(folder_path, object_settings_path):
        file_name_list = os.listdir(folder_path)
        object_dict = {}
        max_category_num = 0
        with open(object_settings_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                # intense_loss_line.append(float(line[0]))
                object_dict[line[0]] = line[1]
                if max_category_num <= int(line[1]):
                    max_category_num = int(line[1])
            next(reader, None)  # jump to the next line
        # print(object_dict)
        # exit()
        all_imgs_contents = []
        for single_csv_file in file_name_list:
            single_csv_path = os.path.join(folder_path, single_csv_file)
            single_imgs_contents = []

            moveable_object_list = []
            lane_object_list = []
            traffic_light_list = []

            # print("single_csv_path", single_csv_path)
            with open(single_csv_path) as csvfile:
                reader = csv.reader(csvfile)
                line_flag = 0
                for line in reader:
                    if line_flag == 0:
                        self_speed_info_x = int(line[0])
                        self_speed_info_y = int(line[1])
                        self_speed_info = [self_speed_info_x, self_speed_info_y]
                        # print(self_speed_info)
                        # exit()
                        line_flag = line_flag + 1
                        continue

                    object_name = line[0]
                    object_serial_num = int(object_dict[object_name])
                    # print(object_dict[object_name])
                    # exit()
                    if object_serial_num <= 5 and object_serial_num >= 1:

                        object_info = [int(object_dict[object_name])] + float_list__to_int_list(line[1:])
                        previous_img_object_info = object_info[-4:]
                        object_info = [object_info[:-4], previous_img_object_info]

                        moveable_object_list.append(object_info)
                        # print("object_info", object_info)

                    if object_serial_num > 5 and object_serial_num <= 9:
                        start_end_point_string = line[1]
                        # start_end_point_string = "[[241, 546, 579, 524], [579, 524, 867, 521], [867, 521, 1172, 536]]"
                        start_end_point_string = start_end_point_string[1:]
                        start_end_point_string = start_end_point_string[:-1]
                        start_end_point_list = start_end_point_string.split("], ")

                        for flag, each_start_end_point in enumerate(start_end_point_list) :
                            each_start_end_point = each_start_end_point[1:]
                            if flag == len(start_end_point_list) - 1:
                                each_start_end_point = each_start_end_point[:-1]
                            point_list = each_start_end_point.split(", ")


                            int_point_list = []
                            for i in point_list:
                                int_point_list.append(int(i))
                            object_info = [int(object_dict[object_name])] + int_point_list
                            lane_object_list.append(object_info)


                    if object_serial_num > 9 and object_serial_num <= 11:
                        object_info = [int(object_dict[object_name])] + float_list__to_int_list(line[1:])
                        traffic_light_list.append(object_info)



            moveable_object_num_for_an_img = len(moveable_object_list)
            lane_object_num_for_an_img = len(lane_object_list)
            traffic_light_object_num_for_an_img = len(traffic_light_list)


            action_label = None
            with open(action_label_csv_path) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    # intense_loss_line.append(float(line[0]))

                    if line[0][:-7] == single_csv_file[:-4]:
                        action_label = float_list__to_int_list(line[1:])

                        reverse_action_label = []
                        for each_action in action_label:
                            if each_action == 1:
                                reverse_action_label.append(0)
                            else:
                                reverse_action_label.append(1)
                        action_label = reverse_action_label

                next(reader, None)  # jump to the next line
            # print(action_label)
            # exit()
            moveable_object_list = np.array(moveable_object_list)
            lane_object_list = np.array(lane_object_list)
            traffic_light_list = np.array(traffic_light_list)

            action_label = np.array(action_label)

            three_kinds_of_object_num_in_an_img = (moveable_object_num_for_an_img, lane_object_num_for_an_img, traffic_light_object_num_for_an_img)
            all_objects_info = (moveable_object_list, lane_object_list, traffic_light_list)
            one_img_info = (self_speed_info, all_objects_info, action_label, single_csv_file, three_kinds_of_object_num_in_an_img)

            all_imgs_contents.append(one_img_info)
        return all_imgs_contents, max_category_num

    dataset_loader_cross_validation_list = []
    all_data = []
    for i in range(5):
        cross_validation_folder_path = cross_validation_folder_path_pre + str(i)
        train, max_category_num = load_dataset(cross_validation_folder_path, object_settings_path)
        all_data = all_data + train
        dataset_loader_cross_validation_list.append(train)



    return dataset_loader_cross_validation_list, all_data, max_category_num


def int_list_to_tensor(int_list, device):
    tensor_list = torch.Tensor(int_list).to(device)
    return  tensor_list



class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数

        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device


    def _to_package(self, datas):

        input_object_info_list = []
        output_label_list = []
        name_list = []
        self_speed_info_list = []
        all_objects_info_list = []
        for single_img in datas:
            input_object_info = single_img[0]
            output_label = single_img[1]
            file_name = single_img[2]
            self_speed_info = single_img[3]
            all_objects_info = single_img[4]

            input_object_info_list.append(input_object_info)


            output_label_list.append(output_label)
            name_list.append(file_name)
            self_speed_info_list.append(self_speed_info)
            all_objects_info_list.append(all_objects_info)

        return input_object_info_list, output_label_list, name_list, self_speed_info_list, all_objects_info_list

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_package(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_package(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, batch_size, device):
    iter = DatasetIterater(dataset, batch_size, device)
    return iter



