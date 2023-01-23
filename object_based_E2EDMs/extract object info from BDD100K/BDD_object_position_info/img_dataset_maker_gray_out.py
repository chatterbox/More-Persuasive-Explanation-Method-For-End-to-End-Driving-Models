import os
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
def draw_line(position_info, target_img_path, original_img_path):
    target_img = cv2.imread(original_img_path)
    (x1, y1) = position_info[0]
    (x2, y2) = position_info[1]
    color = (192, 192, 192)  
    thickness = 30
    lineType = 4

    cv2.line(target_img, (x1, y1), (x2, y2), color,
                 thickness, lineType)
    # assert  len(spotted_y_list) == len(spotted_x_list) or len(spotted_y_list) % 2 == 1, "something wrong with dashed line"
    cv2.imwrite(target_img_path, target_img)

def save_gray_out_imgs_object_info(target_csv_file_path, image_name_list, object_info_list):
    assert len(image_name_list) == len(object_info_list), (len(image_name_list) != len(object_info_list))
    each_info_list = []
    for i in range(len(image_name_list)):
        each_info = [image_name_list[i], object_info_list[i]]
        each_info_list.append(each_info)
    df = pd.DataFrame(each_info_list, columns=["image_name", "object_info"])
    df.to_csv(target_csv_file_path, index = None)

def draw_a_rectangle(object_category, position_info, target_img_path, original_img_path):

    target_img = cv2.imread(original_img_path)
    object_img = cv2.imread(object_category)
    
    x1 = int(position_info["x1"])
    x2 = int(position_info["x2"])
    y1 = int(position_info["y1"])
    y2 = int(position_info["y2"])
    target_size = (x2 - x1, y2 - y1)
    # print(object_category)
    # exit()
    # print(object_category)

    object_img = cv2.resize(object_img, target_size)


    target_img = Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))  # 转换为PIL格式
    object_img = Image.fromarray(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))
    target_img.paste(object_img, (x1, y1))  # img2贴在img1指定位置，位置是(左,上)
    target_img = cv2.cvtColor(np.asarray(target_img), cv2.COLOR_RGB2BGR)  # PIL转换为cv2格式

    target_img = cv2.rectangle(target_img, (x1, y1), (x2, y2), (192, 192, 192), 2)
    cv2.imwrite(target_img_path, target_img)
    

class One_image():
    def __init__(self, image_name):
        self.image_name = image_name
        tracking_info_folder_path = "/home/zhangc/zhangc/dataset/BDD_100K_FLR/explanation/BDD_object_position_info/train"
        self.video_environment_name_folder_path = "/home/zhangc/zhangc/dataset/BDD_100K_FLR/explanation/BDD_object_position_info/bdd100k_labels_images_train_zhang625.json"
        self.img_folder_dir_path = '/home/zhangc/zhangc/dataset/BDD_100K_FLR/imgs_folder'
        self.gray_img_path = "gray.jpg"


        self.tracking_info_file_path = os.path.join(tracking_info_folder_path,
                                           self.image_name + ".json")

        
        self.serial_num_list = [50, 51] # 实际上是51, 52， 我们这里给的是指针，所以需要比实际上的少1
        self.moveable_position_list_list = []
        self.unmoveable_object_position_info = []
        self.lane_position_info_list = []
        
        self.original_img_path_list = []
        self.gray_out_folder_index = 0
        for serial_num in self.serial_num_list:
            original_img_name = self.image_name + "-00000" + str(serial_num + 1) + ".jpg"
            original_img_path = os.path.join(self.img_folder_dir_path, self.image_name, original_img_name)
            assert os.path.exists(original_img_path) == True, ("os.path.exists(original_img_path)", original_img_path)
            self.original_img_path_list.append(original_img_path)
        

    def make_gray_out_image_for_moveable_obejct(self, target_img_folder_path, target_csv_folder_for_each_object_path):
        assert len(self.original_img_path_list) == len(self.moveable_position_list_list), ("original image path list and the moveable object info ont match")
        
        for each_object_index in range(len(self.moveable_position_list_list[0])):
            self.gray_out_folder_index = self.gray_out_folder_index + 1
            image_name_list = []
            object_info_list = []
            new_folder_name =  self.image_name + "_" +str(self.gray_out_folder_index)
            for each_img_index in range(len(self.original_img_path_list)):
                target_img_subfolder_path = os.path.join(target_img_folder_path, new_folder_name)
                if os.path.exists(target_img_subfolder_path) == False:
                    os.mkdir(target_img_subfolder_path)
                original_img_path = self.original_img_path_list[each_img_index]
                new_image_name = self.image_name + "_" +  str(each_img_index) + ".jpg"
                image_name_list.append(new_image_name)
                target_img_path = os.path.join(target_img_subfolder_path, new_image_name)
                each_object_bounding_box_info = self.moveable_position_list_list[each_img_index][each_object_index]
                object_info_list.append(each_object_bounding_box_info)
                draw_a_rectangle(self.gray_img_path, each_object_bounding_box_info, target_img_path, original_img_path)
            new_csv_file_name = new_folder_name + ".csv"
            target_csv_file_path = os.path.join(target_csv_folder_for_each_object_path, new_csv_file_name) 
            save_gray_out_imgs_object_info(target_csv_file_path, image_name_list, object_info_list)
            

        
    def make_gray_out_image_for_unmoveable_object(self, target_img_folder_path, target_csv_folder_for_each_object_path):
        for each_object_index in range(len(self.unmoveable_object_position_info)):
            self.gray_out_folder_index = self.gray_out_folder_index + 1
            image_name_list = []
            object_info_list = []
            new_folder_name =  self.image_name + "_" +str(self.gray_out_folder_index)
            for each_img_index in range(len(self.original_img_path_list)):
                target_img_subfolder_path = os.path.join(target_img_folder_path, new_folder_name)
                if os.path.exists(target_img_subfolder_path) == False:
                    os.mkdir(target_img_subfolder_path)
                original_img_path = self.original_img_path_list[each_img_index]
                new_image_name = self.image_name + "_" +  str(each_img_index) + ".jpg"
                image_name_list.append(new_image_name)
                target_img_path = os.path.join(target_img_subfolder_path, new_image_name)
                each_object_bounding_box_info = self.unmoveable_object_position_info[each_object_index]
                object_info_list.append(each_object_bounding_box_info)
                draw_a_rectangle(self.gray_img_path, each_object_bounding_box_info, target_img_path, original_img_path)
            new_csv_file_name = new_folder_name + ".csv"
            target_csv_file_path = os.path.join(target_csv_folder_for_each_object_path, new_csv_file_name) 
            save_gray_out_imgs_object_info(target_csv_file_path, image_name_list, object_info_list)

    def make_gray_out_image_for_lane_object(self, target_img_folder_path, target_csv_folder_for_each_object_path):
        for each_object_index in range(len(self.lane_position_info_list)):
            self.gray_out_folder_index = self.gray_out_folder_index + 1
            image_name_list = []
            object_info_list = []
            new_folder_name =  self.image_name + "_" +str(self.gray_out_folder_index)
            for each_img_index in range(len(self.original_img_path_list)):
                target_img_subfolder_path = os.path.join(target_img_folder_path, new_folder_name)
                if os.path.exists(target_img_subfolder_path) == False:
                    os.mkdir(target_img_subfolder_path)
                original_img_path = self.original_img_path_list[each_img_index]
                new_image_name = self.image_name + "_" +  str(each_img_index) + ".jpg"
                image_name_list.append(new_image_name)
                target_img_path = os.path.join(target_img_subfolder_path, new_image_name)
                each_start_point_end_point_info = self.lane_position_info_list[each_object_index]
                each_start_point_end_point_info = each_start_point_end_point_info[0]
                object_info_list.append(each_start_point_end_point_info)
                draw_line(each_start_point_end_point_info, target_img_path, original_img_path)
            new_csv_file_name = new_folder_name + ".csv"
            target_csv_file_path = os.path.join(target_csv_folder_for_each_object_path, new_csv_file_name) 
            save_gray_out_imgs_object_info(target_csv_file_path, image_name_list, object_info_list)
    def get_position_info_for_moveable_objects(self):

        with open(self.tracking_info_file_path, 'r') as jsonfile:
            json_string = json.load(jsonfile)
        # print(len(json_string))
        category_list = []
        moveable_id_list_list = []
        occluded_boole_list = []
        truncated_boole_list = []
        for serial_num in self.serial_num_list:
            pseuedo_info = json_string[serial_num]
            # print("pseuedo_info",pseuedo_info)
            # exit()
            moveable_id_list = []
            for moveable_object in pseuedo_info['labels']:
                moveable_id_list.append(moveable_object["id"])
                # print(moveable_object["attributes"]["occluded"])
                # print(moveable_object["attributes"])
                # exit()

            moveable_id_list_list.append(moveable_id_list)
        # print(len(moveable_id_list_list[0]), moveable_id_list_list[0])
        # print(len(moveable_id_list_list[1]), moveable_id_list_list[1])

        # print(moveable_id_list_list)
        interactive_union = set(moveable_id_list_list[0])
        for data in moveable_id_list_list[1:]:
            interactive_union &= set(data)
        # print(interactive_union)
        # print(len(interactive_union), interactive_union)
        # for each_valid_object_id in interactive_union:
        #     print(each_valid_object_id)
        # exit()
        pseuedo_info = json_string[self.serial_num_list[0]]
        for each_valid_object_id in interactive_union:
            for moveable_object in pseuedo_info['labels']:
                if moveable_object["id"] == each_valid_object_id:
                    category_list.append(moveable_object["category"])
                    occluded_boole_list.append(moveable_object["attributes"]["occluded"])
                    truncated_boole_list.append(moveable_object["attributes"]["truncated"])

        moveable_position_list_list = []

        for serial_num in self.serial_num_list:
            pseuedo_info = json_string[serial_num]
            moveable_position_list = []
            for each_valid_object_id in interactive_union:
                for moveable_object in pseuedo_info['labels']:
                    if moveable_object["id"] == each_valid_object_id:
                        moveable_position_list.append(moveable_object["box2d"])

            moveable_position_list_list.append(moveable_position_list)

        # print("category_list",category_list)

        return category_list, moveable_position_list_list

    def get_all_object_info(self):

        with open(self.video_environment_name_folder_path, 'r') as jsonfile:
            json_string = json.load(jsonfile)

        category_list = []
        position_info_list = []
        lane_line_style = ["solid", "dashed", "orange solid", "orange dashed"]

        # 橘色虚线代表所有dashed lane，意味着该线可以跨越，属于该类的是 "laneStyle": "dashed"，且不是road curb
        # 橘色实线代表所有的solid line，意味着该线不可以跨越，属于该类的是 "laneStyle": "solid"，且不是road curb
        # road curb 优先级很高： 为实线
        # crosswalk： 用虚线表示
        # 忽视所有vertical的车线，crosswalk和road curb除外
        # all object: ['person', 'train', 'traffic sign', 'car', 'bike', 'drivable area', 'traffic light', 'motor', 'rider', 'lane', 'truck', 'bus']
        avoid_list = ['person', 'train', 'traffic sign', 'car', 'bike', 'drivable area', 'motor', 'rider', 'truck', 'bus']
        for each_img in json_string:
            if each_img["name"] == self.image_name + ".jpg":
                object_info_list = each_img["labels"]
                flag_lane = False
                for each_object in object_info_list:
                    # print(each_object)
                    if each_object["category"] in avoid_list:
                        continue
                    if each_object["category"] == "lane":
                        # print(each_object)
                        # exit()
                        # print(each_object["poly2d"])
                        # print(each_object["poly2d"][0]["vertices"])
                        # exit()
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
                        # print(each_object["attributes"]["occluded"])
                        # print(each_img["name"])
                        # if each_object["attributes"]["occluded"] == "true":
                        #     print(each_img["name"], each_object["id"])
                        #     exit()
                        if each_object["attributes"]["trafficLightColor"] == "none" or each_object["attributes"]["influence"] == False:
                            pass
                        else:
                            category_list.append(each_object["category"] + each_object["attributes"]["trafficLightColor"] + " " + each_object["attributes"]["shape"])
                            # print(each_object["category"] + each_object["attributes"]["trafficLightColor"] + " " + each_object["attributes"]["shape"])
                            # print(each_object["category"])
                            position_info_list.append(each_object["box2d"])
                    else:
                        category_list.append(each_object["category"])
                        position_info_list.append(each_object["box2d"])
                # if flag_lane == False:
                #     print(each_img["name"], "do not have lane")
                #     exit()
                break # wanted picture has been found
        return category_list, position_info_list
    
    def object_position_info_reader(self):
        
        category_list_move, moveable_position_list_list = self.get_position_info_for_moveable_objects()
        category_list, position_info_list = self.get_all_object_info()   
        
        unmoveable_object_position_info = []
        lane_position_info_list = []
        for serial_num in range(len(position_info_list)):
            i = position_info_list[serial_num]
            if isinstance(i,dict):
                unmoveable_object_position_info.append(i)
            else:
                # grid_info_list = translate_point_line_lane_to_grid_info(i)
                start_end_point_info = translate_point_line_lane_to_start_end_point_info(i)
                lane_position_info_list.append(start_end_point_info)
                
        self.moveable_position_list_list = moveable_position_list_list
        self.unmoveable_object_position_info = unmoveable_object_position_info
        self.lane_position_info_list = lane_position_info_list
        return moveable_position_list_list, unmoveable_object_position_info, lane_position_info_list


def float_list_to_int_list(float_list):
    int_list = []
    for i in float_list:
        int_list.append(int(i))
    return int_list
def translate_point_line_lane_to_start_end_point_info(position_info):

        each_point_list = []
        each_line_list = []
        # print("position_info",position_info)
        #
        # grid_info_list_whole_line = []
        for each_point in position_info:
            each_point = float_list_to_int_list(each_point)
            each_point_list.append(each_point)

        for each_point_serial_num in range(len(each_point_list) - 1) :
            each_line = [(each_point_list[each_point_serial_num][0], each_point_list[each_point_serial_num][1]), (each_point_list[each_point_serial_num + 1][0], each_point_list[each_point_serial_num + 1][1])]
            each_line_list.append(each_line)
        
        return each_line_list

if __name__== '__main__':
    
    img_folder_dir_path = '/home/zhangc/zhangc/dataset/BDD_100K_FLR/imgs_folder'
    target_csv_folder_path  = "/home/zhangc/zhangc/dataset/BDD_100K_FLR/explanation/explanation_500_csv_folder"
    target_img_folder_path =  "/home/zhangc/zhangc/dataset/BDD_100K_FLR/explanation/explanation_500__gray_out_img_folder"
    target_csv_folder_for_each_object_path =  "/home/zhangc/zhangc/dataset/BDD_100K_FLR/explanation/explanation_500__gray_out_csv_folder"

    pesudo_img_csv_path = "/home/zhangc/zhangc/dataset/BDD_100K_FLR/pesudo_img/zhang_pesudo_img.csv"
    pesudo_img_csv = pd.read_csv(pesudo_img_csv_path)
    pesudo_img_name_list = []
    for idx in range(len(pesudo_img_csv)):
        pesudo_img_name = pesudo_img_csv['img_name'].iloc[idx]
        pesudo_img_name = pesudo_img_name[:-7]
        pesudo_img_name_list.append(pesudo_img_name)
    all__img_obejct_num = 0
    count_img = 0
    for img_name in tqdm(pesudo_img_name_list):
        image_object = One_image(img_name)
        moveable_position_list_list, unmoveable_object_position_info, lane_position_info_list = image_object.object_position_info_reader()
        all_obejct_num = len(moveable_position_list_list[0]) + len(unmoveable_object_position_info) + len(lane_position_info_list)
        all__img_obejct_num = all__img_obejct_num + all_obejct_num
        count_img = count_img + 1
        # print("count_img", count_img, len(moveable_position_list_list[0]) + len(unmoveable_object_position_info) + len(lane_position_info_list))
        target_img_subfolder_path = os.path.join(target_img_folder_path, img_name)
        image_object.make_gray_out_image_for_moveable_obejct(target_img_folder_path, target_csv_folder_for_each_object_path)
        image_object.make_gray_out_image_for_unmoveable_object(target_img_folder_path, target_csv_folder_for_each_object_path)
        image_object.make_gray_out_image_for_lane_object(target_img_folder_path, target_csv_folder_for_each_object_path)
        # exit()
    print("all__img_obejct_num", all__img_obejct_num)
