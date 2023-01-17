import sys
import os
import csv
import shutil
from tqdm import tqdm


if __name__=="__main__":


    base_img_path = os.path.abspath('.')

    # folders_folder = "./unmixed_simulation"
    # target_folder = "./mixed_simulation"

    folders_folder = "./unmixed_heatmap"
    target_folder = "./mixed_heatmap"

    folders_folder_path = os.path.join(base_img_path, folders_folder)
    files = os.listdir(folders_folder_path)

    sub_folder_name_list = []

    for sub_folder_name in files:
        sub_folder_name_list.append(sub_folder_name)
        sub_folder_folder_path = os.path.join(folders_folder_path, sub_folder_name)
        sub_folder_img_files_list = os.listdir(sub_folder_folder_path)
        sub_folder_img_files_path_list = []

        for each_img_name in tqdm(sub_folder_img_files_list):
            old_img_file_path = os.path.join(sub_folder_folder_path, each_img_name)
            new_img_file_name = sub_folder_name + "_" + each_img_name

            # new_img_file_path = os.path.join(sub_folder_folder_path, new_img_file_name)
            # os.rename(old_img_file_path, new_img_file_path)

            target_img_file_path = os.path.join(target_folder, new_img_file_name)
            shutil.copy(old_img_file_path, target_img_file_path)