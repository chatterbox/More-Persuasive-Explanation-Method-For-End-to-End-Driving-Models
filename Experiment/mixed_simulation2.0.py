import sys
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import csv
import pandas as pd
import time
gradual_step_num = 4
time_step_num = 2
class GUIclass(QWidget):
    def __init__(self, img_set_path_list, wanted_info_csv_path):
        super().__init__()
        self.initUI(img_set_path_list, wanted_info_csv_path)

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_W:
            if self.Forward_flag == 1:
                self.NoForward()
            else:
                self.Forward()
        if key == Qt.Key_A:
            if self.TurnLeft_flag == 1:
                self.NoTurnLeft()
            else:
                self.TurnLeft()
        if key == Qt.Key_D:
            if self.TurnRight_flag == 1:
                self.NoTurnRight()
            else:
                self.TurnRight()

        if key == Qt.Key_Z:
            self.show_time_step_1()
        if key == Qt.Key_C:
            self.show_time_step_2()


    def initUI(self, img_set_path_list, wanted_info_csv_path):
        self.time_bank = time.time()
        self.unfinished_imgs_path_list_length = len(img_set_path_list)
        self.save_info_list = []
        self.myAddPic_counter = 0
        self.img_set_path_list = img_set_path_list
        self.csv_log_path = wanted_info_csv_path
        self.Forward_flag = 1
        self.TurnRight_flag = 1
        self.TurnLeft_flag = 1

        self.img_set_flag = 0
        self.img_time_flag = 0

        self.setWindowTitle("Experiment")
        self.setGeometry(300,100,400,300)

        input_area_coord_x = 1300
        input_area_coord_y = 100
        x_space = 200
        y_space = 30

        self.lbl=QLabel("图片", self)

        self.pm = QPixmap( self.call_current_img_path() )
        # self.pm = self.pm.scaled(950, 533)
        self.pm = self.pm.scaled(1280, 720)
        self.lbl.setPixmap(self.pm)
        # self.lbl.resize(300,200)
        self.lbl.setScaledContents(True)



        self.img_serial_label = QLabel(self)
        self.img_serial_label.setText("Image serial number:" + "  " + str(self.img_set_flag + 1)+ " " * 20)
        self.img_serial_label.move(0, 720 + 2 * y_space)
        self.img_serial_label.setTextInteractionFlags(Qt.TextSelectableByMouse)



        #########################################################################################################
        #########################################################################################################

        self.time_step_label = QLabel(self)
        self.time_step_label.setText("Showing time step 1")
        self.time_step_label.move(input_area_coord_x + 2 * x_space, input_area_coord_y - 2 * y_space)

        self.forward_label = QLabel(self)
        self.forward_label.setText(str(self.Forward_flag == 1) + " ")
        self.forward_label.move(input_area_coord_x + 2 * x_space, input_area_coord_y + 2 * y_space)

        self.Turn_left_label = QLabel(self)
        self.Turn_left_label.setText(str(self.TurnLeft_flag == 1) + " ")
        self.Turn_left_label.move(input_area_coord_x + x_space, input_area_coord_y + 4 * y_space)

        self.Turn_right_label = QLabel(self)
        self.Turn_right_label.setText(str(self.TurnRight_flag == 1) + " ")
        self.Turn_right_label.move(input_area_coord_x + 2 * x_space + x_space / 2, input_area_coord_y + 4 * y_space)

        #########################################################################################################
        #########################################################################################################


        btn_time_step_1 = QPushButton("Show time step 1 image", self)
        btn_time_step_1.clicked.connect(self.show_time_step_1)
        btn_time_step_1.move(input_area_coord_x, input_area_coord_y - 2 * y_space)

        btn_time_step_2 = QPushButton("Show time step 2 image", self)
        btn_time_step_2.clicked.connect(self.show_time_step_2)
        btn_time_step_2.move(input_area_coord_x + x_space, input_area_coord_y - 2 * y_space)

        #########################################################################################################


        btn_forward = QPushButton("Forward", self)
        btn_forward.clicked.connect(self.Forward)
        btn_forward.move(input_area_coord_x + x_space, input_area_coord_y + 2 * y_space)

        btnNoforward = QPushButton("No Forward", self)
        btnNoforward.clicked.connect(self.NoForward)
        btnNoforward.move(input_area_coord_x + 3 * x_space / 2, input_area_coord_y + 2 * y_space)
        #########################################################################################################
        btnTurnLeft=QPushButton("Turn Left", self)
        btnTurnLeft.clicked.connect(self.TurnLeft)
        btnTurnLeft.move(input_area_coord_x, input_area_coord_y + 4 * y_space)

        btnNoTurnLeft = QPushButton("No Turn Left", self)
        btnNoTurnLeft.clicked.connect(self.NoTurnLeft)
        btnNoTurnLeft.move(input_area_coord_x + x_space / 2, input_area_coord_y + 4 * y_space)
        #########################################################################################################
        btnTurnRight = QPushButton("Turn right", self)
        btnTurnRight.clicked.connect(self.TurnRight)
        btnTurnRight.move(input_area_coord_x + x_space +  x_space / 2, input_area_coord_y + 4 * y_space)

        btnNoTurnRight = QPushButton("No Turn right", self)
        btnNoTurnRight.clicked.connect(self.NoTurnRight)
        btnNoTurnRight.move(input_area_coord_x + 2 * x_space, input_area_coord_y + 4 * y_space)

        #########################################################################################################

        btnPause = QPushButton("Refreshing, Timer starts from Now", self)
        btnPause.clicked.connect(self.Pause)
        btnPause.move(input_area_coord_x, input_area_coord_y + 8 * y_space)


        btnNext=QPushButton("Next", self)
        btnNext.clicked.connect(self.myAddPic)
        btnNext.move(input_area_coord_x + 2 * x_space, input_area_coord_y + 8 * y_space)
        self.show()
    def call_current_img_path(self):
        self.current_img_path = self.img_set_path_list[self.img_set_flag][self.img_time_flag]
        assert os.path.exists(self.current_img_path), (self.current_img_path + " not exists")

        return self.current_img_path
    def show_time_step_1(self):
        self.img_time_flag = 0

        self.pm = QPixmap(self.call_current_img_path() )
        self.lbl.setPixmap(self.pm)
        self.time_step_label.setText("Showing time step 1")
    def show_time_step_2(self):
        self.img_time_flag = 1
        self.pm = QPixmap(self.call_current_img_path() )
        self.lbl.setPixmap(self.pm)
        self.time_step_label.setText("Showing time step 2")
    def AllTrue(self):
        self.Forward_flag = 1
        self.TurnLeft_flag = 1
        self.TurnRight_flag = 1
        self.forward_label.setText(str(self.Forward_flag == 1))
        self.Turn_left_label.setText(str(self.TurnLeft_flag == 1))
        self.Turn_right_label.setText(str(self.TurnRight_flag == 1))
    def Forward(self):
        self.Forward_flag = 1
        self.forward_label.setText(str(self.Forward_flag == 1))
    def NoForward(self):
        self.Forward_flag = 0
        self.forward_label.setText(str(self.Forward_flag == 1))

    def TurnLeft(self):
        self.TurnLeft_flag = 1
        self.Turn_left_label.setText(str(self.TurnLeft_flag == 1))
    def NoTurnLeft(self):
        self.TurnLeft_flag = 0
        self.Turn_left_label.setText(str(self.TurnLeft_flag == 1))

    def TurnRight(self):
        self.TurnRight_flag = 1
        self.Turn_right_label.setText(str(self.TurnRight_flag == 1))
    def NoTurnRight(self):
        self.TurnRight_flag = 0
        self.Turn_right_label.setText(str(self.TurnRight_flag == 1))

    def Pause(self):
        self.time_bank = time.time()

    def save_info(self, spent_time):

        input_content = []
        img_name = self.img_set_path_list[self.img_set_flag][self.img_time_flag].split("\\")[-1][:-6]
        input_content.append(img_name)
        input_content.append(self.Forward_flag)
        input_content.append(self.TurnLeft_flag)
        input_content.append(self.TurnRight_flag)
        input_content.append(spent_time)

        write_a_csv_file(self.csv_log_path, input_content)

    def myAddPic(self):
        self.time_step_label.setText("Showing time step 1")
        current_time = time.time()
        spent_time = current_time - self.time_bank
        self.time_bank = current_time

        if self.img_set_flag >= len(self.img_set_path_list) - 1:
            self.save_info(spent_time)
            print("The labelling is over, Thank you!")
            exit()
        else:

            self.img_time_flag = 0
            self.save_info(spent_time)
            self.img_set_flag = self.img_set_flag + 1


        self.pm = QPixmap(self.call_current_img_path())
        self.lbl.setPixmap(self.pm)

        image_name = self.img_set_path_list[self.img_set_flag][self.img_time_flag].split("\\")[-1]
        image_gradual_number = image_name.split("_")[2]

        if image_gradual_number == "1":
            self.AllTrue()
        # self.img_path_label.setText("Image name:" + "  " + self.img_set_path_list[self.img_set_flag][self.img_time_flag])

        self.img_serial_label.setText("Image serial number:" + "  " + str(self.img_set_flag + 1) + " " + "/" + " " + str(int(self.unfinished_imgs_path_list_length)))


        

def write_a_csv_file(path, input_content):
    with open(path, 'a', newline="", encoding = "utf-8") as f:
        writer = csv.writer(f, delimiter = ",")
        writer.writerow(input_content)
def read_csv_file(file_path):
    saved_image_name_list = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)

        for row in csv_reader:
            saved_image_name_list.append(row[0])
    return saved_image_name_list


def preparation_before_job(imgs_path):  # check_unlabeled_data
    files = os.listdir(imgs_path)
    tracking_file_name_list = []
    folder_path_list = []
    csv_path_list = []
    for file_name in files:

        file_path = os.path.join(imgs_path, file_name)
        if os.path.isdir(file_path):
            folder_path_list.append(file_path)
        elif file_name[-4] == ".csv":
            csv_path_list.append(file_path)
    finish_flag = True
    for each_folder_path in folder_path_list:
        folder_path, file_name_list, finish_flag = check_every_folder_img_label(each_folder_path)
        if finish_flag == False:
            break

    return folder_path, file_name_list, finish_flag

def check_every_folder_img_label(folder_path):
    files = os.listdir(folder_path)

    image_name_list = []
    image_name_set = set()

    for file_name in files:
        image_name = file_name.split("_")[0] +  "_" + file_name.split("_")[1]

        image_name_set.add(image_name)


    image_name_list = list(image_name_set)
    image_gradual_name_list = []
    for image_name in image_name_list:
        for i in range(1, gradual_step_num + 1):
            image_gradual_name = image_name + "_" + str(i)
            image_gradual_name_list.append(image_gradual_name)



    if os.path.exists(folder_path + ".csv"):
        saved_image_name_list = read_csv_file(folder_path + ".csv")
    else:
        finish_flag = False
        return folder_path, image_gradual_name_list,  finish_flag

    if len(saved_image_name_list) == len(image_gradual_name_list):
        print(folder_path, "is done!")
        finish_flag = True
        return folder_path, image_gradual_name_list,  finish_flag
    else:
        print(folder_path, "has not been done!")
        finish_flag = False
        left_over_image_gradual_name_list = []
        for image_gradual_name in image_gradual_name_list:
            if image_gradual_name not in saved_image_name_list:
                left_over_image_gradual_name_list.append(image_gradual_name)

        # exit()
        assert len(left_over_image_gradual_name_list) == len(image_gradual_name_list) - len(saved_image_name_list), ("len(left_over_image_gradual_name_list) == len(image_gradual_name_list) - len(saved_image_name_list)")
        return folder_path, left_over_image_gradual_name_list,  finish_flag
if __name__=="__main__":
    base_img_path = os.path.abspath('.')
    # folders_folder = "./simulation/Object-DP"
    folders_folder = "./mixed_simulation/"
    folders_folder_path = os.path.join(base_img_path, folders_folder)

    unfinish_folder_path, file_name_list, finish_flag = preparation_before_job(folders_folder_path)
    if finish_flag == True:
        print("All the imgs in the imgs_folder are done! Thank you!")
        input("please input any key to exit!")
        exit()

    unfinished_imgs_path_list = []


    for file_name in file_name_list:
        img_category_name = file_name.split("_", 1)[0]
        # if img_category_name != "LR":
        #     continue
        unfinished_imgs_time_path_list = []
        for time_step in range(1, time_step_num + 1):
            img_time_name = os.path.join(unfinish_folder_path, file_name + "_" + str(time_step) + ".jpg")
            img_path = os.path.join(img_time_name)
            unfinished_imgs_time_path_list.append(img_path)
        unfinished_imgs_path_list.append(unfinished_imgs_time_path_list)


    csv_log_path = unfinish_folder_path + ".csv"
    # print(unfinished_imgs_path_list)
    app=QApplication(sys.argv)
    mc=GUIclass(unfinished_imgs_path_list, csv_log_path)
    app.exec_()
