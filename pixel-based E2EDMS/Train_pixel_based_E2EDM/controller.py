import Runner
from collections import OrderedDict
from model import Convolution3D_LSTM_transfer as CNN3D_LSTM
from model import Convolution2D_LSTM_transfer as CNN2D_LSTM
from model import Convolution3D_only_transfer as CNN3D
import pandas as pd
import os
import torch


def same_result_init():
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class model_runner():
    def __init__(self, model_name, gpu_num):
        self.cross_validation_counter = 0
        self.gpu_num = gpu_num
        self.model_name = model_name
        self.network = ""

        self.parameters = OrderedDict(model_name = model_name,
        epoch_num = [1],
        learning_rate = [0.001],
        batch_size = [64],
        seq_len = [2],
        num_workers = [0])

        self.cross_validation_counter = 0
        self.data_recoder = []

        
    def run_model(self, cross_validation_counter, model_name):
        self.cross_validation_counter = cross_validation_counter
        if model_name == ["CNN2D_LSTM"]:
            self.network = CNN2D_LSTM.MyEnsemble() 
        elif model_name == ["CNN3D"]:
            self.network = CNN3D.MyEnsemble() 
        elif model_name == ["CNN3D_LSTM"]:
            self.network = CNN3D_LSTM.MyEnsemble() 
        else:
            print("The model is unknow!")
            exit()
        print(self.parameters)
        trained_model_folder_path = "cross_validation_saver/{}/cross_validation-{}/".format(model_name[0], str(self.cross_validation_counter))
        
        os.makedirs(trained_model_folder_path, exist_ok=True)
        self.train_f1_score_for_best_vali, self.max_validation_f1_score, self.test_f1_score, self.best_epoch_serial_num = Runner.run_deep_learning(self.network, self.parameters, self.gpu_num, cross_validation_counter)
        train_and_vali_f1score = (self.train_f1_score_for_best_vali, self.max_validation_f1_score)
        self.data_saver()
        source_model_save_folder_path = "cross_validation_saver/{}".format(model_name[0])
        target_model_save_folder_path = "/home0/zhangc/code/cross_validation_E2E/3DCNN-LSTM/cross_validation_saver_for_explanation/{}".format(model_name[0])
        return self.test_f1_score, self.best_epoch_serial_num, train_and_vali_f1score, model_name[0], source_model_save_folder_path, target_model_save_folder_path

    def data_saver(self):
        results = OrderedDict()
        
        results["model_name"] = self.model_name[0]
        results["train_f1_score"] = self.train_f1_score_for_best_vali
        results["validation_f1_score"] = self.max_validation_f1_score
        results["test_f1_score"] = self.test_f1_score
        results["cross_validation_counter"] = self.cross_validation_counter
        results["Best_network_epoch_num "] = self.best_epoch_serial_num
        self.data_recoder.append(results)

        pd.DataFrame.from_dict(
                self.data_recoder,
                orient = 'columns'
            ).to_csv(f'{self.model_name[0]}.csv')
        self.cross_validation_counter = self.cross_validation_counter + 1


if __name__== '__main__':
    
    # same_result_init()
    
    model_name_CNN2D_LSTM = ["CNN2D_LSTM"]
    model_name_CNN3D = ["CNN3D"]
    model_name_CNN3D_LSTM = ["CNN3D_LSTM"]

    test_f1_score_list = []
    CNN3D_LSTM_instance = model_runner(model_name_CNN3D_LSTM, 0)
    CNN3D_instance = model_runner(model_name_CNN3D, 1)
    CNN2D_LSTM_instance = model_runner(model_name_CNN2D_LSTM, 1)    
    
    for cross_validation_index in range(5):
        test_score, best_epoch_serial_num, train_and_vali_f1score, model_name, source_model_save_folder_path, target_model_save_folder_path = CNN2D_LSTM_instance.run_model(cross_validation_index, model_name_CNN2D_LSTM)
        test_score, best_epoch_serial_num, train_and_vali_f1score, model_name, source_model_save_folder_path, target_model_save_folder_path = CNN3D_instance.run_model(cross_validation_index, model_name_CNN3D)
        test_score, best_epoch_serial_num, train_and_vali_f1score, model_name, source_model_save_folder_path, target_model_save_folder_path = CNN3D_LSTM_instance.run_model(cross_validation_index, model_name_CNN3D_LSTM)