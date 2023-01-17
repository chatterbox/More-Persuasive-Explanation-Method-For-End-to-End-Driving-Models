#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import copy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import torch.nn as nn 
import torch.optim as optim
import os
import pandas as pd
from Run import RunBuilder as RB
from Run import RunManager as RM
from DataLoading import UdacityDataset as UD
from sklearn.metrics import f1_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def run_deep_learning(network, parameters, GPU_num, cross_validation_counter):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_num)

    if torch.cuda.is_available() == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    m = RM.RunManager()


    pesudo_img_csv_path = "/home0/zhangc/Dataset/BDD_100K_FLR/pesudo_img/zhang_pesudo_img.csv"
    pesudo_img_csv = pd.read_csv(pesudo_img_csv_path)
    pesudo_img_name_list = []
    for idx in range(len(pesudo_img_csv)):
        pesudo_img_name = pesudo_img_csv['img_name'].iloc[idx]
        pesudo_img_name_list.append(pesudo_img_name)


    csv_folder_path = '/home0/zhangc/Dataset/BDD_100K_FLR/cross_validation_500/'
    each_csv_folder_path_list = []
    for i in range(5):
        each_csv_folder_path = os.path.join(csv_folder_path, "regularized_csv_folder_500_" + str(i))
        each_csv_folder_path_list.append(each_csv_folder_path)
    
    train_csv_folder_path = []
    test_csv_folder_path = [each_csv_folder_path_list[cross_validation_counter]]
    if cross_validation_counter >= len(each_csv_folder_path_list)-1:
        vali_counter_num = 0
    else:
        vali_counter_num = cross_validation_counter + 1
    
    vali_csv_folder_path = [each_csv_folder_path_list[vali_counter_num]]
    for list_index in range(len(each_csv_folder_path_list)):
        if list_index != vali_counter_num and list_index != cross_validation_counter:
            train_csv_folder_path.append(each_csv_folder_path_list[list_index])
            
    img_folder_dir_path = '/home0/zhangc/Dataset/BDD_100K_FLR/imgs_folder_500'

    assert os.path.exists(csv_folder_path) and os.path.exists(img_folder_dir_path), "Dataset do not exists!"
    
    for run in RB.RunBuilder.get_runs(parameters):
        
        network = network.to(device)

        optimizer = optim.Adam(network.parameters(),lr = run.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
        # optimizer = optim.RAdam(network.parameters(),lr = run.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

        training_loader = cross_validation_dataset_maker(train_csv_folder_path, img_folder_dir_path, run)
        validation_loader = cross_validation_dataset_maker(vali_csv_folder_path, img_folder_dir_path, run)
        test_loader = cross_validation_dataset_maker(test_csv_folder_path, img_folder_dir_path, run, True)

        m.begin_run(run,network)
        class_weights = [1, 1, 1]

        w = torch.FloatTensor(class_weights).cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
        max_validation_f1_score = 0
        
        for epoch in range(run.epoch_num):
            epoch_test_prediction_list = np.array([])
            epoch_test_label_list = np.array([])

            train_iteration_num = 0
            vali_iteration_num = 0
            m.begin_epoch()

            network.train()
            loss_list = []
            
            for training_sample in tqdm(training_loader):
                
                train_iteration_num = train_iteration_num + 1
                
                training_sample['image'] = training_sample['image'].permute(0,2,1,3,4)
                
                param_values = [v for v in training_sample.values()]

                
                image = param_values[0]
                angle = param_values[1]
                image = image.to(device)   
                
                prediction = network(image)

                labels = angle.to(device)
                labels = labels[:,-1,:,]

                del param_values, image, angle
                if labels.shape[0]!=prediction.shape[0]:
                    prediction = prediction[-labels.shape[0],:]
        
                labels = labels.float()

                
                loss = criterion(prediction, labels)
                loss_list.append(loss)
                
                predict_action = torch.sigmoid(prediction) > 0.5

                labels = labels.cpu().data.numpy()
                predict_action = predict_action.cpu().data.numpy()

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                
                m.track_loss(loss, "train")
                m.track_num_correct_and_f1score(predict_action,labels, "train")

                if len(epoch_test_prediction_list) == 0:
                    epoch_test_prediction_list = predict_action
                    epoch_test_label_list = labels
                else:
                    epoch_test_prediction_list = np.concatenate([epoch_test_prediction_list, predict_action], axis = 0)
                    epoch_test_label_list = np.concatenate([epoch_test_label_list, labels], axis = 0)
            

            epoch_test_prediction_list = np.array([])
            epoch_test_label_list = np.array([])
    # Calculation on Validation Loss
            loss_list = []
            with torch.no_grad():    
                network.eval()
                # for Validation_sample in tqdm(validation_loader):
                for Validation_sample in (validation_loader):
                    vali_iteration_num = vali_iteration_num + 1
                    
                    Validation_sample['image'] = Validation_sample['image'].permute(0,2,1,3,4)

                    param_values = [v for v in Validation_sample.values()]
                    image = param_values[0]
                    angle = param_values[1]
                    image = image.to(device)
                    prediction = network(image)
                    # print("validation", Validation_sample["image_name"][0][0], prediction[0])
                    labels = angle.to(device)
                    labels = labels[:,-1,:,]
                    del param_values, image, angle
                    
                    labels = labels.float()
                    loss = criterion(prediction, labels)
                    loss_list.append(loss)
                    predict_action = torch.sigmoid(prediction) > 0.5
                    
                    labels = labels.cpu().data.numpy()
                    predict_action = predict_action.cpu().data.numpy()
                    
                    m.track_loss(loss, "validation")
                    m.track_num_correct_and_f1score(predict_action,labels, "validation")

                    if len(epoch_test_prediction_list) == 0:
                        epoch_test_prediction_list = predict_action
                        epoch_test_label_list = labels
                    else:
                        epoch_test_prediction_list = np.concatenate([epoch_test_prediction_list, predict_action], axis = 0)
                        epoch_test_label_list = np.concatenate([epoch_test_label_list, labels], axis = 0)
            
            log_csv_file_path = "cross_validation_saver/{}/cross_validation-{}/{}.csv".format(run.model_name, str(cross_validation_counter), run.model_name + "-" +str(cross_validation_counter))
            train_f1score, vali_f1score = m.end_epoch(train_iteration_num,vali_iteration_num, log_csv_file_path)
            
            if vali_f1score > max_validation_f1_score:
                max_validation_f1_score = vali_f1score
                train_f1_score_for_best_vali = train_f1score
                best_network = copy.deepcopy(network)
                
                best_epoch_serial_num = epoch
            torch.save(network.state_dict(), "cross_validation_saver/{}/cross_validation-{}/{}.pt".format(run.model_name, str(cross_validation_counter), str(epoch)))
                
        m.end_run()
        test_f1_score, _ = test_dataset_loss_and_f1score_calculator(test_loader, best_network, device, criterion, "second_step")
        
        return train_f1_score_for_best_vali, max_validation_f1_score, test_f1_score, best_epoch_serial_num

def test_dataset_loss_and_f1score_calculator(test_loader, best_network, device, criterion, position_comment):
    loss_list = []

    epoch_test_prediction_list = np.array([])
    epoch_test_label_list = np.array([])
    
    with torch.no_grad():    
        best_network.eval()
        
        for Validation_sample in (test_loader):
        # for Validation_sample in tqdm(test_loader):
            Validation_sample['image'] = Validation_sample['image'].permute(0,2,1,3,4)
            
            param_values = [v for v in Validation_sample.values()]
            image = param_values[0]
            angle = param_values[1]
            image = image.to(device)
            prediction = best_network(image)
            
            labels = angle.to(device)
            labels = labels[:,-1,:,]
            del param_values, image, angle
            labels = labels.float()
            loss = criterion(prediction, labels)
            loss_list.append(loss)
            # print(Validation_sample["image_name"][0][0], prediction[0])
            
            predict_action = torch.sigmoid(prediction) > 0.5
            labels = labels.cpu().data.numpy()
            predict_action = predict_action.cpu().data.numpy()


            if len(epoch_test_prediction_list) == 0:
                epoch_test_prediction_list = predict_action
                epoch_test_label_list = labels
            else:
                epoch_test_prediction_list = np.concatenate([epoch_test_prediction_list, predict_action], axis = 0)
                epoch_test_label_list = np.concatenate([epoch_test_label_list, labels], axis = 0)
        assert len(epoch_test_prediction_list) == len(epoch_test_label_list), ("len(epoch_test_prediction_list) != len(epoch_test_label_list)")
        
        test_f1score = f1_score(epoch_test_prediction_list, epoch_test_label_list, average="macro")

    return test_f1score, sum(loss_list)/len(loss_list)
    

def cross_validation_dataset_maker(csv_folder_path, img_folder_dir_path, run, test_loader_flag = False):
    csv_file_path_list = []
    for each_csv_folder in csv_folder_path:
        for i in os.listdir(each_csv_folder):
            csv_file_path = os.path.join(each_csv_folder, i)
            csv_file_path_list.append(csv_file_path)
    
    full_set = UD.UdacityDataset(csv_file_path_list, img_folder_dir_path, test_loader_flag, seq_len = run.seq_len, shuffle = False)
    
    training_loader = DataLoader(full_set, batch_size = run.batch_size, num_workers=run.num_workers)
    
    return training_loader

    
    
    
    
    
    

