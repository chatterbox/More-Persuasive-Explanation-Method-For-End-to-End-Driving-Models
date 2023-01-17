#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import time
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import f1_score


class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss_vali = 0
        self.epoch_loss_train = 0
        
        self.epoch_f1score_train = 0
        self.epoch_f1score_vali = 0

        self.epoch_train_prediction_list = np.array([])
        self.epoch_vali_prediction_list = np.array([])
        self.epoch_train_label_list = np.array([])
        self.epoch_vali_label_list = np.array([])

        self.epoch_num_correct_train = 0
        self.epoch_num_correct_vali = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = [] # keep track of parameter values and the result of each epoch
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
        
        

    def begin_run(self, run,network):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
    

        
    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_train_prediction_list = np.array([])
        self.epoch_vali_prediction_list = np.array([])
        self.epoch_train_label_list = np.array([])
        self.epoch_vali_label_list = np.array([])
        
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss_vali = 0
        self.epoch_loss_train = 0
        self.f1_score = 0
        self.epoch_num_correct = 0
        self.epoch_f1score_train = 0
        self.epoch_f1score_vali = 0
        self.epoch_num_correct_train = 0
        self.epoch_num_correct_vali = 0
    def end_epoch(self, train_iteration_num, vali_iteration_num, file_name):
        
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time()-self.run_start_time
        
        train_loss = self.epoch_loss_train / train_iteration_num
        vali_loss = self.epoch_loss_vali / vali_iteration_num

        train_accuracy = self.epoch_num_correct_train / train_iteration_num
        vali_accuracy = self.epoch_num_correct_vali/ vali_iteration_num
        
        train_f1score = f1_score(self.epoch_train_prediction_list, self.epoch_train_label_list, average="macro")

        vali_f1score = f1_score(self.epoch_vali_prediction_list, self.epoch_vali_label_list, average="macro")

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["train_loss"] = train_loss
        results["vali_loss"] = vali_loss
        results["train_f1score"] = train_f1score
        results["vali_f1score"] = vali_f1score
        results["train_accuracy"] = train_accuracy
        results["vali_accuracy"] = vali_accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        
        for k,v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        self.save(file_name)
        return train_f1score, vali_f1score
        
    def track_loss(self,loss, data_type):
        if data_type == "train":
            self.epoch_loss_train += loss.item()
        else:
            self.epoch_loss_vali +=loss.item()
        
        
        
    def track_num_correct_and_f1score(self,preds, labels, data_type):
        if data_type == "train":
            self.epoch_num_correct_train += self._get_num_correct(preds,labels)

            assert len(self.epoch_train_prediction_list) == len(self.epoch_train_label_list), ("len(self.epoch_prediction_list) != len(self.epoch_label_list)")
            if len(self.epoch_train_prediction_list) == 0:
                self.epoch_train_prediction_list = preds
                self.epoch_train_label_list = labels
            else:
                self.epoch_train_prediction_list = np.concatenate([self.epoch_train_prediction_list, preds], axis = 0)
                self.epoch_train_label_list = np.concatenate([self.epoch_train_label_list, labels], axis = 0)


        else:
            self.epoch_num_correct_vali += self._get_num_correct(preds,labels)

            assert len(self.epoch_vali_prediction_list) == len(self.epoch_vali_label_list), ("len(self.epoch_prediction_list) != len(self.epoch_label_list)")
            if len(self.epoch_vali_prediction_list) == 0:
                self.epoch_vali_prediction_list = preds
                self.epoch_vali_label_list = labels
            else:
                self.epoch_vali_prediction_list = np.concatenate([self.epoch_vali_prediction_list, preds], axis = 0)
                self.epoch_vali_label_list = np.concatenate([self.epoch_vali_label_list, labels], axis = 0)
        
        
    @torch.no_grad()
    def _get_num_correct(self,preds,labels):
        preds_length = preds.shape[0]
        labels_length = labels.shape[0]
        assert preds_length == labels_length, "preds_length != labels_length"
        correct_num_iter = 0
        for i in range(preds_length):
            if f1_score(preds[i], labels[i]) == 1:
                correct_num_iter = correct_num_iter + 1
        return correct_num_iter
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient = 'columns'
            
        ).to_csv(f'{fileName}')
        
        # with open(f'{fileName}.json','w',encoading = 'utf-8') as f:
        #     json.dump(self.run_data, f, ensure_ascii = False, indent =4)