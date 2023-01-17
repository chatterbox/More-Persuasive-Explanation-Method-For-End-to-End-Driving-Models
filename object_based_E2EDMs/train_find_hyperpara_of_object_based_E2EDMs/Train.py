# coding: UTF-8
from sklearn.metrics import f1_score
import numpy as np
import os
import pickle
from config import Config
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from tqdm import tqdm
import K_means_zhang
import warnings
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble
warnings.filterwarnings("ignore")
config = Config()


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
            self.train_dataset_part.append(cross_each_cross_validation_iter_list[index])

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



def find_the_sepearte_model_structure_hyperparameter_MLP(cross_each_cross_validation_iter_list):
    custom_corss_validation_instance = custom_corss_validation(cross_each_cross_validation_iter_list)

    max_f1_score = 0

    parameters = {
        'i': [90],
        # 'solver': ['lbfgs', 'liblinear'],
        'j': [190],
        "k": [90],
        # "random_state": [None, 1, 982],
        "random_state": [740],

    }
    save_model_target_path = "cross_validation_saved_model"
    for i in parameters["i"]:
        for j in parameters["j"]:
            for k in parameters["k"]:
                for random_state_hp in tqdm(parameters["random_state"]):
                    cross_validation_score = 0

                    for index in range(5):
                        train_dataset_part, validation_dataset_part, test_dataset_part = custom_corss_validation_instance.get_dataset_part_fold_by_serial_num(
                            index)
                        train_data_input, train_data_ouput = read_info_outof_cross_validation_dataset(
                            train_dataset_part)
                        test_data_input, test_data_ouput = read_info_outof_cross_validation_dataset(
                            test_dataset_part)


                        train_data_ouput = np.array(train_data_ouput)
                        test_data_ouput = np.array(test_data_ouput)
                        train_data_input = np.array(train_data_input)
                        test_data_input = np.array(test_data_input)

                        train_forward_data = train_data_ouput[:, 0]
                        train_left_data = train_data_ouput[:, 1]
                        train_right_data = train_data_ouput[:, 2]

                        test_forward_data = test_data_ouput[:, 0]
                        test_left_data = test_data_ouput[:, 1]
                        test_right_data = test_data_ouput[:, 2]

                        RF_forward = MLPClassifier(hidden_layer_sizes=(i, j, k),  random_state = random_state_hp, batch_size = 10).fit(
                            train_data_input,
                            train_forward_data)

                        RF_left = MLPClassifier(hidden_layer_sizes=(i, j, k),  random_state = random_state_hp, batch_size = 10).fit(
                            train_data_input,
                            train_left_data)
                        RF_right = MLPClassifier(hidden_layer_sizes=(i, j, k),  random_state = random_state_hp, batch_size = 10).fit(
                            train_data_input,
                            train_right_data)

                        # The following stuff is used to save the desirable model
                        # target_model_list = [RF_forward, RF_left, RF_right]
                        # model_name_list = ["Forward", "Left", "Right"]
                        # for i in range(len(target_model_list)):
                        #     model_name = model_name_list[i]
                        #     trained_model_name = os.path.join(save_model_target_path,
                        #                                       model_name + "_" + str(
                        #                                           index) + ".pkl")
                        #     with open(trained_model_name, 'wb') as f:
                        #         pickle.dump(target_model_list[i], f)


                        pred_forward = RF_forward.predict(test_data_input)
                        pred_left = RF_left.predict(test_data_input)
                        pred_right = RF_right.predict(test_data_input)

                        score_forward = sklearn.metrics.f1_score(test_forward_data, pred_forward, average='binary')
                        score1_left = sklearn.metrics.f1_score(test_left_data, pred_left, average='binary')
                        score1_right = sklearn.metrics.f1_score(test_right_data, pred_right, average='binary')

                        all_score1 = (score_forward + score1_left + score1_right) / 3
                        print(all_score1)
                        cross_validation_score = cross_validation_score + all_score1
                    if cross_validation_score > max_f1_score:
                        max_f1_score = cross_validation_score
                        best_estimator = (RF_forward, RF_left, RF_right)
    print(cross_validation_score/5)

def find_the_sepearte_model_structure_hyperparameter_LR(cross_each_cross_validation_iter_list):
    custom_corss_validation_instance = custom_corss_validation(cross_each_cross_validation_iter_list)


    parameters = {
        'C': [0.01, 0.1, 1.0],
        # 'solver': ['lbfgs', 'liblinear'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
        "max_iter": [100, 500, 1000],
        "random_state": [None],
    }


    save_model_target_path = "cross_validation_saved_model"
    for i in parameters["C"]:
        for j in parameters["solver"]:
            for k in parameters["max_iter"]:
                for random_state_hp in tqdm(parameters["random_state"]):
                    cross_validation_score = 0

                    for index in range(5):
                        train_dataset_part, validation_dataset_part, test_dataset_part = custom_corss_validation_instance.get_dataset_part_fold_by_serial_num(
                            index)
                        train_data_input, train_data_ouput = read_info_outof_cross_validation_dataset(
                            train_dataset_part)
                        test_data_input, test_data_ouput = read_info_outof_cross_validation_dataset(
                            test_dataset_part)


                        train_data_ouput = np.array(train_data_ouput)
                        test_data_ouput = np.array(test_data_ouput)
                        train_data_input = np.array(train_data_input)
                        test_data_input = np.array(test_data_input)

                        train_forward_data = train_data_ouput[:, 0]
                        train_left_data = train_data_ouput[:, 1]
                        train_right_data = train_data_ouput[:, 2]

                        test_forward_data = test_data_ouput[:, 0]
                        test_left_data = test_data_ouput[:, 1]
                        test_right_data = test_data_ouput[:, 2]

                        RF_forward = LogisticRegression(C= i, solver=j, max_iter = k, random_state = random_state_hp).fit(
                            train_data_input,
                            train_forward_data)

                        RF_left = LogisticRegression(C= i, solver=j, max_iter = k, random_state = random_state_hp).fit(
                            train_data_input,
                            train_left_data)
                        RF_right = LogisticRegression(C= i, solver=j, max_iter = k, random_state = random_state_hp).fit(
                            train_data_input,
                            train_right_data)


                        # target_model_list = [RF_forward, RF_left, RF_right]
                        # model_name_list = ["Forward", "Left", "Right"]
                        # for i in range(len(target_model_list)):
                        #     model_name = model_name_list[i]
                        #     trained_model_name = os.path.join(save_model_target_path,
                        #                                       model_name + "_" + str(
                        #                                           index) + ".pkl")
                        #     with open(trained_model_name, 'wb') as f:
                        #         pickle.dump(target_model_list[i], f)
                        #
                        # continue

                        pred_forward = RF_forward.predict(test_data_input)
                        pred_left = RF_left.predict(test_data_input)
                        pred_right = RF_right.predict(test_data_input)

                        score_forward = sklearn.metrics.f1_score(test_forward_data, pred_forward, average='binary')
                        score1_left = sklearn.metrics.f1_score(test_left_data, pred_left, average='binary')
                        score1_right = sklearn.metrics.f1_score(test_right_data, pred_right, average='binary')

                        all_score1 = (score_forward + score1_left + score1_right) / 3
                        cross_validation_score = cross_validation_score + all_score1

    print(cross_validation_score / 5)


def find_the_sepearte_model_structure_hyperparameter_RF(cross_each_cross_validation_iter_list):
    custom_corss_validation_instance = custom_corss_validation(cross_each_cross_validation_iter_list)

    max_f1_score = 0

    parameters = {
        'min_samples_leaf': list(range(1, 10)),
        'n_estimators': list(range(1, 50)),
        "max_depth": list(range(2, 30)),
        "random_state": [None],
    }


    save_model_target_path = "cross_validation_saved_model"
    for i in tqdm(parameters["min_samples_leaf"]):
        for j in parameters["n_estimators"]:
            for k in parameters["max_depth"]:
                for random_state_hp in (parameters["random_state"]):
                    cross_validation_score = 0

                    for index in range(5):
                        train_dataset_part, validation_dataset_part, test_dataset_part = custom_corss_validation_instance.get_dataset_part_fold_by_serial_num(
                            index)
                        train_data_input, train_data_ouput = read_info_outof_cross_validation_dataset(
                            train_dataset_part)
                        test_data_input, test_data_ouput = read_info_outof_cross_validation_dataset(
                            test_dataset_part)


                        train_data_ouput = np.array(train_data_ouput)
                        test_data_ouput = np.array(test_data_ouput)
                        train_data_input = np.array(train_data_input)
                        test_data_input = np.array(test_data_input)

                        train_forward_data = train_data_ouput[:, 0]
                        train_left_data = train_data_ouput[:, 1]
                        train_right_data = train_data_ouput[:, 2]

                        test_forward_data = test_data_ouput[:, 0]
                        test_left_data = test_data_ouput[:, 1]
                        test_right_data = test_data_ouput[:, 2]

                        RF_forward = sklearn.ensemble.RandomForestClassifier(max_depth=k,min_samples_leaf=i, n_estimators=j, random_state=random_state_hp).fit(
                            train_data_input,
                            train_forward_data)

                        RF_left = sklearn.ensemble.RandomForestClassifier(max_depth=k,min_samples_leaf=i, n_estimators=j, random_state=random_state_hp).fit(
                            train_data_input,
                            train_left_data)

                        RF_right = sklearn.ensemble.RandomForestClassifier(max_depth=k,min_samples_leaf=i, n_estimators=j, random_state=random_state_hp).fit(
                            train_data_input,
                            train_right_data)


                        # target_model_list = [RF_forward, RF_left, RF_right]
                        # model_name_list = ["Forward", "Left", "Right"]
                        # for i in range(len(target_model_list)):
                        #     model_name = model_name_list[i]
                        #     trained_model_name = os.path.join(save_model_target_path,
                        #                                       model_name + "_" + str(
                        #                                           index) + ".pkl")
                        #     with open(trained_model_name, 'wb') as f:
                        #         pickle.dump(target_model_list[i], f)
                        #
                        # continue

                        pred_forward = RF_forward.predict(test_data_input)
                        pred_left = RF_left.predict(test_data_input)
                        pred_right = RF_right.predict(test_data_input)

                        score_forward = sklearn.metrics.f1_score(test_forward_data, pred_forward, average='binary')
                        score1_left = sklearn.metrics.f1_score(test_left_data, pred_left, average='binary')
                        score1_right = sklearn.metrics.f1_score(test_right_data, pred_right, average='binary')

                        all_score1 = (score_forward + score1_left + score1_right) / 3
                        # print(all_score1)
                        cross_validation_score = cross_validation_score + all_score1
                    if cross_validation_score > max_f1_score:
                        max_f1_score = cross_validation_score
                        best_estimator = (RF_forward, RF_left, RF_right)
    print(cross_validation_score/5)

if __name__ == '__main__':

    # use k_means to turn the object information to 168-length feature dataset
    cross_each_cross_validation_iter_list, all_data_iter, all_object_codebook = K_means_zhang.K_means_zhang_train_iter()

    # for each object-based E2EDMs(MLP,LR..), we kind grid search the optimal parameters
    # find_the_sepearte_model_structure_hyperparameter_MLP(cross_each_cross_validation_iter_list)
    find_the_sepearte_model_structure_hyperparameter_LR(cross_each_cross_validation_iter_list)
    # find_the_sepearte_model_structure_hyperparameter_RF(cross_each_cross_validation_iter_list)


