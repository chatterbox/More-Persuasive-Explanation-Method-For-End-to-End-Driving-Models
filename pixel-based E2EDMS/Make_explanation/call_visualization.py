#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
import os
from DataLoading import UdacityDataset_zhang_gradcam as UD
import numpy as np
# import cv2
from model import Convolution3D_LSTM_transfer as CNN3D_LSTM
from model import Convolution2D_LSTM_transfer as CNN2D_LSTM
from model import Convolution3D_only_transfer as CNN3D
from Visualization import transfer_learning_Visualization as Vs

def load_dataset(cross_validation_counter):
    csv_folder_path = "/home0/zhangc/Dataset/BDD_100K_FLR/cross_validation_500"
    each_csv_folder_path_list = []
    for i in range(5):
        each_csv_folder_path = os.path.join(csv_folder_path, "regularized_csv_folder_500_" + str(i))
        each_csv_folder_path_list.append(each_csv_folder_path)

    test_csv_folder_path = [each_csv_folder_path_list[cross_validation_counter]]
    
    img_folder_dir_path = '/home0/zhangc/Dataset/BDD_100K_FLR/imgs_folder'
    test_csv_file_path_list = []

    Batch_size = 1
    Seq_len = 2 
    
    for each_csv_folder in test_csv_folder_path:
        for i in os.listdir(each_csv_folder):
            csv_file_path = os.path.join(each_csv_folder, i)
            test_csv_file_path_list.append(csv_file_path)
    
    explanation_dataset = UD.UdacityDataset(test_csv_file_path_list, img_folder_dir_path, seq_len = Seq_len, shuffle = True)
    
    loader_3dcnn = DataLoader(explanation_dataset, batch_size = Batch_size)
    return loader_3dcnn
def fake_main():

    if torch.cuda.is_available() == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")



    class_weights = [1, 1, 1]


    CNN2D_LSTM_model = CNN2D_LSTM.MyEnsemble()
    CNN3D_LSTM_model = CNN3D_LSTM.MyEnsemble()
    CNN3D_model = CNN3D.MyEnsemble()
    
    cnn3d_model_list = [CNN2D_LSTM_model, CNN3D_LSTM_model, CNN3D_model]
    model_category_list = ["CNN2D_LSTM", "CNN3D_LSTM", "CNN3D"]
    # we could select the model that we want to explain
    target_model_serial_num_list = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

    cnn_cam_extractor_list = [Vs.CamExtractor2DCNN_Resnet(cnn3d_model_list[0]), Vs.CamExtractor2DCNN_Resnet(cnn3d_model_list[1]), Vs.CamExtractor3DCNN(cnn3d_model_list[2])]
    trained_model_original_action_label_csv_folder_path = os.path.join(os.path.split(os.getcwd())[0], "Train_E2EDM","cross_validation_saver") 
    assert os.path.exists(trained_model_original_action_label_csv_folder_path)
    
    w = torch.FloatTensor(class_weights)
    criterion = nn.BCEWithLogitsLoss(pos_weight=w).to(device)
    for index in range(len(cnn3d_model_list)):
        cnn3d_model = cnn3d_model_list[index]
        
        model_category = model_category_list[index]
        cnn_cam_extractor = cnn_cam_extractor_list[index]

        target_heatmap_folder = os.path.join("./attention_mask_img/Pixel_heatmap", model_category) 
        target_mask_heatmap_folder = os.path.join("./attention_mask_img/Pixel_heatmap_mask", model_category)

        os.makedirs(target_heatmap_folder, exist_ok=True)
        os.makedirs(target_mask_heatmap_folder, exist_ok=True)
        # target_mask_heatmap_folder = os.path.join("Pixel_based_explanation", model_category)
        for cross_validation_index in range(5):
            loader_3dcnn = load_dataset(cross_validation_index)
            cross_validation_name = "cross_validation-" + str(cross_validation_index)
            target_model_serial_num = target_model_serial_num_list[index][cross_validation_index]
            
            target_model_name = str(target_model_serial_num) + ".pt"
            
            model_path = os.path.join(trained_model_original_action_label_csv_folder_path, model_category, cross_validation_name, target_model_name) 
            
            cnn3d_model.load_state_dict(torch.load(model_path))
            cnn3d_model = cnn3d_model.to(device)

            

            # with torch.no_grad(): 
            cnn3d_model.eval()
            for testing_sample in tqdm(loader_3dcnn):

                testing_sample['image'] = testing_sample['image'].permute(0,2,1,3,4)
                testing_sample['original_image'] = testing_sample['original_image'].permute(0,2,1,3,4)
                
                param_values = [v for v in testing_sample.values()]

                
                image = param_values[0]
                angle = param_values[1]
                image = image.to(device)   
                
                temp_reverse_image = image.clone()
                temp_reverse_image = temp_reverse_image.cpu()
                reverse_image = np.concatenate((temp_reverse_image[:,:,1,:,:][:,:,np.newaxis,:,:], temp_reverse_image[:,:,0,:,:][:,:,np.newaxis,:,:]), axis=2)

                assert reverse_image.all() != image.all(), ("reverse_image and image should not be the same")
                
                reverse_image = torch.from_numpy(reverse_image).to(device)   
                
                prediction = cnn3d_model(image)
                
                labels = angle.to(device)[:,-1,:]
                output_for_bakcprop = prediction.sum().float().to(device)


                labels = labels.float().to(device) 
                prediction = prediction.float().to(device) 

                
                loss = criterion(prediction, labels)

                output_for_bakcprop.backward()


                
                for time_index in range(len(testing_sample['image_name'])):
                    target_name = testing_sample['image_name'][time_index][0].split("-")[0] + "-" + testing_sample['image_name'][time_index][0].split("-")[1]
                    target_mask_heatmap_name = target_name + "_" + str(time_index+1) + ".jpg"
                    
                    target_mask_heatmap_path = os.path.join(target_mask_heatmap_folder, target_mask_heatmap_name)            
                    target_heatmap_path = os.path.join(target_heatmap_folder, target_mask_heatmap_name)            

                    cam_image = cnn_cam_extractor.to_image(width=1280, height=720) # Use this line to extract CAM image from the model!
                    

                    
                    if cam_image.shape[0] > 1:
                        
                        cam_image = cam_image[time_index, 0, :, :].numpy()
                    else:
                        
                        if time_index == 1:
                            prediction = cnn3d_model(image)
                            loss = criterion(prediction, labels)
                            loss.backward() # calculate the gradients
                            cam_image = cnn_cam_extractor.to_image(width=1280, height=720) # Use this line to extract CAM image from the model!
                            cam_image = cam_image[0, :, :].numpy()
                        else:
                            if time_index == 0:
                                reverse_prediction = cnn3d_model(reverse_image)
                                reverse_loss = criterion(reverse_prediction, labels)
                                reverse_loss.backward() # calculate the gradients
                                reverse_cam_image = cnn_cam_extractor.to_image(width=1280, height=720) # Use this line to extract CAM image from the model!
                                cam_image = reverse_cam_image[0, :, :].numpy()
                            else:
                                print("time_index", time_index)
                                exit()
                            

                    
                    plt.figure()
                    plt.axis('off')
                    
                    plt.imshow(cam_image, cmap="gray")
                    
                    plt.savefig(target_heatmap_path, bbox_inches='tight', dpi=258.1, pad_inches=0.0)
                    plt.close()

                    

                    plt.figure()
                    plt.axis('off')


                    plt.imshow(testing_sample['original_image'][0, :, time_index, :, :].permute(1, 2, 0))
                    plt.imshow(cam_image, cmap='jet', alpha=0.3)
                    
                    plt.savefig(target_mask_heatmap_path, bbox_inches='tight', dpi=258.1, pad_inches=0.0)
                    plt.close()



if __name__== '__main__':
    fake_main()