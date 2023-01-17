
# In[58]:
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 
import torch
from skimage.transform import resize
from torch.nn import functional as F
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, register_hooks=True):
        
        self.model = model
        self.grad = None
        self.conv_output = None
        
        if register_hooks:
            
            self.register_hooks()
        
    def gradient_hook(self, model, grad_input, grad_output):
        
        self.grad = grad_output[0].cpu().detach().numpy()
        
    def conv_output_hook(self, model, input, output):
        
        self.conv_output = output.cpu().detach().numpy()
        
    def register_hooks(self):
        
        raise NotImplementedError("You should implement this method for your own model!")
        
    def forward(self, x):
        raise NotImplementedError("You should implement this method for your own model!")
        
    def to_image(self, height=None, width=None):
        
        raise NotImplementedError("You should implement this method for your own model!")
        
    
class CamExtractor2DCNN_Resnet(CamExtractor):
    
    def gradient_hook(self, model, grad_input, grad_output):
        
        
        self.grad = grad_output[0].cpu().detach().numpy()
        # self.grad = np.moveaxis(grad, 1, 2) # restore old dimension (batch x seq_len x channel x H x W)
        
    def conv_output_hook(self, model, input, output):
        
        self.conv_output = output.cpu().detach().numpy()
        # print("output.shape",output.shape) # (5, 2, 8, 19, 69)
        # self.conv_output = np.moveaxis(conv_output, 1, 2) # restore old dimension (batch x seq_len x channel x H x W)
    
    def register_hooks(self):
        # print(self.model)
        # print(self.model.modelA)
        # print("break here")
        # exit()
        self.model.modelA.layer4[1].conv2.register_forward_hook(self.conv_output_hook)
        self.model.modelA.layer4[1].conv2.register_full_backward_hook(self.gradient_hook)
    def to_image(self, height=None, width=None):
        
        # print("self.grad.shape",self.grad.shape) # (5, 2, 8, 19, 69)
        channel_weight = np.mean(self.grad, axis=(-2, -1)) # *, channel
        # print("channel_weight.shape",channel_weight.shape) # (5, 2, 8)

        # print("self.conv_output.shape",self.conv_output.shape) # (5, 2, 8, 19, 69)
        
        conv_permuted = np.moveaxis(self.conv_output, [-2, -1], [0, 1]) # H, W, *, channel
        # print(conv_permuted.shape) # (19, 69, 5, 2, 8)
        # exit()
        # cam_image_permuted = channel_weight * conv_permuted # H, W, *, channel
        # print(cam_image_permuted.shape) # (19, 69, 5, 2, 8)
        

        # cam_image_permuted = np.mean(cam_image_permuted, axis=-1) # H, W, *
        # print(cam_image_permuted.shape) # (19, 69, 5, 2)
        
        # cam_image = np.moveaxis(cam_image_permuted, [0, 1], [-2, -1]) # *, H, W
        # print(cam_image.shape) # (5, 2, 19, 69)

        # if height is not None and width is not None:
        #     image_shape = list(cam_image.shape)
        #     image_shape[-2] = height
        #     image_shape[-1] = width
        #     cam_image = resize(cam_image, image_shape)
        # cam_image = cam_image[:,-1,:,:]
        # print(cam_image.shape) # (5, 2, 120, 320)
        # print(conv_permuted.shape)
        # exit()
        
        
        
        single_img_fm = torch.from_numpy(conv_permuted).permute(2, 3, 0, 1)
        single_img_channel_weight = torch.from_numpy(conv_permuted).permute(2, 3, 0, 1)
        gcam = torch.mul(single_img_fm, single_img_channel_weight).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # print(gcam.shape)
        gcam = F.interpolate(
            gcam, [height, width], mode="bilinear", align_corners=False
        )
        # print(gcam.shape)
        # exit()
        B, C, H, W = gcam.shape
        # print(B, C, H, W)
        gcam = gcam.view(B, -1)
        # gcam -= gcam.min(dim=1, keepdim=True)[0]
        # gcam /= gcam.max(dim=1, keepdim=True)[0]
        cam_image = gcam.view(C, H, W)

        
        return cam_image 

class CamExtractor3DCNN(CamExtractor):
    
    def gradient_hook(self, model, grad_input, grad_output):
        
        grad = grad_output[0].cpu().detach().numpy()
        self.grad = np.moveaxis(grad, 1, 2) # restore old dimension (batch x seq_len x channel x H x W)
        
    def conv_output_hook(self, model, input, output):
        
        conv_output = output.cpu().detach().numpy()
        self.conv_output = np.moveaxis(conv_output, 1, 2) # restore old dimension (batch x seq_len x channel x H x W)
    
    def register_hooks(self):
        # print(self.model.modelA)
        # print(self.model.modelA.layer1)
        # exit()
        
        # self.model.modelA.layer1[1].conv2[0].register_forward_hook(self.conv_output_hook)
        # self.model.modelA.layer1[1].conv2[0].register_full_backward_hook(self.gradient_hook)

        self.model.modelA.layer4[1].conv2[0].register_forward_hook(self.conv_output_hook)
        self.model.modelA.layer4[1].conv2[0].register_full_backward_hook(self.gradient_hook)

        # self.model.Convolution6.register_forward_hook(self.conv_output_hook)
        # self.model.Convolution6.register_full_backward_hook(self.gradient_hook)
    def to_image(self, height=None, width=None):
        
        assert self.grad is not None and self.conv_output is not None, "You should perform both forward pass and backward propagation first!"
        # both grad and conv_output should have the same dimension of: (*, channel, H, W)
        # we produce image(s) of shape: (*, H, W)

        # print("self.grad.shape",self.grad.shape) # (5, 2, 8, 19, 69)
        channel_weight = np.mean(self.grad, axis=(-2, -1)) # *, channel
        # print("channel_weight.shape",channel_weight.shape) # (5, 2, 8)

        # print("self.conv_output.shape",self.conv_output.shape) # (5, 2, 8, 19, 69)
        
        conv_permuted = np.moveaxis(self.conv_output, [-2, -1], [0, 1]) # H, W, *, channel
        # print(conv_permuted.shape) # (19, 69, 5, 2, 8)

        cam_image_permuted = channel_weight * conv_permuted # H, W, *, channel
        # print(cam_image_permuted.shape) # (19, 69, 5, 2, 8)
        
        
        temp_conv_permuted = conv_permuted.squeeze(2)

        single_img_fm = torch.from_numpy(temp_conv_permuted).permute(2, 3, 0, 1)
        single_img_channel_weight = torch.from_numpy(temp_conv_permuted).permute(2, 3, 0, 1)
        gcam = torch.mul(single_img_fm, single_img_channel_weight)
        gcam = gcam.sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, [height, width], mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        # print(gcam.shape)
        # exit()
        gcam = gcam.view(B, -1)
        
        # gcam -= gcam.min(dim=1, keepdim=True)[0]
        # gcam /= gcam.max(dim=1, keepdim=True)[0]
        cam_image = gcam.view(1, C, H, W)
        double_cam_image = torch.cat((cam_image, cam_image), dim = 0)
        # cam_image = gcam.view(2, C, H, W)
        # print(cam_image.shape) # (5, 2, 120, 320)
        return double_cam_image
        
        single_img_fm = torch.from_numpy(conv_permuted[:,:,:,-1,:]).permute(2, 3, 0, 1)
        single_img_channel_weight = torch.from_numpy(conv_permuted[:,:,:,-1,:]).permute(2, 3, 0, 1)
        gcam = torch.mul(single_img_fm, single_img_channel_weight)
        gcam = gcam.sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, [height, width], mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        # print(B, C, H, W)
        gcam = gcam.view(B, -1)
        # gcam -= gcam.min(dim=1, keepdim=True)[0]
        # gcam /= gcam.max(dim=1, keepdim=True)[0]
        cam_image = gcam.view(C, H, W)
        # exit()

        # cam_image_permuted = np.mean(cam_image_permuted, axis=-1) # H, W, *
        # print(cam_image_permuted.shape) # (19, 69, 5, 2)
        
        # cam_image = np.moveaxis(cam_image_permuted, [0, 1], [-2, -1]) # *, H, W
        # print(cam_image.shape) # (5, 2, 19, 69)

        # if height is not None and width is not None:
        #     image_shape = list(cam_image.shape)
        #     image_shape[-2] = height
        #     image_shape[-1] = width
        #     cam_image = resize(cam_image, image_shape)
        print(cam_image.shape)
        # cam_image = cam_image[:,-1,:,:]
        # print(cam_image.shape)
        # print(cam_image.shape) # (5, 2, 120, 320)
        # exit()
        return cam_image
    

