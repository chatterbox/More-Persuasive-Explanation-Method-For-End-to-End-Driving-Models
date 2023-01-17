import torch
class Config(object):
    """配置参数"""

    def __init__(self):
        self.action_label_csv_path = "action_label/zhang_pesudo_img.csv"

        self.train_folder_path = "train_pesudo_img_label_821"
        self.vali_folder_path = "vali_pesudo_img_label_821"
        self.cross_validation_folder_path_pre = "cross_validation_pesudo_img_"

        self.object_settings_path = "settings/object_setting_no_repeat.csv"
        self.object_settings_path_reverse = "settings/object_setting_no_repeat_reverse.csv"

        self.original_image_heatmap = "original_image_heatmap"
        self.heatmap_mask = "heatmap_mask"
        self.simulation_gradual_image = "simulation"
        self.verification_action_label = "verification"
        self.counterfactual_image_folder = "counterfactual"

        self.Network_global_explanation_folder = "Network_global_explanation"
        self.pesudo_img_folder_name = "Pesudo_img_folder_500"

        self.moveable_object_num = 5
        self.lane_object_num = 4
        self.traffic_light_object_num = 2
        self.object_position_cluster_num = 3
        self.object_size_cluster_num = 2
        self.object_motion_cluster_num = 4

        self.moveable_object_cluster_num = self.object_position_cluster_num * self.object_size_cluster_num * self.object_motion_cluster_num
        self.lane_object_cluster_num = self.object_position_cluster_num * self.object_position_cluster_num
        self.traffic_light_object_cluster_num = self.object_position_cluster_num * self.object_size_cluster_num

        self.position_info_range_in_all_info = [0, 1]
        self.size_info_range_in_all_info = [2, 3]
        self.motion_info_range_in_all_info = [4, 5]
        self.end_position_info_range_in_all_info = [2, 3]

        self.save_path = 'Logistic_Regression_3_cluster_1or0'
        self.weight_decay = 1
        self.decay_step = 5000

        self.device = 'cpu'

