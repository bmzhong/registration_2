import os
import torch


class ModelSaver:
    def __init__(self, max_save_num):
        """
        :param max_save_num: max checkpoint number to save
        """
        self.save_path_list = []
        self.max_save_num = max_save_num

    def save(self, path, state_dict):
        self.save_path_list.append(path)
        if len(self.save_path_list) > self.max_save_num:
            top = self.save_path_list.pop(0)
            os.remove(top)
        torch.save(state_dict, path)
