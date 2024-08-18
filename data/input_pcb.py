import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config
import torch
from add_noise import *

def read_split(num_segmented: int, kind: str):
    fn = f"PCB/PCB_list"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = torch.load(f)
        if kind == 'TRAIN':
            return train_samples
        elif kind == 'TEST':
            return test_samples
        else:
            raise Exception('Unknown')


class PCBDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(PCBDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

        data_points = read_split(self.cfg.NUM_SEGMENTED, self.kind)
        for part in data_points:
            image_path = os.path.join(self.path, self.kind.lower(), f"{part}_0.jpg")
            dirty_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_1.jpg")
            clean_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_2.jpg")

            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            dirty_mask, _ = self.read_label_resize(dirty_mask_path, self.image_size, None)
            clean_mask, _ = self.read_label_resize(clean_mask_path, self.image_size, None)

            image = self.to_tensor(image)
            dirty_mask = self.get_patch_level(self.to_tensor(dirty_mask))
            clean_mask = self.get_patch_level(self.to_tensor(clean_mask))
            positive = (dirty_mask!=clean_mask).any()
            if not self.cfg.CLEAN_TRAIN and self.kind == 'TRAIN':
                clean_mask = dirty_mask
            
            seg_loss_mask = torch.where((clean_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
            
            if positive:
                pos_samples.append((image, dirty_mask, seg_loss_mask, True, image_path, clean_mask_path, part, clean_mask, torch.ones(1)))
            else:
                neg_samples.append((image, dirty_mask, seg_loss_mask, True, image_path, clean_mask_path, part, clean_mask, torch.zeros(1)))
        
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        print(self.kind, len(self.pos_samples), len(self.neg_samples))
        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)
        self.len = len(self.pos_samples) + len(self.neg_samples)

        self.init_extra()
