import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config
import torch
from add_noise import *

def read_split(num_segmented: int, kind: str):
    fn = f"KSDD2/split_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples
        elif kind == 'TEST':
            return test_samples
        else:
            raise Exception('Unknown')


class KSDD2Dataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(KSDD2Dataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

        data_points = read_split(self.cfg.NUM_SEGMENTED, self.kind)
        #self.cfg.NUM_NOISY, self.cfg.NOISY_TYPE
        for part, is_segmented in data_points:
            image_path = os.path.join(self.path, self.kind.lower(), f"{part}.png")
            seg_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_GT.png")

            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, None)
            
            image = self.to_tensor(image)
            seg_mask = self.get_patch_level(self.to_tensor(seg_mask))
            seg_loss_mask = torch.where((seg_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
            
            if seg_mask.sum():
                pos_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part, seg_mask, torch.zeros(1)))
            else:
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part, seg_mask, torch.zeros(1)))
        
        print(self.kind, len(pos_samples), len(neg_samples))
        
        noisy_flag = False
        if self.cfg.NUM_NOISY is not None and self.cfg.NUM_NOISY > 0:
            num_noisy = len(pos_samples)*self.cfg.NUM_NOISY//100
            print(f'num_noisy={num_noisy}')
            _, order_pos = torch.tensor([pos[1].sum().item() for pos in pos_samples]).sort()
            offset = [i*(len(pos_samples)//3) for i in range(3)]#0 add, 1 big, 2 small
            
            if self.cfg.NOISY_TYPE in [2, 3] and self.kind == 'TRAIN':
                noisy_flag = True
                for i in range(3):
                    for k in range(num_noisy):
                        sample = list(pos_samples[order_pos[k+offset[i]]])
                        noise_mask = add_noise(sample[1], i)
                        sample[1] = noise_mask
                        if self.cfg.NOISY_TYPE in [2]:
                            sample[-2] = noise_mask
                            sample[2] = torch.where((noise_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
                        sample[-1] = torch.ones(1)
                        pos_samples[order_pos[k+offset[i]]] = tuple(sample)
                
                for k in range(num_noisy):
                    sample = list(neg_samples[k])
                    noise_mask = add_noise(sample[1], 0)
                    sample[1] = noise_mask
                    if self.cfg.NOISY_TYPE in [2]:
                        sample[-2] = noise_mask
                        sample[2] = torch.where((noise_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
                    sample[-1] = torch.ones(1)
                    neg_samples[k] = tuple(sample)
                
            elif self.cfg.NOISY_TYPE in [2, 3] and self.kind == 'TEST':
                noisy_flag = True
                for i in range(3):
                    for k in range(num_noisy):
                        sample = list(pos_samples[order_pos[k+offset[i]]])
                        noise_mask = add_noise(sample[1], i)
                        sample[1] = noise_mask
                        sample[-1] = torch.ones(1)
                        pos_samples[order_pos[k+offset[i]]] = tuple(sample)
                
                for k in range(num_noisy):
                    sample = list(neg_samples[k])
                    noise_mask = add_noise(sample[1], 0)
                    sample[1] = noise_mask
                    sample[-1] = torch.ones(1)
                    neg_samples[k] = tuple(sample)
        
        if noisy_flag:
            self.pos_samples = []
            self.neg_samples = []
            for item in pos_samples+neg_samples:
                if item[1].max():
                    self.pos_samples.append(item)
                else:
                    self.neg_samples.append(item)
            
        else:
            self.pos_samples = pos_samples
            self.neg_samples = neg_samples
        print(self.kind, len(self.pos_samples), len(self.neg_samples))
        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)
        self.len = 2 * len(self.pos_samples) if self.kind in ['TRAIN'] else len(self.pos_samples) + len(self.neg_samples)

        self.init_extra()
