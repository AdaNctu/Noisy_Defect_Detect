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
        defect_size = []
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
                defect_size.append(seg_mask.sum().item())
            else:
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part, seg_mask, torch.zeros(1)))
        
        print(self.kind, len(pos_samples), len(neg_samples))
        defect_size = torch.tensor(defect_size).float()
        print("mean & std",defect_size.mean(),defect_size.std())
        th0 = (defect_size.mean()-defect_size.std())//9
        th0 = max(th0, 1.0)
        th1 = (defect_size.mean()+defect_size.std())//9
        th1 = max(th1, th0+1)
        print("th=",th0, th1)
        
        if self.cfg.NOISE_RATE is not None and self.cfg.NOISE_RATE > 0:
            nois_rate = self.cfg.NOISE_RATE
            nois_rate = nois_rate/(6.0-3.0*nois_rate)
            num_noisy = int(len(pos_samples)*nois_rate)
            print(f'num_noisy={num_noisy}')
            _, order_pos = torch.tensor([pos[1].sum().item() for pos in pos_samples]).sort(descending=True)
            offset = [i*(len(pos_samples)//3) for i in range(3)]#2 add, 1 big, 0 small
            
            if self.kind == 'TRAIN':
                for i in range(3):
                    for k in range(num_noisy):
                        sample = list(pos_samples[order_pos[k+offset[i]]])
                        noise_mask = add_noise(sample[1], i, th0, th1)
                        sample[1] = noise_mask
                        if not self.cfg.CLEAN_TRAIN:
                            sample[-2] = noise_mask
                            sample[2] = torch.where((noise_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
                        sample[-1] = torch.ones(1)
                        pos_samples[order_pos[k+offset[i]]] = tuple(sample)
                
                for k in range(3*num_noisy):
                    sample = list(neg_samples[k])
                    noise_mask = add_noise(sample[1], 2, th0, th1)
                    sample[1] = noise_mask
                    if not self.cfg.CLEAN_TRAIN:
                        sample[-2] = noise_mask
                        sample[2] = torch.where((noise_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
                    sample[-1] = torch.ones(1)
                    neg_samples[k] = tuple(sample)
                
            elif self.kind == 'TEST':
                correct_label = []
                for i in range(3):
                    for k in range(num_noisy):
                        correct_label.append(pos_samples[order_pos[k+offset[i]]])
                        sample = list(pos_samples[order_pos[k+offset[i]]])
                        noise_mask = add_noise(sample[1], i, th0, th1)
                        sample[1] = noise_mask
                        sample[-1] = torch.ones(1)
                        pos_samples[order_pos[k+offset[i]]] = tuple(sample)
                
                for k in range(3*num_noisy):
                    correct_label.append(neg_samples[k])
                    sample = list(neg_samples[k])
                    noise_mask = add_noise(sample[1], 2, th0, th1)
                    sample[1] = noise_mask
                    sample[-1] = torch.ones(1)
                    neg_samples[k] = tuple(sample)
        
        self.pos_samples = []
        self.neg_samples = []
        if self.kind in ['TRAIN']:
            for item in pos_samples+neg_samples:
                if item[-1]:
                    self.pos_samples.append(item)
                elif item[1].max():
                    self.neg_samples.append(item)
        elif self.kind in ['TEST']:
            for item in pos_samples+neg_samples:
                if item[-1]:
                    self.pos_samples.append(item)
                elif item[1].max():
                    self.neg_samples.append(item)
            self.neg_samples = self.neg_samples+correct_label
        
        print(self.kind, len(self.pos_samples), len(self.neg_samples))
        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)
        self.len = len(self.pos_samples) + len(self.neg_samples)
        
        self.init_extra()
