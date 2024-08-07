import numpy as np
import os
import pickle
from data.dataset import Dataset
import torch
from add_noise import *


def read_split(num_segmented: int, fold: int, kind: str):
    fn = f"DAGM/split_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples[fold - 1]
        elif kind == 'TEST':
            return test_samples[fold - 1]
        else:
            raise Exception('Unknown')


class DagmDataset(Dataset):
    def __init__(self, kind: str, cfg):
        super(DagmDataset, self).__init__(os.path.join(cfg.DATASET_PATH, f"Class{cfg.FOLD}"), cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []
        
        samples = read_split(self.cfg.NUM_SEGMENTED, self.cfg.FOLD, self.kind)

        sub_dir = self.kind.lower().capitalize()
        
        defect_size = []

        for image_name, is_segmented in samples:
            image_path = os.path.join(self.path, sub_dir, image_name)
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            img_name_short = image_name[:-4]
            seg_mask_path = os.path.join(self.path, sub_dir, "Label",  f"{img_name_short}_label.PNG")
            
            if os.path.exists(seg_mask_path):
                seg_mask, _ = self.read_label_resize(seg_mask_path, self.image_size, None)
                image = self.to_tensor(image)
                seg_mask = self.get_patch_level(self.to_tensor(seg_mask))
                defect_size.append(seg_mask.sum().item())
                seg_loss_mask = torch.where((seg_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))

            else:
                seg_mask = np.zeros_like(image)
                image = self.to_tensor(image)
                seg_mask = self.get_patch_level(self.to_tensor(seg_mask))
                seg_loss_mask = torch.where((seg_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
            
            if seg_mask.sum():
                pos_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, img_name_short, seg_mask, torch.zeros(1)))
            else:
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, img_name_short, seg_mask, torch.zeros(1)))

        print(self.kind, len(pos_samples), len(neg_samples))
        defect_size = torch.tensor(defect_size).float()
        print("mean & std",defect_size.mean(),defect_size.std())
        th0 = (defect_size.mean()-defect_size.std())//9
        th0 = max(th0, 1.0)
        th1 = (defect_size.mean()+defect_size.std())//9
        th1 = max(th1, th0+1)
        print("th=",th0, th1)
        
        noisy_flag = False
        if self.cfg.NUM_NOISY is not None and self.cfg.NUM_NOISY > 0:
            num_noisy = len(pos_samples)*self.cfg.NUM_NOISY//100
            print(f'num_noisy={num_noisy}')
            _, order_pos = torch.tensor([pos[1].sum().item() for pos in pos_samples]).sort(descending=True)
            offset = [i*(len(pos_samples)//3) for i in range(3)]#2 add, 1 big, 0 small
            
            if self.cfg.NOISY_TYPE in [2, 3] and self.kind == 'TRAIN':
                noisy_flag = True
                for i in range(3):
                    for k in range(num_noisy):
                        sample = list(pos_samples[order_pos[k+offset[i]]])
                        noise_mask = add_noise(sample[1], i, th0, th1)
                        sample[1] = noise_mask
                        if self.cfg.NOISY_TYPE in [2]:
                            sample[-2] = noise_mask
                            sample[2] = torch.where((noise_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
                        sample[-1] = torch.ones(1)
                        pos_samples[order_pos[k+offset[i]]] = tuple(sample)
                
                for k in range(3*num_noisy):
                    sample = list(neg_samples[k])
                    noise_mask = add_noise(sample[1], 2, th0, th1)
                    sample[1] = noise_mask
                    if self.cfg.NOISY_TYPE in [2]:
                        sample[-2] = noise_mask
                        sample[2] = torch.where((noise_mask>0.0), torch.tensor(self.cfg.WEIGHTED_DEFECT), torch.tensor(1.0))
                    sample[-1] = torch.ones(1)
                    neg_samples[k] = tuple(sample)
                
            elif self.cfg.NOISY_TYPE in [2, 3] and self.kind == 'TEST':
                correct_label = []
                noisy_flag = True
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
        
        if noisy_flag:
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
        else:
            self.pos_samples = pos_samples
            self.neg_samples = neg_samples
        print(self.kind, len(self.pos_samples), len(self.neg_samples))
        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)
        self.len = 2*len(self.pos_samples) if self.kind in ['TRAIN'] else len(self.pos_samples) + len(self.neg_samples)
        if self.kind in ['TRAIN'] and not self.cfg.FREQUENCY_SAMPLING:
            self.len = len(self.pos_samples) + len(self.neg_samples)
        
        self.init_extra()
