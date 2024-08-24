import cv2
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt
from scipy.signal import convolve2d
from config import Config
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, cfg: Config, kind: str):
        super(Dataset, self).__init__()
        self.path: str = path
        self.cfg: Config = cfg
        self.kind: str = kind
        self.image_size: (int, int) = (self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT)
        self.grayscale: bool = self.cfg.INPUT_CHANNELS == 1

        self.num_negatives_per_one_positive: int = 1
        #self.frequency_sampling: bool = self.cfg.FREQUENCY_SAMPLING and self.kind == 'TRAIN'
        self.pool = torch.nn.AvgPool2d(32, stride=16, padding=0)
        self.th = 0.2
        self.with_index = None
        
        fake_image = torch.zeros(1, 1, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)
        self.ouput_size = self.pool(fake_image).shape
        
        if self.cfg.DATASET == "KSDD2":
            mean = (0.1767, 0.1706, 0.1750)
            std = (0.0347, 0.0354, 0.0403)
            self.normalize = transforms.Normalize(mean=mean, std=std)
        elif self.cfg.DATASET == "DAGM":
            mean = [0.2742, 0.3964, 0.5166, 0.6914, 0.4999, 0.3819]
            std = [0.1121, 0.2252, 0.1279, 0.0762, 0.1186, 0.2680]
            self.normalize = transforms.Normalize(mean=mean[self.cfg.FOLD-1], std=std[self.cfg.FOLD-1])
        else:
            self.normalize = None
    def get_ADL(self):
        total_noise = 0
        
        for item in self.neg_samples:
            total_noise += (item[1] != item[-2]).sum().item()
        for item in self.pos_samples:
            total_noise += (item[1] != item[-2]).sum().item()
        
        print("ADL", self.len, total_noise, item[1].shape)
        return self.len, total_noise, item[1].shape
        
    def set_with_index(self, with_index):
        self.with_index = with_index
    
    def clear_with_index(self):
        self.with_index = None
    
    def remain_noisy(self, with_index=None):
        if with_index is not None:
            noisy = 0
            for index in with_index:
                if index < self.num_neg:
                    ix = index
                    item = self.neg_samples[ix]
                else:
                    ix = index - self.num_neg
                    item = self.pos_samples[ix]
                noisy += item[-1].item()
            return noisy
        else:
            return -1
        
    def init_extra(self):
        self.counter = 0
        self.neg_imgs_permutation = np.random.permutation(self.num_neg)

        self.neg_retrieval_freq = np.zeros(shape=self.num_neg)
            
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor, bool, str):
        if self.with_index is not None:
            index = self.with_index[index]
        
        if index < self.num_neg:
            ix = index
            item = self.neg_samples[ix]
        else:
            ix = index - self.num_neg
            item = self.pos_samples[ix]

        image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, sample_name, train_mask, is_pos = item
        original_image = image.clone()
        
        if self.normalize is not None:
            image = self.normalize(image)

        self.counter = self.counter + 1
        
        if self.kind != 'TRAIN':
            return image, seg_mask, seg_loss_mask, is_segmented, sample_name, is_pos, train_mask, original_image
        else:
            return image, seg_mask, seg_loss_mask, is_segmented, sample_name, is_pos, train_mask, index

    def __len__(self):
        if self.with_index is None:
            return self.len
        else:
            return len(self.with_index)

    def read_contents(self):
        pass

    def read_img_resize(self, path, grayscale, resize_dim) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim)
        return np.array(img, dtype=np.float32) / 255.0

    def read_label_resize(self, path, resize_dim, dilate=None) -> (np.ndarray, bool):
        lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if dilate is not None and dilate > 1:
            lbl = cv2.dilate(lbl, np.ones((dilate, dilate)))
        if resize_dim is not None:
            lbl = cv2.resize(lbl, dsize=resize_dim)
        return np.array((lbl / 255.0), dtype=np.float32), np.max(lbl) > 0

    def to_tensor(self, x) -> torch.Tensor:
        if x.dtype != np.float32:
            x = (x / 255.0).astype(np.float32)

        if len(x.shape) == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        else:
            x = np.expand_dims(x, axis=0)

        x = torch.from_numpy(x)
        return x

    def downsize(self, image: np.ndarray, downsize_factor: int = 8) -> np.ndarray:
        img_t = torch.from_numpy(np.expand_dims(image, 0 if len(image.shape) == 3 else (0, 1)).astype(np.float32))
        img_t = torch.nn.ReflectionPad2d(padding=(downsize_factor))(img_t)
        image_np = torch.nn.AvgPool2d(kernel_size=2 * downsize_factor + 1, stride=downsize_factor)(img_t).detach().numpy()
        return image_np[0] if len(image.shape) == 3 else image_np[0, 0]
    
    def get_patch_level(self, mask):
        mask = (mask>0.0).float()
        patch_mask = self.pool(mask)
        patch_mask = (patch_mask > self.th).float()
        
        return patch_mask
        
