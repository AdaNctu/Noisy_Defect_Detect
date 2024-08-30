from sklearn.mixture import GaussianMixture
import torch
class Gmm2:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.image_output = []
        self.image_label = []
        self.indexs = []
        
    def add_output(self, model_output, train_label, index):
        model_output = model_output.cpu()
        train_label = train_label.cpu()
        for image in model_output:
            self.image_output.append(image)
        for image in train_label:
            self.image_label.append(image)
        self.indexs = self.indexs + index.tolist()
    
    def new_epoch(self):
        self.image_output = []
        self.image_label = []
        self.indexs = []
    
    def train(self):
        self.gmm = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4)
        
        image_output = torch.stack(self.image_output, dim=0)
        image_label = torch.stack(self.image_label, dim=0)
        mask = image_label > 0.0
        
        data = torch.where(mask, image_output, -image_output)
        
        self.gmm.fit(torch.sigmoid(data).view(-1,1))
        
        self.disagree = []
        for k in range(len(self.image_output)):
            mask_pos = (self.image_label[k]>0.0)
            
            data = torch.where(mask_pos, self.image_label[k], -self.image_label[k])
            
            prob = self.gmm.predict_proba(torch.sigmoid(data).view(-1,1))
            prob = torch.tensor(prob[:, self.gmm.means_.argmin()]).view(-1)
            
            self.disagree.append(prob.sum().item())
        
        self.disagree, order = torch.tensor(self.disagree).sort(descending=True)
        self.indexs = [self.indexs[i] for i in order]
    
    def subset_sampling(self, num, drop_rate, subset_size):
        total_drop = int(num*drop_rate)
        #split to subsets
        padding = (subset_size-1)-(num-1)%subset_size
        if padding:
            order = torch.cat([torch.randperm(num),torch.tensor([-1]*padding)])
        else:
            order = torch.randperm(num)
        order = order.view(-1, subset_size)
        order, _ = order.sort(descending=False, dim=1)
        
        #drop
        num_subset = order.shape[0]
        clear_idx = []
        for i in range(num_subset):
            drop_num = int(total_drop*(i+1)/num_subset)-int(total_drop*i/num_subset)
            if i+1 == num_subset:
                drop_num += padding
            clear_idx += order[i][drop_num:].tolist()
        
        return clear_idx
        
    def get_clear_index(self, drop_rate):
        indexs = self.subset_sampling(len(self.indexs), drop_rate, 8)
        
        return [self.indexs[i] for i in indexs]
        
        #return self.indexs[int(drop_rate*len(self.disagree)):]