from sklearn.mixture import GaussianMixture
import torch
class Gmm:
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
        self.gmm_pos = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4)
        self.gmm_neg = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4)
        
        image_output = torch.stack(self.image_output, dim=0)
        image_label = torch.stack(self.image_label, dim=0)
        mask = image_label > 0.0
        
        pos_data = image_output[mask]
        
        neg_data = image_output[mask.logical_not()]
        
        self.gmm_pos.fit(torch.sigmoid(pos_data))
        self.gmm_neg.fit(torch.sigmoid(neg_data))
        
        self.disagree = []
        for k in range(len(self.image_output)):
            mask_pos = (self.image_label[k]>0.0).view(-1)
            
            prob_pos = self.gmm_pos.predict_proba(torch.sigmoid(self.image_output[k]).view(-1,1))
            prob_pos = torch.tensor(prob_pos[:, self.gmm_pos.means_.argmin()]).view(-1)
            
            prob_neg = self.gmm_neg.predict_proba(torch.sigmoid(self.image_output[k]).view(-1,1))
            prob_neg = torch.tensor(prob_neg[:, self.gmm_neg.means_.argmax()]).view(-1)
            
            prob = torch.where(mask_pos, prob_pos, prob_neg)
            self.disagree.append(prob.sum().item())
        
        self.disagree, order = torch.tensor(self.disagree).sort(descending=True)
        self.indexs = [self.indexs[i] for i in order]
    
    def get_clear_index(self, drop_rate):
        return self.indexs[int(drop_rate*len(self.disagree)):]