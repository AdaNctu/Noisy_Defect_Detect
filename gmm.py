from sklearn.mixture import GaussianMixture
import torch
class Gmm:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.recoder_pos = []
        self.recoder_neg = []
        self.image_output = []
        self.image_label = []
    
    def add_output(self, model_output, train_label, epoch):
        model_output = model_output.cpu()
        train_label = train_label.cpu()
        if (train_label>0.0).any():
            self.recoder_pos[epoch].append(model_output[(train_label>0.0)].view(-1))
        self.recoder_neg[epoch].append(model_output[(train_label<=0.0)].view(-1))
        for image in model_output:
            self.image_output.append(image)
        for image in train_label:
            self.image_label.append(image)
        
        self.image_output = self.image_output[-512:]
        self.image_label = self.image_label[-512:]
    
    def new_epoch(self):
        self.recoder_pos.append([])
        self.recoder_neg.append([])
    
    def train(self):
        self.gmm_pos = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        self.gmm_neg = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        
        pos_data = [torch.cat(pos).view(-1) for pos in self.recoder_pos[-5:]]
        pos_data = torch.cat(pos_data).view(-1,1)
        
        neg_data = [torch.cat(neg).view(-1) for neg in self.recoder_neg[-5:]]
        neg_data = torch.cat(neg_data).view(-1,1)
        
        self.gmm_pos.fit(pos_data)
        self.gmm_neg.fit(neg_data)
        
        self.disagree = []
        for k in range(len(self.image_output)):
            mask_pos = (self.image_label[k]>0.0).view(-1)
            
            prob_pos = self.gmm_pos.predict_proba(self.image_output[k].view(-1,1))
            prob_pos = torch.tensor(prob_pos[:, self.gmm_pos.means_.argmin()]).view(-1)
            
            prob_neg = self.gmm_neg.predict_proba(self.image_output[k].view(-1,1))
            prob_neg = torch.tensor(prob_neg[:, self.gmm_neg.means_.argmax()]).view(-1)
            
            prob = torch.where(mask_pos, prob_pos, prob_neg)
            self.disagree.append(prob.pow(2.0).sum().item())
        
        self.disagree, _ = torch.tensor(self.disagree).sort(descending=True)
    
    def get_drop_th(self, drop_rate):
        drop_th = self.disagree[int(drop_rate*len(self.disagree))]
        
        return drop_th
    
    def eval(self, model_output, train_label, drop_rate):
        model_output = model_output.cpu()
        train_label = train_label.cpu()
        
        drop_th = self.disagree[int(drop_rate*len(self.disagree))]
        
        drop = []
        
        for k in range(model_output.shape[0]):
            mask_pos = (train_label[k]>0.0).view(-1)
            
            prob_pos = self.gmm_pos.predict_proba(model_output[k].view(-1,1))
            prob_pos = torch.tensor(prob_pos[:, self.gmm_pos.means_.argmin()]).view(-1)
            
            prob_neg = self.gmm_neg.predict_proba(model_output[k].view(-1,1))
            prob_neg = torch.tensor(prob_neg[:, self.gmm_neg.means_.argmax()]).view(-1)
            
            prob = torch.where(mask_pos, prob_pos, prob_neg)
            drop.append(prob.pow(2.0).sum().item() >= drop_th)
        
        return torch.tensor(drop)
    
        
        
        
        