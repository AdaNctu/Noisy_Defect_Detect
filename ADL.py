import torch
class ADL:
    def __init__(self):
        pass
    
    def new_epoch(self, num_samples, shape):
        C, W, H = shape
        self.confidence = torch.ones(num_samples, C, W, H)
        self.mask = torch.ones(num_samples, C, W, H)
    
    def add_output(self, model_output, train_label, index):
        model_output = model_output.cpu()
        model_confidence = torch.sigmoid(model_output).cpu()
        train_label = train_label.cpu()
        index = index.view(-1)
        
        for i in range(len(index)):
            idx = index[i]
            self.confidence[idx] = torch.where(train_label[i]>0.0, model_confidence[i], 1.0-model_confidence[i])
    
    def sample_selection(self, drop_num):
        _, order = self.confidence.view(-1).topk(drop_num, largest=False)
        self.mask.view(-1)[order] = 0.0
    
    def get_mask(self, index):
        
        return self.mask[index.view(-1)]