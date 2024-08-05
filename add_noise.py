import torch
import torch.nn as nn

def add_noise(mask, noise_type):
    h, w = mask.shape[-2:]
    
    h0, h1 = 0, h
    w0, w1 = 0, w
    
    H0, H1 = 0, h
    W0, W1 = 0, w
    
    if noise_type==0:
        #small
        shift = torch.zeros(mask.shape)
        offset = torch.randint(1,4,(1,))
        offset = (offset, 4-offset)
        if torch.randn(1) > 0:
            h0 += offset[0]
            H1 -= offset[0]
        else:
            h1 -= offset[0]
            H0 += offset[0]
        if torch.randn(1) > 0:
            w0 += offset[1]
            W1 -= offset[1]
        else:
            w1 -= offset[1]
            W0 += offset[1]
        
        shift[:,h0:h1,w0:w1] = mask[:,H0:H1,W0:W1]
        
        return mask*shift
        
    elif noise_type==1:
        #big
        while True:
            shift = torch.zeros(mask.shape)
            offset = torch.randint(1,4,(1,))
            offset = (offset, 4-offset)
            if torch.randn(1) > 0:
                h0 += offset[0]
                H1 -= offset[0]
            else:
                h1 -= offset[0]
                H0 += offset[0]
            if torch.randn(1) > 0:
                w0 += offset[1]
                W1 -= offset[1]
            else:
                w1 -= offset[1]
                W0 += offset[1]
            
            shift[:,h0:h1,w0:w1] = mask[:,H0:H1,W0:W1]
            if (shift*(1-mask)).sum():
                break
            h0, h1 = 0, h
            w0, w1 = 0, w
            
            H0, H1 = 0, h
            W0, W1 = 0, w
        
        return (mask+shift).clip(0,1)
    
    elif noise_type==2:
        #add
        shift = torch.zeros(mask.shape)
        h0 = torch.randint(0,h,(1,))
        w0 = torch.randint(0,w,(1,))
        while shift[:,h0,w0] == 1.0:
            h0 = torch.randint(0,h,(1,))
            w0 = torch.randint(0,w,(1,))
        pool = nn.MaxPool2d(3,1,1)
        for _ in range(torch.randint(1,10,(1,)).item()):
            if len(shift.shape)==4:
                shift[:,:,h0,w0] = 1.0
            else:
                shift[:,h0,w0] = 1.0
            h0 += torch.randint(-1,2,(1,))*2
            w0 += torch.randint(-1,2,(1,))*2
            h0 = h0.clip(0, h-1)
            w0 = w0.clip(0, w-1)
        shift = pool(shift)
        
        return (mask+shift).clip(0,1)
    
    return None