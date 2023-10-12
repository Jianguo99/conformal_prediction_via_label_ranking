import torch
import numpy as np



def postHocLogits(transforamtion,logits_loader,device,num_classes):
    """
    Transform the logits by the calibration technology
    """
    transforamtion.eval()
    transforamtion.to(device)

    
    logits = torch.zeros((len(logits_loader.dataset), num_classes)) # 1000 classes in Imagenet.
    labels = torch.zeros((len(logits_loader.dataset),))
    i = 0
    with torch.no_grad():
        for batch_logits, targets in logits_loader:
            batch_logits = transforamtion(batch_logits.to(device))
            logits[i:(i+batch_logits.shape[0]), :] = batch_logits
            labels[i:(i+batch_logits.shape[0])] = targets.cpu()
            i = i + batch_logits.shape[0]
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long()) 
    return dataset_logits



import torch.nn as nn 


class PostHoc(nn.Module):


    def forward(self,batch_logits):
        return batch_logits



class OptimalTeamperatureScaling(PostHoc):
    """optimal teamperature"""
    def __init__(self,temperature=1) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)*temperature)


    def forward(self,batch_logits):
        return batch_logits/self.temperature
    
    