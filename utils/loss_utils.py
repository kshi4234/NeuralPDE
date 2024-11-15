# Define loss functions in this file depending on type of PDE.
import torch
# Generic loss function class. Can be implemented by specific loss functions
# Pretty much useless, just wrote this to have something to import
class Generic_Loss():
    def __init__(self):
        return
    
    def compute_loss(self):
        return

# Mean Squared Error
def mse_loss(x, y, reduction='mean'):
    return torch.nn.MSELoss(x, y, reduction=reduction)
    
    