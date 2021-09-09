import warnings
import torch
import torch.backends.cudnn
from mmbt.train import Trainer

#! Main Class
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    Trainer()