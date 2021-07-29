import torch
import torch.nn as nn

class CrossSimilarity(torch.nn.Module):
    
    def __init__(self):
        super(CrossSimilarity, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def similarities(self,out_l,out_r):
        out_l = out_l / torch.norm(out_l,dim=1,keepdim=True)
        out_r = out_r / torch.norm(out_r,p=2,dim=1,keepdim=True)
        return torch.matmul(out_l,out_r.t())
    
    def hinge_loss(self,gt,pt):
        """
        Hinge pairwise loss function.

        Parameters
        ----------

        positive_predictions: tensor
            Tensor containing predictions for known positive items.
        negative_predictions: tensor
            Tensor containing predictions for sampled negative items.

        Returns
        -------

        loss, float
            The mean value of the loss function.
        """
        # checked, usually we need to use a threshold as soft-margin (but this function does not have it)
        loss = torch.clamp(pt - gt + 1.0, 0.0)
        return loss.sum()
    
    def forward(self, output_l,output_r, target):
        res = self.similarities(output_l,output_r)
        return res,self.hinge_loss(target,res)
    