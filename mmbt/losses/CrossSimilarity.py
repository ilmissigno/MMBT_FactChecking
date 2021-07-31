import torch
import torch.nn as nn
from matchzoo.losses.rank_hinge_loss import RankHingeLoss
from matchzoo.losses.rank_cross_entropy_loss import RankCrossEntropyLoss

class CrossSimilarity(torch.nn.Module):
    
    def __init__(self):
        super(CrossSimilarity, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.hinge = RankHingeLoss()
        #self.num_neg = 1
        
    def similarities(self,out_l,out_r):
        out_l = out_l / torch.norm(out_l,dim=1,keepdim=True)
        out_r = out_r / torch.norm(out_r,p=2,dim=1,keepdim=True)
        return torch.matmul(out_l,out_r.t())
    
    def hinge_loss(self,gt,pt):
        # """
        # Hinge pairwise loss function.

        # Parameters
        # ----------

        # positive_predictions: tensor
        #     Tensor containing predictions for known positive items.
        # negative_predictions: tensor
        #     Tensor containing predictions for sampled negative items.

        # Returns
        # -------

        # loss, float
        #     The mean value of the loss function.
        # """
        # # checked, usually we need to use a threshold as soft-margin (but this function does not have it)
        """
        y_pos = pt[::(self.num_neg + 1), :]
        y_neg = []
        for neg_idx in range(self.num_neg):
            neg = pt[(neg_idx + 1)::(self.num_neg + 1), :]
            y_neg.append(neg)
        y_neg = torch.cat(y_neg, dim=-1)
        y_neg = torch.mean(y_neg, dim=-1, keepdim=True)
        loss = torch.clamp(y_neg - y_pos + 1.0, 0.0)
        return loss.mean()
        """
        loss = torch.clamp(pt - gt + 1.0, 0.0)
        return loss.mean()
        # # loss = torch.mean(torch.max(1. - gt * pt))
        # # return loss
        # return self.hinge(pt,gt)
        
    
    def forward(self, output_l,output_r, target):
        res_1 = self.similarities(output_l,output_r)
        res = torch.flatten(res_1, start_dim=1)
        return res,self.hinge_loss(target,res)
    