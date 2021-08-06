import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from matchzoo.losses.rank_hinge_loss import RankHingeLoss
from matchzoo.losses.rank_cross_entropy_loss import RankCrossEntropyLoss

PADDED_Y_VALUE = -1

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
    
    def bce(self,y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
        """
        Binary Cross-Entropy loss.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        y_pred = y_pred.float()
        y_true = y_true.float()

        mask = y_true == padded_value_indicator
        valid_mask = y_true != padded_value_indicator

        ls = BCEWithLogitsLoss(reduction='none')(y_pred, y_true)
        ls[mask] = 0.0

        document_loss = torch.sum(ls, dim=-1)
        sum_valid = torch.sum(valid_mask, dim=-1).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

        loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

        return loss_output.mean()
    
    def hinge_loss(self,neg_p,pos_p,target):
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
        return F.margin_ranking_loss(
            pos_p, neg_p, target,
            margin=0.,
            reduction='mean',
        )
        # loss = torch.clamp(pt - gt + 1.0, 0.0)
        # return loss.mean()
        # # loss = torch.mean(torch.max(1. - gt * pt))
        # # return loss
        # return self.hinge(pt,gt)
        
    
    def forward(self, output_l,output_r, target, output_neg_r):
        res_1 = self.similarities(output_l,output_r)
        res_2 = self.similarities(output_l,output_neg_r)
        res = torch.flatten(res_1, start_dim=1)
        target = target.expand(res.size()[0],res.size()[0])
        neg_res = torch.flatten(res_2, start_dim=1)
        return res,self.bce(res,target)
    