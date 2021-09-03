import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from allrank.allrank.models.losses import approxNDCG

PADDED_Y_VALUE = -1

class CrossSimilarity(torch.nn.Module):
    
    def __init__(self):
        super(CrossSimilarity, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        
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
        
    
    def forward(self, output_l,output_r, target, output_neg_r):
        res_1 = self.similarities(output_l,output_r)
        res_2 = self.similarities(output_l,output_neg_r)
        res = torch.flatten(res_1, start_dim=1)
        target = target.expand(res.size()[0],res.size()[0])
        neg_res = torch.flatten(res_2, start_dim=1)
        return res,approxNDCG.approxNDCGLoss(res,target)
        # return res,self.triplet(output_l,output_r,output_neg_r)
    