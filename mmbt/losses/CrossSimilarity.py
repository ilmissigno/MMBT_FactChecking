import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from allrank.allrank.models.losses import approxNDCG

PADDED_Y_VALUE = -1 #Costant padding value for ApproxNDCG

class CrossSimilarity(torch.nn.Module):
    """Cross Similarity class, is a class for calculate losses 
    and similarity between the outputs of the model.
    
    Functions:
        similarities : function for calculate the cosine similarity
            :return: cosine similarity tensor
        forward: a function inherit by torch.nn.Module
            :return: the output of similarity and the loss 
                calculated between similarity tensor and the target
    Args:
        torch (torch.nn.Module): [description]
    """
    
    def __init__(self):
        super(CrossSimilarity, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=1,eps=1e-6)
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        
    def similarities(self,out_l,out_r):
        """[Cosine similarity between query and document from the single batch]

        Args:
            out_l ([torch.Tensor]): [Query tensor]
            out_r ([torch.Tensor]): [Doc tensor]

        Returns:
            [torch.Tensor]: [Cosine similarity Tensor between Query and Doc Tensors]
        """
        #! Cosine similarity between query and document from the single batch
        return self.cos(out_l,out_r).unsqueeze(1)
        
    def forward(self, output_l,output_r, target):
        res = self.similarities(output_l,output_r)
        target = target.unsqueeze(1)
        score = self.cos(output_l,output_r)
        #! If using approxndcg loss
        return score
        #! If using triplet hinge loss
        # return res,self.triplet(output_l,output_r,output_neg_r)
    