from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import torch
import sys
sys.path.append("")
from mmbt.utils.utils import *
from mmbt.utils.optimizer import *
from mmbt.utils.scheduler import *
from mmbt.utils.parseargs import *
from mmbt.forward import *
from mmbt.metrics.RankMetrics import *
from mmbt.metrics.RankMetrics import ndcg as normalized_dcg
from torchmetrics.utilities.data import select_topk

class Evaluate():
    """[Evaluate Class : Model evaluation]
    """
    def model_eval(self, data, model, args,loss_obj, store_preds=False):
        """[Evaluate function]

        Args:
            data ([torch.Dataloader]): [Dataloader]
            model ([torch.nn.Module]): [Model]
            args ([dict]): [Args]
            loss_obj ([torch.nn.Module]): [CrossSimilarity Class]
            store_preds (bool, optional): [Write prediction on file]. Defaults to False.

        """
        with torch.no_grad():
            losses, preds,tgts = [], [], []
            ndcg_list_at_10,ndcg_list_at_5,ndcg_list_at_3,ndcg_list_at_1, hit_list_at_10,hit_list_at_5,hit_list_at_3,hit_list_at_1, map_list, map_list_at_1 = [], [], [], [], [], [], [], [], [], []
            for batch in tqdm(data, total=len(data)):
                loss,out,tgt = model_forward(model,args,loss_obj,batch)
                losses.append(loss.item())
                if args.task_type == "multilabel":
                    pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
                else:
                    pred = torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy()
                    pred_no_npy = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach()
                    pred_no_npy_2 = torch.nn.functional.softmax(out, dim=1).squeeze(1).cpu().detach()
                    # pred_right = torch.nn.functional.softmax(out_r, dim=1).argmax(dim=1).cpu().detach().numpy()
                new_preds_5 = select_topk(pred_no_npy_2,topk=5,dim=0)
                new_preds_3 = select_topk(pred_no_npy_2,topk=3,dim=0)
                new_preds_1 = select_topk(pred_no_npy_2,topk=2,dim=0)
                new_preds_x = select_topk(pred_no_npy_2,topk=4,dim=0)
                tgt2 = tgt.cpu().detach().numpy()
                tgt_no_npy = tgt.cpu().detach()
                ndcg_list_at_5.append(normalized_dcg(new_preds_5,tgt_no_npy,ats=[5]).numpy())
                ndcg_list_at_3.append(normalized_dcg(new_preds_3,tgt_no_npy,ats=[3]).numpy())
                ndcg_list_at_1.append(normalized_dcg(new_preds_1,tgt_no_npy,ats=[2]).numpy())
                map_list.append(accuracy_score(tgt2,pred_no_npy.numpy()))
                map_list_at_1.append(accuracy_score(tgt2,pred_no_npy.numpy()))
                # hit_list_at_10.append(precision_at_k(pred,10))
                hit_list_at_5.append(accuracy_score(tgt.unsqueeze(1).cpu().detach().numpy(),new_preds_1.unsqueeze(1).numpy()))
                hit_list_at_3.append(accuracy_score(tgt.unsqueeze(1).cpu().detach().numpy(),new_preds_3.unsqueeze(1).numpy()))
                hit_list_at_1.append(accuracy_score(tgt.unsqueeze(1).cpu().detach().numpy(),new_preds_x.unsqueeze(1).numpy()))

            metrics = {"loss": np.nanmean(losses)}
            if args.task_type == "multilabel":
                tgts = np.vstack(tgts)
                preds = np.vstack(preds)
                metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
                metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
            else:
                tgts = [l for sl in tgts for l in sl]
                preds = [l for sl in preds for l in sl]
                # metrics["ndcg_10"] = np.nanmean(ndcg_list_at_10)
                metrics["ndcg_5"] = np.nanmean(ndcg_list_at_5)
                metrics["ndcg_3"] = np.nanmean(ndcg_list_at_3)
                metrics["ndcg_1"] = np.nanmean(np.array(ndcg_list_at_1).astype('float64'))
                # metrics["acc_10"] = np.nanmean(hit_list_at_10)
                metrics["acc_5"] = np.nanmean(hit_list_at_5)
                metrics["acc_3"] = np.nanmean(hit_list_at_3)
                metrics["acc_1"] = np.nanmean(hit_list_at_1)
                metrics["prec5"] = np.nanmean(map_list)
                metrics["prec1"] = np.nanmean(map_list_at_1)

            if store_preds:
                store_preds_to_disk(tgts, preds, args)

            return metrics