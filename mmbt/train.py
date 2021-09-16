#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import sys
import gc
from random import randint
sys.path.append("")
from mmbt.data.helpers import get_data_loaders_left_right, get_data_loaders_doc_extraction
from mmbt.models import get_model
from mmbt.utils.logger import create_logger
from mmbt.utils.utils import *
from mmbt.utils.optimizer import *
from mmbt.utils.scheduler import *
from mmbt.utils.parseargs import *
from mmbt.evaluate import Evaluate
from mmbt.forward import *
from mmbt.losses.CrossSimilarity import CrossSimilarity
from mmbt.metrics.RankMetrics import *
from mmbt.metrics.RankMetrics import ndcg as normalized_dcg


class Trainer():
    """
    [Trainer Class : This class execute the training process]
    """
    def __init__(self):
        args = get_parsed_args()
        self.train(args)

    def train(self,args):
        """
            Train function
        """
        loss_obj = CrossSimilarity()
        logger = create_logger("%s/logfile.log" % args.savedir, args)
        seed_val = randint(0, 10000)
        logger.warning("*"*50+" SEED : "+str(seed_val)+" "+"*"*50)
        set_seed(seed_val)
        args.savedir = os.path.join(args.savedir, args.name)
        os.makedirs(args.savedir, exist_ok=True)
        train_loader, val_loader, test_loaders = get_data_loaders_left_right(args)
        model = get_model(args)
        optimizer1 = get_optimizer(model, args)
        scheduler1 = get_scheduler(optimizer1, args)
        logger.info(model)
        model.cuda()
        torch.save(args, os.path.join(args.savedir, "args.pt"))
        start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
        logger.info("Training..")
        
        """
        TRAINING PHASE
        """
        for i_epoch in range(start_epoch,args.max_epochs):
            logger.warning("*"*50+" EPOCH "+str(i_epoch)+" "+"*"*50)
            train_losses = []
            model.train()
            optimizer1.zero_grad()
            model.zero_grad()
            counterz = 0
            for batch in tqdm(train_loader, total=len(train_loader)):
                with torch.cuda.amp.autocast():
                    loss,_,_ = model_forward(model,args,loss_obj,batch)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                train_losses.append(loss.item())
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                counterz+=1
                if counterz == 100:
                    break
            
            """
            [VALIDATION PHASE]
            """
            model.eval()
            logger.info("Validation...")
            metrics = Evaluate().model_eval(val_loader, model,args,loss_obj)
            log_metrics("Val", metrics, args, logger)
            tuning_metric = (
                metrics["micro_f1"] if args.task_type == "multilabel" else metrics["ndcg_5"]
            )
            scheduler1.step(tuning_metric)
            is_improvement = tuning_metric > best_metric
            if is_improvement:
                best_metric = tuning_metric
                n_no_improve = 0
            else:
                n_no_improve += 1

            save_checkpoint(
                {
                    "epoch": i_epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer1.state_dict(),
                    "scheduler": scheduler1.state_dict(),
                    "n_no_improve": n_no_improve,
                    "best_metric": best_metric,
                },
                is_improvement,
                args.savedir,
            )

            if n_no_improve >= args.patience:
                logger.info("No improvement. Breaking out of loop.")
                break
        
        """
        TEST PHASE
        """
        model.eval()
        logger.info("Test...")
        load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
        test_metrics = Evaluate().model_eval(test_loaders,model,args,loss_obj, store_preds=True)
        save_metrics(test_metrics,"snopes_test_no_text.txt")
        log_metrics(f"Test - ", test_metrics, args, logger)
    
    def doc_feat_extraction(self,args):
        loss_obj = CrossSimilarity()
        logger = create_logger("%s/logfile.log" % args.savedir, args)
        seed_val = randint(0, 10000)
        logger.warning("*"*50+" SEED : "+str(seed_val)+" "+"*"*50)
        set_seed(seed_val)
        args.savedir = os.path.join(args.savedir, args.name)
        os.makedirs(args.savedir, exist_ok=True)
        train_query_loader,train_doc_loader,val_query_loader,val_doc_loader,test_query_loaders,test_doc_loaders = get_data_loaders_doc_extraction(args)
        model = get_model(args)
        optimizer1 = get_optimizer(model, args)
        scheduler1 = get_scheduler(optimizer1, args)
        logger.info(model)
        model.cuda()
        torch.save(args, os.path.join(args.savedir, "args.pt"))
        start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
        logger.info("Doc Extraction..")
        """
        Addestrare il modello come normale (nel branch master)
        Poi fare un caricamento dei checkpoint del modello
        poi per ogni documento confrontare con ogni query e prendere il massimo della similarità
        questo per validation e test
        """
        actual_epoch = 0
        model.train()
        for i_epoch in range(start_epoch,args.max_epochs):
            logger.warning("*"*50+" EPOCH "+str(i_epoch)+" "+"*"*50)
            optimizer1.zero_grad()
            model.zero_grad()
            iter_doc = iter(train_doc_loader)
            for batch_d in tqdm(iter_doc, total=len(train_doc_loader)):
                with torch.cuda.amp.autocast():
                    out_d,tgt = model_forward_feat(model, args, batch_d)
                optimizer1.step()
                optimizer1.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()   
            actual_epoch = i_epoch
        logger.info("*"*50+" Saving model to checkpoint for epoch : "+str(actual_epoch)+" "+"*"*50)
        save_checkpoint(
        {
            "epoch": actual_epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer1.state_dict(),
            "scheduler": scheduler1.state_dict(),
            "n_no_improve": n_no_improve,
            "best_metric": best_metric,
        },
        False,
        args.savedir,"doc_extraction_model.pt")
        
        logger.warning("*"*50+" VALIDATION "+"*"*50)
        logger.info("*"*50+" Loading model from checkpoint "+"*"*50)
        load_checkpoint(model, os.path.join(args.savedir, "doc_extraction_model.pt"))
        model.eval()
        train_losses = []
        val_total_similarities = []
        val_counterz = 0
        iter_doc = iter(val_doc_loader)
        for batch_d in tqdm(iter_doc, total=len(val_doc_loader)):
            val_similarities_res = []
            q_counterz = 0
            iter_query = iter(val_query_loader)
            for batch_q in tqdm(iter_query, total=len(val_query_loader)):
                with torch.cuda.amp.autocast():
                    out_d,tgt = model_forward_feat(model, args, batch_d)
                    out_q,_ = model_forward_feat(model, args, batch_q)
                score,res,loss = loss_obj(out_d,out_q,tgt)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                train_losses.append(loss.item())
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                val_similarities_res.append(score.cpu().detach().numpy())
                torch.cuda.empty_cache()
                gc.collect()
                q_counterz+=1
                if q_counterz == 100:
                    break
            #? Similarità qui tra singolo doc e query valutate (media?)
            val_total_similarities.append(np.nanmean(val_similarities_res))
            val_counterz+=1
            if val_counterz == 5:
                break
        logger.info("VALIDATION : Total similarity per DOCUMENT : ")
        print(val_total_similarities)
        logger.info("VALIDATION : Total similarity of ALL DOCUMENTS : "+str(np.nanmax(val_total_similarities)))
        
        
        logger.warning("*"*50+" TEST "+"*"*50)
        model.eval()
        train_losses = []
        test_total_similarities = []
        test_counterz = 0
        iter_doc = iter(test_doc_loaders)
        for batch_d in tqdm(iter_doc, total=len(test_doc_loaders)):
            test_similarities_res = []
            q_counterz = 0
            iter_query = iter(test_query_loaders)
            for batch_q in tqdm(iter_query, total=len(test_query_loaders)):
                with torch.cuda.amp.autocast():
                    out_d,tgt = model_forward_feat(model, args, batch_d)
                    out_q,_ = model_forward_feat(model, args, batch_q)
                score,res,loss = loss_obj(out_d,out_q,tgt)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                train_losses.append(loss.item())
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                test_similarities_res.append(score.cpu().detach().numpy())
                torch.cuda.empty_cache()
                gc.collect()
                q_counterz+=1
                if q_counterz == 100:
                    break
            #? Similarità qui tra singolo doc e query valutate (media?)
            test_total_similarities.append(np.nanmean(test_similarities_res))
            test_counterz+=1
            if test_counterz == 5:
                break
        logger.info("TEST : Total similarity per DOCUMENT : ")
        print(test_total_similarities)
        logger.info("TEST : Total similarity of ALL DOCUMENTS : "+str(np.nanmax(test_total_similarities)))