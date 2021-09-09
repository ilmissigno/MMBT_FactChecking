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
from mmbt.data.helpers import get_data_loaders_left_right
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
