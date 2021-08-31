#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from sklearn.metrics import f1_score, accuracy_score,ndcg_score,average_precision_score,top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import configargparse
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import sys
import gc
from random import randint
from pytorch_pretrained_bert import BertAdam
sys.path.append("")
import itertools
from matchzoo.metrics import normalized_discounted_cumulative_gain as ndcg
from mmbt.data.helpers import get_data_loaders,get_data_loaders_left_right,get_data_loaders_new
from mmbt.models import get_model
from mmbt.utils.logger import create_logger
from mmbt.utils.utils import *
from mmbt.losses.CrossSimilarity import CrossSimilarity
from mmbt.losses.CrossSimilarity_feat import CrossSimilarity_feat
from mmbt.metrics.RankMetrics import *
from mmbt.metrics.RankMetrics import ndcg as normalized_dcg


def get_args(parser):
    parser.add("--batch_sz", type=int, default=128)
    parser.add("--bert_model", type=str, default="bert-base-uncased")
    parser.add("--image_model", type=str, default="resnet152")
    parser.add("--data_path", type=str, default="/path/to/data_dir/")
    parser.add("--drop_img_percent", type=float, default=0.0)
    parser.add("--dropout", type=float, default=0.1)
    parser.add("--embed_sz", type=int, default=300)
    parser.add("--freeze_img", type=int, default=0)
    parser.add("--freeze_txt", type=int, default=0)
    parser.add("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add("--gradient_accumulation_steps", type=int, default=24)
    parser.add("--evaluation_steps", type=int, default=24)
    parser.add("--hidden", nargs="*", type=int, default=[])
    parser.add("--hidden_sz", type=int, default=768)
    parser.add("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add("--img_hidden_sz", type=int, default=2048)
    parser.add("--include_bn", default=True)
    parser.add("--lr", type=float, default=1e-4)
    parser.add("--lr_factor", type=float, default=0.5)
    parser.add("--lr_patience", type=int, default=2)
    parser.add("--max_epochs", type=int, default=100)
    parser.add("--max_seq_len", type=int, default=512)
    parser.add("--model", type=str, default="bow")
    parser.add("--n_workers", type=int, default=12)
    parser.add("--name", type=str, default="nameless")
    parser.add("--num_image_embeds", type=int, default=1)
    parser.add("--patience", type=int, default=10)
    parser.add("--savedir", type=str, default="/path/to/save_dir/")
    parser.add("--data_type", type=str, default="jsonl")
    parser.add("--seed", type=int, default=123)
    parser.add("--task", type=str, default="mmimdb")
    parser.add("--task_type", type=str, default="multilabel", choices=["multilabel", "classification","extraction"])
    parser.add("--warmup", type=float, default=0.1)
    parser.add("--weight_classes", type=int, default=1)
    parser.add("--doc_extraction", default=False)
    

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
        )
        # optimizer = optim.Adadelta(model.parameters(),lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args,loss_obj, store_preds=False):
    with torch.no_grad():
        losses, preds,tgts = [], [], []
        ndcg_list,ndcg_list_at_1, hit_list, map_list, map_list_at_1 = [], [], [], [], []
        for batch in tqdm(data, total=len(data)):
            loss,out,tgt = model_forward(i_epoch,model,args,loss_obj,batch)
            losses.append(loss.item())
            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
                pred_no_npy = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach()
                # pred_right = torch.nn.functional.softmax(out_r, dim=1).argmax(dim=1).cpu().detach().numpy()
            tgt = tgt.cpu().detach()
            ndcg_list.append(normalized_dcg(pred_no_npy,tgt,ats=[5]).numpy())
            ndcg_list_at_1.append(normalized_dcg(pred_no_npy,tgt,ats=[3]).numpy())
            map_list.append(average_precision(pred))
            map_list_at_1.append(average_precision(pred))
            hit_list.append(precision_at_k(pred,5))

        metrics = {"loss": np.nanmean(losses)}
        if args.task_type == "multilabel":
            tgts = np.vstack(tgts)
            preds = np.vstack(preds)
            metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
            metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
        else:
            tgts = [l for sl in tgts for l in sl]
            preds = [l for sl in preds for l in sl]
            metrics["ndcg"] = np.nanmean(ndcg_list)
            metrics["ndcg_1"] = np.nanmean(ndcg_list_at_1)
            metrics["acc"] = np.nanmean(hit_list)
            metrics["prec5"] = np.nanmean(map_list)
            metrics["prec1"] = np.nanmean(map_list_at_1)

        if store_preds:
            store_preds_to_disk(tgts, preds, args)

        return metrics


def model_forward(i_epoch, model, args,loss_obj, batch):
    txt_left, segment_left, mask_left, img_left,txt_right, segment_right, mask_right, img_right,neg_txt_right, neg_segment_right, neg_mask_right, neg_img_right, tgt = batch
    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
    elif args.model == "concatbow":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    elif args.model == "concatbert":
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)
    else:
        assert args.model == "mmbt" or args.model == "mmbt_cpu"
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt

        txt_left, img_left = txt_left.cuda(), img_left.cuda()
        txt_right, img_right = txt_right.cuda(), img_right.cuda()
        mask_left, segment_left = mask_left.cuda(), segment_left.cuda()
        mask_right, segment_right = mask_right.cuda(), segment_right.cuda()
        neg_txt_right,neg_mask_right,neg_segment_right,neg_img_right = neg_txt_right.cuda(), neg_mask_right.cuda(), neg_segment_right.cuda(),neg_img_right.cuda()
        out_l, out_r, out_neg_r = model(txt_left,txt_right,mask_left,mask_right,segment_left,segment_right,img_left,img_right, neg_txt_right,neg_mask_right,neg_segment_right,neg_img_right)
    
    tgt = tgt.cuda()
    res,loss = loss_obj(out_l,out_r, tgt, out_neg_r)
    return loss,res,tgt    

def model_forward_feat(i_epoch, model, args, batch):
    txt, segment, mask, img, tgt = batch
    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    assert args.model == "mmbt" or args.model == "mmbt_cpu"
    for param in model.enc.img_encoder.parameters():
        param.requires_grad = not freeze_img
    for param in model.enc.encoder.parameters():
        param.requires_grad = not freeze_txt
    
    txt, img = txt.cuda(), img.cuda()
    mask, segment = mask.cuda(), segment.cuda()
    tgt = tgt.cuda()
    out = model(txt, mask, segment, img)
    
    # if not os.path.exists('./features/{}'.format(args.name)):
    #     os.makedirs('./features/{}'.format(args.name))
    # i = 1
    # outputs_splitted = torch.tensor_split(out,args.batch_sz)
    # for elem in outputs_splitted:
    #     pickle.dump(elem,open("./features/{}/feature".format(args.name)+"_"+str(lab_loader)+"_"+str(i_epoch)+"_"+str(i_batch)+"_"+str(i)+".pkl",'wb'))
    #     i=i+1
    return out, tgt

def train(args):
    if args.doc_extraction:
        loss_obj = CrossSimilarity_feat()
    else:
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
    if args.doc_extraction:
        train_query_loader, train_doc_loader, val_query_loader, val_doc_loader, test_query_loaders, test_doc_loaders = get_data_loaders_new(args)
        settings_dict = {
            'model':model,
            'optimizer':optimizer1,
            'criterion': loss_obj,
            'scheduler':scheduler1,
            'logger':logger,
            'start_epoch':start_epoch,
            'global_step':global_step,
            'train_loader_q':train_query_loader,
            'train_loader_d':train_doc_loader,
            'val_loader_q':val_query_loader,
            'val_loader_d':val_doc_loader,
            'test_loaders_q':test_query_loaders,
            'test_loaders_d':test_doc_loaders,
            'best_metric':best_metric,
            'n_no_improve':n_no_improve
        }
        doc_feat_extraction(args,settings_dict)
    else:
        settings_dict = {
            'model':model,
            'optimizer':optimizer1,
            'criterion': loss_obj,
            'scheduler':scheduler1,
            'logger':logger,
            'start_epoch':start_epoch,
            'global_step':global_step,
            'train_loader':train_loader,
            'val_loader':val_loader,
            'test_loaders':test_loaders,
            'best_metric':best_metric,
            'n_no_improve':n_no_improve
        }
        train_phase_multi(args,settings_dict)


def cli_main():
    parser = configargparse.ArgParser(default_config_files=['configurazione.conf'])

    parser.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


def train_phase_single(args, settings_dict):
    model = settings_dict['model']
    logger = settings_dict['logger']
    train_loader = settings_dict['train_loader']
    val_loader = settings_dict['val_loader']
    test_loaders = settings_dict['test_loaders']
    criterion = settings_dict['criterion']
    optimizer = settings_dict['optimizer']
    scheduler = settings_dict['scheduler']
    start_epoch = settings_dict['start_epoch']
    global_step = settings_dict['global_step']
    best_metric = settings_dict['best_metric']
    n_no_improve = settings_dict['n_no_improve']
    if args.task_type == 'extraction':
        logger.info("Feature Extraction Training...")
        for i_batch, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            model_forward_feat("train",i_batch,0, model, args, criterion, batch)
        logger.info("Feature Extraction Validation...")
        for i_batch, batch in enumerate(tqdm(val_loader, total=len(val_loader))):
            model_forward_feat("val",i_batch,0, model, args, criterion, batch)
        logger.info("Feature Extraction Test...")
        for test_name, test_loader in test_loaders.items():
            for i_batch, batch in enumerate(tqdm(test_loader, total=len(test_loader))):
                model_forward_feat("test",i_batch,0, model, args, criterion, batch)
    else:
        logger.info("Training..")
        for i_epoch in range(start_epoch, args.max_epochs):
            train_losses = []
            model.train()
            optimizer.zero_grad()
            
            for i_batch, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
                loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                train_losses.append(loss.item())
                loss.backward()
                global_step += 1
                if global_step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            model.eval()
            logger.info("Validation...")
            metrics = model_eval(i_epoch, val_loader, model, args, criterion)
            logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
            log_metrics("Val", metrics, args, logger)

            tuning_metric = (
                metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
            )
            scheduler.step(tuning_metric)
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
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_no_improve": n_no_improve,
                    "best_metric": best_metric,
                },
                is_improvement,
                args.savedir,
            )

            if n_no_improve >= args.patience:
                logger.info("No improvement. Breaking out of loop.")
                break

        model.eval()
        logger.info("Test...")
        for test_name, test_loader in test_loaders.items():
            load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
            test_metrics = model_eval(np.inf, test_loader, model, args, criterion, store_preds=True)
            log_metrics(f"Test - {test_name}", test_metrics, args, logger)

def train_phase_multi(args, settings_dict):
    model = settings_dict['model']
    optimizer1 = settings_dict['optimizer']
    scheduler1 = settings_dict['scheduler']
    loss_obj = settings_dict['criterion']
    logger = settings_dict['logger']
    start_epoch = settings_dict['start_epoch']
    global_step = settings_dict['global_step']
    train_loader = settings_dict['train_loader']
    val_loader = settings_dict['val_loader']
    test_loaders = settings_dict['test_loaders']
    best_metric = settings_dict['best_metric']
    n_no_improve = settings_dict['n_no_improve']
    logger.info("Training..")
    total_steps = (
            len(train_loader)
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
    for i_epoch in range(start_epoch,args.max_epochs):
        logger.warning("*"*50+" EPOCH "+str(i_epoch)+" "+"*"*50)
        train_losses = []
        model.train()
        optimizer1.zero_grad()
        model.zero_grad()
        counterz = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            with torch.cuda.amp.autocast():
                loss,_,_ = model_forward(i_epoch,model,args,loss_obj,batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            train_losses.append(loss.item())
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            counterz+=1
            if counterz == 1000:
                break
        model.eval()
        logger.info("Validation...")
        metrics = model_eval(i_epoch, val_loader, model,args,loss_obj)
        log_metrics("Val", metrics, args, logger)
        # logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["ndcg"]
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

    model.eval()
    logger.info("Test...")
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    test_metrics = model_eval(np.inf, test_loaders,model,args,loss_obj, store_preds=True)
    log_metrics(f"Test - ", test_metrics, args, logger)


def doc_feat_extraction(args, settings_dict):
    model = settings_dict['model']
    optimizer1 = settings_dict['optimizer']
    scheduler1 = settings_dict['scheduler']
    loss_obj = settings_dict['criterion']
    logger = settings_dict['logger']
    start_epoch = settings_dict['start_epoch']
    global_step = settings_dict['global_step']
    train_query_loader = settings_dict['train_loader_q']
    train_doc_loader = settings_dict['train_loader_d']
    val_query_loader = settings_dict['val_loader_q']
    val_doc_loader = settings_dict['val_loader_d']
    test_query_loaders = settings_dict['test_loaders_q']
    test_doc_loaders = settings_dict['test_loaders_d']
    best_metric = settings_dict['best_metric']
    n_no_improve = settings_dict['n_no_improve']
    logger.info("Doc Extraction..")
    """
    Addestrare il modello come normale (nel branch master)
    Poi fare un caricamento dei checkpoint del modello
    poi per ogni documento confrontare con ogni query e prendere il massimo della similarità
    questo per validation e test
    """
    model.train()
    train_counterz = 0
    for i_epoch in range(start_epoch,args.max_epochs):
        logger.warning("*"*50+" EPOCH "+str(i_epoch)+" "+"*"*50)
        optimizer1.zero_grad()
        model.zero_grad()
        iter_doc = iter(train_doc_loader)
        d_counterz=0
        for batch_d in tqdm(iter_doc, total=len(train_doc_loader)):
            with torch.cuda.amp.autocast():
                out_d,tgt = model_forward_feat(i_epoch, model, args, batch_d)
            optimizer1.step()
            optimizer1.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            d_counterz+=1
            if d_counterz == 100:
                break
        train_counterz+=1
        if train_counterz == 3:
            break
    logger.warning("*"*50+" VALIDATION "+"*"*50)
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
                out_d,tgt = model_forward_feat(i_epoch, model, args, batch_d)
                out_q,_ = model_forward_feat(i_epoch, model, args, batch_q)
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
        if val_counterz == 3:
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
                out_d,tgt = model_forward_feat(i_epoch, model, args, batch_d)
                out_q,_ = model_forward_feat(i_epoch, model, args, batch_q)
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
        if test_counterz == 3:
            break
    logger.info("TEST : Total similarity per DOCUMENT : ")
    print(test_total_similarities)
    logger.info("TEST : Total similarity of ALL DOCUMENTS : "+str(np.nanmax(test_total_similarities)))
    
    
    


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    cli_main()
