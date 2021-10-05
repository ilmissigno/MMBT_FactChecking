import torch
from tqdm import tqdm
from mmbt.utils.utils import *
from mmbt.data.from_folder_dataset import *
from mmbt.forward import *
import gc
import time
from sentence_transformers.util import semantic_search
from mmbt.metrics.RankMetrics import evaluate_scoreMMBT

def doc_train(model,setting_dict,args,train_query_loader,train_doc_loader):
    start_epoch = setting_dict['start_epoch']
    logger = setting_dict['logger']
    optimizer1 = setting_dict['optimizer1']
    actual_epoch = 0
    model.train()
    logger.warning("*"*50+" DOCUMENT EMBEDDING : TRAINING PHASE "+"*"*50)
    for i_epoch in range(start_epoch,args.max_epochs):
        logger.warning("*"*50+" EPOCH "+str(i_epoch)+" "+"*"*50)
        iter_doc = iter(train_doc_loader)
        for batch_d in tqdm(iter_doc, total=len(train_doc_loader)):
            with torch.cuda.amp.autocast():
                out_d,tgt = model_forward_feat(model, args, batch_d)
            optimizer1.step()
            optimizer1.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
        actual_epoch = i_epoch
    return model,actual_epoch

def save_doc_extracted(model,args,doc_loader,path):
    iter_doc = iter(doc_loader)
    if not os.path.exists(path):
        os.makedirs(path)
        for n_batch, batch_d in enumerate(tqdm(iter_doc, total=len(doc_loader))):
            with torch.cuda.amp.autocast():
                out_d,_ = model_forward_feat(model,args,batch_d)
                save_documents_to_folder(out_d,path,"doc_"+str(n_batch)+".pt")
                torch.cuda.empty_cache()
                gc.collect()

def inference_docs(model,logger,args,query_loader,path):
    logger.warning("*"*50+" DOCUMENT EMBEDDING : TEST (LOADING FROM FOLDER) "+"*"*50)
    logger.info("*"*50+" TEST "+"*"*50)
    logger.info("*"*50+" Loading documents from folder (TEST SET) "+"*"*50)
    doc_dataload,doc_tensor = get_documents_features_from_folder(path)
    iter_query = iter(query_loader)
    maxes_sim_query_doc = {}
    times_per_query = {}
    mean_ndcgs = []
    mean_hits = []
    df_test = pd.read_csv(os.path.join(args.data_path, args.task, "annotations/test.tsv"),sep="\t")
    top_k = min(args.top_k_extract, len(doc_dataload))
    for n_batch_q, batch_q in enumerate(tqdm(iter_query,total=len(query_loader))):
        start_time = time.time()
        rankings = {}
        scores, docIDs, labels = [],[],[]
        with torch.cuda.amp.autocast():
            out_q,tgt = model_forward_feat(model, args, batch_q)
        hits = semantic_search(out_q, doc_tensor.squeeze(1), top_k=top_k)[0]    #Get the hits for the first query
        for hit in hits:
            scores.append(hit['score'])
            docIDs.append(hit['corpus_id'])
            labels.append(int(df_test[(df_test['DocID'] == hit['corpus_id'])]['Label'].iloc[0]))
            maxes_sim_query_doc[str(n_batch_q)+"_"+str(hit['corpus_id'])] = hit['score']
        rankings[n_batch_q] = (docIDs,labels,scores)
        results = evaluate_scoreMMBT(rankings,int(top_k))
        mean_ndcgs.append(results['ndcg'])
        end_time = time.time()
        times_per_query['time_query_'+str(n_batch_q)+'_top'+str(top_k)] = end_time-start_time
        mean_hits.append(results['hits'])
        torch.cuda.empty_cache()
        gc.collect()
    
    mean_ndcg = np.mean(mean_ndcgs)
    mean_hit = np.mean(mean_hits)
    return maxes_sim_query_doc,top_k,times_per_query,mean_ndcg,mean_hit