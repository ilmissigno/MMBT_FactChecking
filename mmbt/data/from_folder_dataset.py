import torch
import os
import pandas as pd
import functools
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from mmbt.data.dataset import TsvDatasetMulti
from mmbt.data.helpers import get_vocab, get_transforms, collate_fn_2

def getint(name):
    basename = name.split('.')
    alpha, num = basename[0].split('_')
    return int(num)

def get_documents_features_from_folder(folderpath):
    document_pt = []
    for root, dirs, files in os.walk(folderpath):
        for file in sorted(files,key=getint):
            subpath = os.path.join(root,file)
            tensor = torch.load(subpath)
            document_pt.append(tensor)
    
    document_pt = torch.stack(document_pt)
    data = torch.utils.data.TensorDataset(document_pt)
    data_loader = torch.utils.data.DataLoader(data,batch_size=1,shuffle=False,num_workers=0)
    return data_loader,document_pt

def save_documents_to_folder(batch, path, filename):
    filename = os.path.join(path, filename)
    torch.save(batch, filename)

def load_relevances(args):
    df = pd.read_csv(os.path.join(args.data_path, args.task, "annotations/"+args.type_dataset+"_query_article_interaction.csv"),header=None)
    relevances = {}
    for index, value in df.iterrows():
        if value[0] in relevances:
            relevances[value[0]].add(value[1])
        else:
            relevances[value[0]] = set()
            relevances[value[0]].add(value[1])
    return relevances

def get_queries_pack(args):
    df = pd.read_csv(os.path.join(args.data_path, args.task, "annotations/"+args.type_dataset+"_QueriesDataset.tsv"),sep="\t")
    return {val['QueryID']: val['QueryText'] for idx, val in df.iterrows()}

def get_documents_pack(args):
    df = pd.read_csv(os.path.join(args.data_path, args.task, "annotations/"+args.type_dataset+"_DocumentsDataset.tsv"),sep="\t")
    return {val['DocID']: val['DocText'] for idx, val in df.iterrows()}

def get_queries_dataloader(args):
    tokenizer = (
        AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert","mmbt_feat"]
        else str.split
    )
    transforms = get_transforms(args)
    vocab = get_vocab(args)
    queries = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/"+args.type_dataset+"_QueriesDataset.tsv"),
        'query',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    collate_2 = functools.partial(collate_fn_2, args=args)
    queries_dataloader = DataLoader(
        queries,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )
    return queries_dataloader

def get_documents_dataloader(args):
    tokenizer = (
        AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert","mmbt_feat"]
        else str.split
    )
    transforms = get_transforms(args)
    vocab = get_vocab(args)
    documents = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/"+args.type_dataset+"_DocumentsDataset.tsv"),
        'doc',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    collate_2 = functools.partial(collate_fn_2, args=args)
    documents_dataloader = DataLoader(
        documents,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )
    return documents_dataloader