#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import pandas as pd
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from mmbt.data.dataset import TsvDatasetMulti
from mmbt.data.vocab import Vocab


def get_transforms(args):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    #! Function for get the labels as a List
    label_freqs = Counter()
    df = pd.read_csv(path,sep="\t")
    data_labels = df['Label'].astype(int).tolist()
    data_labels = list(set(data_labels))
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


#? Unused
def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    #! Function for get Tokenizer from Transformers
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert"]:
        bert_tokenizer = AutoTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


#! Function of collate for create the batch from dataloader
def collate_fn(batch, args):
    lens_q = [len(row[0]) for row in batch]
    lens_d = [len(row[3]) for row in batch]
    lens_neg_d = [len(row[6]) for row in batch]
    bsz, max_seq_len_q, max_seq_len_d, max_seq_neg_len_d = len(batch), max(lens_q), max(lens_d), max(lens_neg_d)

    mask_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    mask_tensor_d = torch.zeros(bsz, max_seq_len_d).long()
    mask_tensor_neg_d = torch.zeros(bsz, max_seq_neg_len_d).long()
    text_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    text_tensor_d = torch.zeros(bsz, max_seq_len_d).long()
    text_tensor_neg_d = torch.zeros(bsz, max_seq_neg_len_d).long()
    segment_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    segment_tensor_d = torch.zeros(bsz, max_seq_len_d).long()
    segment_tensor_neg_d = torch.zeros(bsz, max_seq_neg_len_d).long()

    img_tensor_q = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor_q = torch.stack([row[2] for row in batch])
    
    img_tensor_d = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor_d = torch.stack([row[5] for row in batch])
    img_tensor_neg_d = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor_neg_d = torch.stack([row[8] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[9] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[9] for row in batch]).long()
    
    # sentence_q, segment_q, image_q, sentence_d, segment_d, image_d, neg_sentence_d, neg_segment_d, neg_img_d, label
    for i_batch, (input_row, length_q, length_d, length_neg_d) in enumerate(zip(batch, lens_q, lens_d, lens_neg_d)):
        tokens_q = input_row[0]
        segment_q = input_row[1]
        tokens_d = input_row[3]
        segment_d = input_row[4]
        neg_tokens_d = input_row[6]
        neg_segment_d = input_row[7]
        text_tensor_q[i_batch, :length_q] = tokens_q
        text_tensor_d[i_batch, :length_d] = tokens_d
        text_tensor_neg_d[i_batch, :length_neg_d] = neg_tokens_d
        segment_tensor_q[i_batch, :length_q] = segment_q
        segment_tensor_d[i_batch, :length_d] = segment_d
        segment_tensor_neg_d[i_batch, :length_neg_d] = neg_segment_d
        mask_tensor_q[i_batch, :length_q] = 1
        mask_tensor_d[i_batch, :length_d] = 1
        mask_tensor_neg_d[i_batch, :length_neg_d] = 1
    return text_tensor_q, segment_tensor_q, mask_tensor_q, img_tensor_q, text_tensor_d, segment_tensor_d, mask_tensor_d, img_tensor_d, text_tensor_neg_d, segment_tensor_neg_d, mask_tensor_neg_d, img_tensor_neg_d, tgt_tensor

def collate_fn_2(batch, args):
    lens_q = [len(row[0]) for row in batch]
    bsz, max_seq_len_q = len(batch), max(lens_q)
    mask_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    text_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    segment_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    img_tensor_q = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor_q = torch.stack([row[2] for row in batch])
    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()
    # sentence_q, segment_q, image_q, sentence_d, segment_d, image_d, neg_sentence_d, neg_segment_d, neg_img_d, label
    for i_batch, (input_row, length_q) in enumerate(zip(batch, lens_q)):
        tokens_q = input_row[0]
        segment_q = input_row[1]
        text_tensor_q[i_batch, :length_q] = tokens_q
        segment_tensor_q[i_batch, :length_q] = segment_q
        mask_tensor_q[i_batch, :length_q] = 1
    return text_tensor_q, segment_tensor_q, mask_tensor_q, img_tensor_q, tgt_tensor

#! Load dataloader from dataset class
def get_data_loaders_left_right(args):
    tokenizer = (
        AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert"]
        else str.split
    )

    transforms = get_transforms(args)
    
    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "annotations/train.tsv")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/train.tsv"),
        'query',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    dev = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/val.tsv"),
        'query',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    test_set = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/test.tsv"),
        'query',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    args.train_data_len = len(train)

    collate = functools.partial(collate_fn, args=args)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate,
        pin_memory=True
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
        pin_memory=True
    )

    return train_loader, val_loader,test_loader

def get_data_loaders_doc_extraction(args):
    tokenizer = (
        AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert"]
        else str.split
    )

    transforms = get_transforms(args)
    
    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "annotations/train.tsv")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train_q = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/train.tsv"),
        'query',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    train_d = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/train.tsv"),
        'doc',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    dev_q = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/val.tsv"),
        'query',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    dev_d = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/val.tsv"),
        'doc',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    test_set_q = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/test.tsv"),
        'query',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    test_set_d = TsvDatasetMulti(
        os.path.join(args.data_path, args.task, "annotations/test.tsv"),
        'doc',
        tokenizer,
        transforms,
        vocab,
        args,
    )
    
    args.train_data_len = len(train_d)


    collate_2 = functools.partial(collate_fn_2, args=args)

    train_loader_q = DataLoader(
        train_q,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )

    train_loader_d = DataLoader(
        train_d,
        batch_size=args.batch_sz,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )

    val_loader_q = DataLoader(
        dev_q,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )

    val_loader_d = DataLoader(
        dev_d,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )

    test_loader_q = DataLoader(
        test_set_q,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )

    test_loader_d = DataLoader(
        test_set_d,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_2,
        pin_memory=True
    )

    return train_loader_q,train_loader_d,val_loader_q,val_loader_d,test_loader_q,test_loader_d