#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import pandas as pd
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from mmbt.data.dataset import JsonlDataset, TsvDataset, TsvDatasetMulti
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


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
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


def collate_fn(batch, args):
    lens_q = [len(row[0]) for row in batch]
    lens_d = [len(row[3]) for row in batch]
    bsz, max_seq_len_q, max_seq_len_d = len(batch), max(lens_q), max(lens_d)

    mask_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    mask_tensor_d = torch.zeros(bsz, max_seq_len_d).long()
    text_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    text_tensor_d = torch.zeros(bsz, max_seq_len_d).long()
    segment_tensor_q = torch.zeros(bsz, max_seq_len_q).long()
    segment_tensor_d = torch.zeros(bsz, max_seq_len_d).long()

    img_tensor_q = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor_q = torch.stack([row[2] for row in batch])
    
    img_tensor_d = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor_d = torch.stack([row[5] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[6] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[6] for row in batch]).long()

    for i_batch, (input_row, length_q, length_d) in enumerate(zip(batch, lens_q, lens_d)):
        tokens_q = input_row[0]
        segment_q = input_row[1]
        tokens_d = input_row[3]
        segment_d = input_row[4]
        text_tensor_q[i_batch, :length_q] = tokens_q
        text_tensor_d[i_batch, :length_d] = tokens_d
        segment_tensor_q[i_batch, :length_q] = segment_q
        segment_tensor_d[i_batch, :length_d] = segment_d
        mask_tensor_q[i_batch, :length_q] = 1
        mask_tensor_d[i_batch, :length_d] = 1
    return text_tensor_q, segment_tensor_q, mask_tensor_q, img_tensor_q, text_tensor_d, segment_tensor_d, mask_tensor_d, img_tensor_d, tgt_tensor


def get_data_loaders(args):
    tokenizer = (
        AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert"]
        else str.split
    )

    transforms = get_transforms(args)
    
    if args.data_type.startswith('tsv'):
        args.labels, args.label_freqs = get_labels_and_frequencies(
            os.path.join(args.data_path, args.task, "annotations/train.tsv")
        )
        vocab = get_vocab(args)
        args.vocab = vocab
        args.vocab_sz = vocab.vocab_sz
        args.n_classes = len(args.labels)

        train = TsvDataset(
            os.path.join(args.data_path, args.task, "annotations/train.tsv"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        dev = TsvDataset(
            os.path.join(args.data_path, args.task, "annotations/val.tsv"),
            tokenizer,
            transforms,
            vocab,
            args,
        )
        test_set = TsvDataset(
            os.path.join(args.data_path, args.task, "annotations/test.tsv"),
            tokenizer,
            transforms,
            vocab,
            args,
        )
    else:
        args.labels, args.label_freqs = get_labels_and_frequencies(
            os.path.join(args.data_path, args.task, "train.jsonl")
        )
        vocab = get_vocab(args)
        args.vocab = vocab
        args.vocab_sz = vocab.vocab_sz
        args.n_classes = len(args.labels)

        train = JsonlDataset(
            os.path.join(args.data_path, args.task, "train.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        dev = JsonlDataset(
            os.path.join(args.data_path, args.task, "dev.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )
        test_set = JsonlDataset(
            os.path.join(args.data_path, args.task, "test.jsonl"),
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
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )


    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    # if args.task == "vsnli":
    #     test_hard = JsonlDataset(
    #         os.path.join(args.data_path, args.task, "test_hard.jsonl"),
    #         tokenizer,
    #         transforms,
    #         vocab,
    #         args,
    #     )

    #     test_hard_loader = DataLoader(
    #         test_hard,
    #         batch_size=args.batch_sz,
    #         shuffle=False,
    #         num_workers=args.n_workers,
    #         collate_fn=collate,
    #     )

    #     test = {"test": test_loader, "test_hard": test_hard_loader}

    # else:
    #     test_gt = JsonlDataset(
    #         os.path.join(args.data_path, args.task, "test_hard_gt.jsonl"),
    #         tokenizer,
    #         transforms,
    #         vocab,
    #         args,
    #     )

    #     test_gt_loader = DataLoader(
    #         test_gt,
    #         batch_size=args.batch_sz,
    #         shuffle=False,
    #         num_workers=args.n_workers,
    #         collate_fn=collate,
    #     )

    test = {
        "test": test_loader,
    }

    return train_loader, val_loader, test


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
