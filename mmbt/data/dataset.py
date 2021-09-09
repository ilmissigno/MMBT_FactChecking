#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import pandas as pd
import os
import cv2
import PIL
from PIL import Image
from random import randint

import torch
from torch.utils.data import Dataset

from mmbt.utils.utils import numpy_seed

class TsvDatasetMulti(Dataset):
    """ TSV Dataset Loading class
        This class help for loading a dataset from a tsv file.
    Args:
        Dataset : [Pytorch Dataset]
    """
    def __init__(self, data_path,multi, tokenizer, transforms, vocab, args):
        """[Constructor]

        Args:
            data_path ([string]): [Path to TSV Annotation dataset]
            multi ([string]): [Type of Query or Doc Dataset]
            tokenizer ([Tokenizer]): [Tokenizer Object]
            transforms ([Pytorch Transform]): [Pytorch Transform Object for Agumenting]
            vocab ([string]): [Path to vocab (use for other models)]
            args ([dict]): [config parameters]
        """
        self.data = pd.read_csv(data_path,sep="\t")
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.multi = multi
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """[Get current element in dataloader]

        Args:
            index ([int]): [Index of current element in dataloader]

        Returns:
            [current element in dataloader]
        """
        sentence_q = (
        self.text_start_token
        + self.tokenizer(self.data.loc[index,"QueryText"])[
            : (self.args.max_seq_len - 1)
        ]
        )
        sentence_d = (
        self.text_start_token
        + self.tokenizer(self.data.loc[index,"DocText"])[
            : (self.args.max_seq_len - 1)
        ]
        )
        
        #negative random document
        j_idx = randint(0,len(self.data)-1)
        neg_sentence_d = (
        self.text_start_token
        + self.tokenizer(self.data.loc[j_idx,"DocText"])[
            : (self.args.max_seq_len - 1)
        ]
        )
        segment_q = torch.zeros(len(sentence_q))
        segment_d = torch.zeros(len(sentence_d))
        neg_segment_d = torch.zeros(len(neg_sentence_d))
        sentence_q = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence_q
            ]
        )
        sentence_d = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence_d
            ]
        )
        neg_sentence_d = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in neg_sentence_d
            ]
        )
        label = torch.LongTensor(
            [self.args.labels.index(self.data.loc[index,"Label"])]
        )
        
        #! Image Loading, search by ID in folder (query and document)
        image_q = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            if self.data.loc[index,"QueryID"]:
                img_name_q = self.args.data_path+'/'+self.args.task+'/images/query/'+str(self.data.loc[index,'QueryID'])+".png"
                try:
                    image_q = Image.open(
                        img_name_q
                    ).convert("RGB")
                except PIL.UnidentifiedImageError:
                    image_q = cv2.imread(img_name_q)
                    image_q = cv2.cvtColor(image_q,cv2.COLOR_BGR2RGB)
            else:
                image_q = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image_q = self.transforms(image_q)
        
        image_d = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            if self.data.loc[index,"DocID"]:
                img_name_d = self.args.data_path+'/'+self.args.task+'/images/doc/'+str(self.data.loc[index,'DocID'])+".png"
                try:
                    image_d = Image.open(
                        img_name_d
                    ).convert("RGB")
                except PIL.UnidentifiedImageError:
                    image_d = cv2.imread(img_name_d)
                    image_d = cv2.cvtColor(image_d,cv2.COLOR_BGR2RGB)
            else:
                image_d = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image_d = self.transforms(image_d)
        
        neg_img_d = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            if self.data.loc[j_idx,"DocID"]:
                neg_img_name_d = self.args.data_path+'/'+self.args.task+'/images/doc/'+str(self.data.loc[j_idx,'DocID'])+".png"
                try:
                    neg_img_d = Image.open(
                        neg_img_name_d
                    ).convert("RGB")
                except PIL.UnidentifiedImageError:
                    neg_img_d = cv2.imread(neg_img_name_d)
                    neg_img_d = cv2.cvtColor(neg_img_d,cv2.COLOR_BGR2RGB)
            else:
                neg_img_d = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            neg_img_d = self.transforms(neg_img_d)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment_q = segment_q[1:]
            sentence_q = sentence_q[1:]
            # The first segment (0) is of images.
            segment_q += 1
            # The first SEP is part of Image Token.
            segment_d = segment_d[1:]
            sentence_d = sentence_d[1:]
            # The first segment (0) is of images.
            segment_d += 1
            neg_segment_d = neg_segment_d[1:]
            neg_sentence_d = neg_sentence_d[1:]
            # The first segment (0) is of images.
            neg_segment_d += 1
        
        return sentence_q, segment_q, image_q, sentence_d, segment_d, image_d, neg_sentence_d, neg_segment_d, neg_img_d, label