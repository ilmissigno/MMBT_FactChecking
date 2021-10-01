#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from transformers import BertModel

from mmbt.models.image import ImageEncoder

#! Class that implements MMBT Model
class ImageBertEmbeddings(nn.Module):
    """[ImageBertEmbeddings : Class that implements 
    Image Embeddings, this class convert image features in embeddings.]
    """
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultimodalBertEncoder(nn.Module):
    """
    [MultimodalBertEncoder : This Class implements the MMBT Model, that consists of
    Text Embeddings extracted from BERT, Image Features as input of Image Embedding Layer.
    The outputs of Text embeddings and Image embeddings are concatenated and encoded in a Bert Encoder
    and pooled in a Bert Pooler layer.]
    """
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        if args.task == "vsnli":
            ternary_embeds = nn.Embedding(3, args.hidden_sz)
            ternary_embeds.weight.data[:2].copy_(
                bert.embeddings.token_type_embeddings.weight
            )
            ternary_embeds.weight.data[2].copy_(
                bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)
            )
            self.txt_embeddings.token_type_embeddings = ternary_embeds

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
    
    def forward(self, input_txt_left,input_txt_right, attention_mask_left, attention_mask_right,segment_left,segment_right, input_img_left,input_img_right,neg_txt_right,neg_mask_right,neg_segment_right,neg_img_right):
        """[forward method: This is extended for Query, Doc and Negative Doc.
        The batch encoded as Query,Doc and Negative Doc is processed 
        simultaneously in the model, the same process is applied to each inputs.
        The pooled layer is applied to each encoded output (query, doc and negative doc).]

        Args:
            input_txt_left ([torch.Tensor]): [Text embeddings for Query]
            input_txt_right ([torch.Tensor]): [Text embeddings for Doc]
            attention_mask_left ([torch.Tensor]): [Attention Mask for Query]
            attention_mask_right ([torch.Tensor]): [Attention Mask for Doc]
            segment_left ([torch.Tensor]): [Embedding index for Query]
            segment_right ([torch.Tensor]): [Embedding index for Doc]
            input_img_left ([torch.Tensor]): [Image features for Query]
            input_img_right ([torch.Tensor]): [Image features for Doc]
            neg_txt_right ([torch.Tensor]): [Text embeddings for Negative Doc]
            neg_mask_right ([torch.Tensor]): [Attention Mask for Negative Doc]
            neg_segment_right ([torch.Tensor]): [Embedding index for Negative Doc]
            neg_img_right ([torch.Tensor]): [Image Features for Negative Doc]

        Returns:
            The pooled features of Query, Doc and Negative Doc
        """
        bsz_left = input_txt_left.size(0)
        bsz_right = input_txt_right.size(0)
        neg_bsz_right = neg_txt_right.size(0)
        attention_mask_left = torch.cat(
            [
                torch.ones(bsz_left, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask_left,
            ],
            dim=1,
        )
        attention_mask_right = torch.cat(
            [
                torch.ones(bsz_right, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask_right,
            ],
            dim=1,
        )
        neg_attention_mask_right = torch.cat(
            [
                torch.ones(neg_bsz_right, self.args.num_image_embeds + 2).long().cuda(),
                neg_mask_right,
            ],
            dim=1,
        )
        extended_attention_mask_left = attention_mask_left.unsqueeze(1).unsqueeze(2)
        extended_attention_mask_right = attention_mask_right.unsqueeze(1).unsqueeze(2)
        neg_extended_attention_mask_right = neg_attention_mask_right.unsqueeze(1).unsqueeze(2)
        extended_attention_mask_left = extended_attention_mask_left.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask_right = extended_attention_mask_right.to(
            dtype=next(self.parameters()).dtype
        )
        neg_extended_attention_mask_right = neg_extended_attention_mask_right.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask_left = (1.0 - extended_attention_mask_left) * -10000.0
        extended_attention_mask_right = (1.0 - extended_attention_mask_right) * -10000.0
        neg_extended_attention_mask_right = (1.0 - neg_extended_attention_mask_right) * -10000.0

        img_tok_left = (
            torch.LongTensor(input_txt_left.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        img_tok_right = (
            torch.LongTensor(input_txt_right.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        neg_img_tok_right = (
            torch.LongTensor(neg_txt_right.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        img_left = self.img_encoder(input_img_left)  # BxNx3x224x224 -> BxNx2048
        img_right = self.img_encoder(input_img_right)  # BxNx3x224x224 -> BxNx2048
        neg_img_right = self.img_encoder(neg_img_right)  # BxNx3x224x224 -> BxNx2048
        img_embed_out_left = self.img_embeddings(img_left, img_tok_left)
        img_embed_out_right = self.img_embeddings(img_right, img_tok_right)
        neg_img_embed_out_right = self.img_embeddings(neg_img_right, neg_img_tok_right)
        txt_embed_out_left = self.txt_embeddings(input_txt_left, segment_left)
        txt_embed_out_right = self.txt_embeddings(input_txt_right, segment_right)
        neg_txt_embed_out_right = self.txt_embeddings(neg_txt_right, neg_segment_right)
        encoder_input_left = torch.cat([img_embed_out_left, txt_embed_out_left], 1)  # Bx(TEXT+IMG)xHID
        encoder_input_right = torch.cat([img_embed_out_right, txt_embed_out_right], 1)  # Bx(TEXT+IMG)xHID
        neg_encoder_input_right = torch.cat([neg_img_embed_out_right, neg_txt_embed_out_right], 1)  # Bx(TEXT+IMG)xHID

        encoded_layers_left = self.encoder(
            encoder_input_left, extended_attention_mask_left
        )
        encoded_layers_right = self.encoder(
            encoder_input_right, extended_attention_mask_right
        )
        neg_encoded_layers_right = self.encoder(
            neg_encoder_input_right, neg_extended_attention_mask_right
        )
        
        
        return self.pooler(encoded_layers_left[-1]),self.pooler(encoded_layers_right[-1]),self.pooler(neg_encoded_layers_right[-1])

class MultimodalBertClf(nn.Module):
    """[MMBT Classifier : This applies model to each dataloader batches]
    """
    def __init__(self, args):
        super(MultimodalBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)

    def forward(self, txt_left,txt_right, mask_left,mask_right, segment_left,segment_right, img_left,img_right,neg_txt_right,neg_mask_right,neg_segment_right,neg_img_right):
        x,y,neg_y = self.enc(txt_left,txt_right, 
                             mask_left,mask_right, 
                             segment_left,segment_right, 
                             img_left,img_right,
                             neg_txt_right,neg_mask_right,
                             neg_segment_right,neg_img_right)
        return x,y,neg_y

