#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmbt.models.bert import BertClf
from mmbt.models.image import ImageClf
from mmbt.models.mmbt import MultimodalBertClf
from mmbt.models.mmbt_feat import MultimodalBertClf


MODELS = {
    "bert": BertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    "mmbt_feat": MultimodalBertClf,
}


def get_model(args):
    return MODELS[args.model](args)