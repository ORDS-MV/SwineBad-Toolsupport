#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TensorMask Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.modeling import build_model



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args)
    default_setup(cfg, args)
    return cfg

def _build_model(cfg):
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir="pretrained_model")
    checkpoint = checkpointer.resume_or_load('pretrained_model/'+cfg.MODEL.WEIGHTS, resume=False)
    return model, checkpoint


