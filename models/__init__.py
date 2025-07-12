# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from .backbone import build_backbone
from .deformable_detr import DeformableDETR, SetCriterion as DefSetCriterion, PostProcess as DefPostProcess
from .detr import DETR, SetCriterion as DETRSetCriterion, PostProcess as DETRPostProcess
from .dab_detr import DABDETR, SetCriterion as DABSetCriterion, PostProcess as DABPostProcess
from .rt_detr import RTDETR, SetCriterion as RTSetCriterion, RTDETRPostProcessor
from .def_matcher import build_matcher as build_def_matcher
from .detr_matcher import build_matcher as build_detr_matcher
from .dab_matcher import build_matcher as build_dab_matcher
from .rt_matcher import build_matcher as build_rt_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .dab_transformer import build_dab_transformer
from .transformer import build_transformer
from .rt_encoder import build_encoder
from .rt_decoder import build_decoder


def build_model(args):
    if args.dataset_file == 'coco':
        num_classes = 1
    elif args.dataset_file == 'coco_panoptic':
        num_classes = 1
    elif args.dataset_file == 'airbus':
        num_classes = 1
    else:
        num_classes = 1
    num_classes += 1
    device = torch.device(args.device)

    weight_dict = {'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef}
    if args.model != 'rt_detr':
        weight_dict['loss_ce']=args.cls_loss_coef
    else:
        weight_dict['loss_vfl'] = args.vfl_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})

        # only in def detr impl.
        if args.model == 'deformable_detr':
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['boxes', 'cardinality']
    if args.model != 'rt_detr':
        losses.append('labels')
    else:
        losses.append('vfl')
    if args.object_embedding_loss:
        losses.append('object_embedding')
        weight_dict['loss_object_embedding'] = args.object_embedding_coef

    if args.masks:
        losses += ["masks"]

    
    if args.model != 'rt_detr':
        backbone = build_backbone(args)

    if args.model == 'deformable_detr':
        transformer = build_deforamble_transformer(args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head
        )
        matcher = build_def_matcher(args)
        criterion = DefSetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
        postprocessors = {'bbox': DefPostProcess()}
    elif args.model == 'detr':
        transformer = build_transformer(args)
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head
        )
        matcher = build_detr_matcher(args)
        criterion = DETRSetCriterion(num_classes, matcher, weight_dict, args.eos_coef,
                                     losses, object_embedding_loss=args.object_embedding_loss)
        postprocessors = {'bbox': DETRPostProcess()}
    elif args.model == 'dab_detr':
        transformer = build_dab_transformer(args)
        model = DABDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_dec_layers=args.dec_layers,
            aux_loss=args.aux_loss,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head
        )
        matcher = build_dab_matcher(args)
        criterion = DABSetCriterion(num_classes, matcher, weight_dict, args.focal_alpha,
                                     losses, object_embedding_loss=args.object_embedding_loss)
        postprocessors = {'bbox': DABPostProcess(num_select=300)}
    elif args.model == 'rt_detr':
        backbone = build_backbone(args)
        encoder = build_encoder(args)
        decoder = build_decoder(args)
        model = RTDETR(
            backbone,
            encoder,
            decoder,
            object_embedding_loss=args.object_embedding_loss,
            obj_embedding_head=args.obj_embedding_head,
            hidden_dim=args.hidden_dim,
            multi_scale=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
        )
        matcher = build_rt_matcher(args)
        criterion = RTSetCriterion(matcher, weight_dict,losses, eos_coef=args.eos_coef, num_classes=num_classes,
                                 object_embedding_loss=args.object_embedding_loss)
        postprocessors = {'bbox': RTDETRPostProcessor(num_classes=num_classes)}
    else:
        raise ValueError("Wrong model.")

    criterion.to(device)

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
