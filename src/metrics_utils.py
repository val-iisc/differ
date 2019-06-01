from __future__ import division
import os
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
import sys
import json
import argparse
import glob
import time
import csv
from tqdm import tqdm

import cv2
import numpy as np
import random
import re
import tensorflow as tf
import tflearn
from itertools import product
from scipy import misc

from metrics_dataloader import *
from blend_background import blendBg
from tf_auctionmatch import auction_match
import tf_nndistance

BASE_DIR = dirname(abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('../src')
sys.path.append('../src/utils_chamfer')


def fetch_batch_paths(models, indices, batch_num, batch_size):
    '''
    Returns image_ids for all models.

    '''
    paths = []
    for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
        model_path = models[ind[0]]
        try:
            line = model_path.strip().split('/')
            category_id = line[-2]
            model_id = line[-1]
            fid = '_'.join([line[-2], line[-1], str(ind[1])])
            paths.append(fid)
        except:
            print fid
            pass

    return paths


def fetch_batch_paths_seg(models, indices, batch_num, batch_size):
    '''
    Returns model_ids for all models.

    '''
    paths = []
    for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size]:
        model_path = models[ind]
        try:
            line = model_path.strip().split('/')
            category_id = line[-2]
            model_id = line[-1]
            fid = '_'.join([line[-2], line[-1]])
            paths.append(fid)
        except:
            print fid
            pass

    return paths


def get_rec_metrics(gt_pcl, pred_pcl, batch_size=10, num_points=1024):
    '''
    Calculate chamfer and emd metrics.

    args:
            gt_pcl: float, (BS,N_PTS,3); ground truth point cloud
            pred_pcl: float, (BS,N_PTS,3); predicted point cloud
    '''
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pcl,
                                                                    pred_pcl)
    dists_forward = tf.reduce_mean(tf.sqrt(dists_forward), axis=1)  # (BS,)
    dists_backward = tf.reduce_mean(tf.sqrt(dists_backward), axis=1)  # (BS,)
    chamfer_distance = dists_backward + dists_forward

    X, _ = tf.meshgrid(range(batch_size), range(num_points), indexing='ij')
    # Ind corresponds to points in pcl_gt
    ind, _ = auction_match(pred_pcl, gt_pcl)
    print X.get_shape()
    print ind.get_shape()
    ind = tf.stack((X, ind), -1)
    print gt_pcl.get_shape()
    print ind.get_shape()

    # (BS,N_PTS,3) --> (BS,N_PTS) --> (BS,)
    emd = tf.reduce_mean(tf.sqrt(tf.reduce_sum((
        tf.gather_nd(gt_pcl, ind) - pred_pcl)**2, axis=-1)), axis=1)

    return dists_forward, dists_backward, chamfer_distance, emd


def get_labels_seg(pcl_gt, pcl_pred, metric):
    '''
    Point wise correspondences between two point sets.

    args:
        pcl_gt: (batch_size, n_pts, 3), gt pcl
        pcl_pred: (batch_size, n_pts, 3), predicted pcl
        metric: str, 'chamfer' or 'emd'
                metric to be considered for returning corresponding
                points
    returns:
        pts_match_fwd: gt to pred point-wise correspondence
                       each point in gt is mapped to nearest point in
                       pred
        pts_match_bwd: pred to gt point-wise correspondence
                       each point in pred is mapped to nearest point in
                       gt
        pts_match: one-to-one mapping between pred and gt, acc. to emd
    '''
    if metric == 'chamfer':
        _, pts_match_fwd, _, pts_match_bwd = tf_nndistance.nn_distance(
            pcl_gt, pcl_pred)
        return pts_match_fwd, pts_match_bwd
    elif metric == 'emd':
        pts_match, _ = auction_match(pcl_gt, pcl_pred)
        return pts_match
    else:
        print 'Undefined metric'

        return None


def get_rgb_loss(rgb_gt, rgb_pred):
    '''
    Loss calculation for color prediction.

    MSE loss between GT and predicted point colors after caclultating
    forward and backward point correspondences.
    args:
        rgb_gt: (BS, N_PTS, 3), gt rgb values
        rgb_pred: (BS, N_PTS, 3), predicted rgb values
    returns:
        rgb_loss: (); mean rgb loss
        per_instance_rgb_loss: (BS,); instance-wise loss
        per_instance_rgb_pred_res: (BS, N_PTS, 3); predicted rgb values

    '''
    per_instance_rgb_loss = tf.reduce_mean((rgb_pred-rgb_gt)**2, axis=(1, 2))
    rgb_loss = tf.reduce_mean(per_instance_rgb_loss)
    per_instance_rgb_pred_res = rgb_pred

    return rgb_loss, per_instance_rgb_loss, per_instance_rgb_pred_res


def get_seg_loss(seg_gt, seg_pred):
    '''
    Loss calculation for part-segmentation prediction.

    Cross-entropy loss between GT and predicted point labels after
    caclultating forward and backward point correspondences.
    args:
        seg_gt: (BS, N_PTS,), gt class labels
        seg_pred: (BS, N_PTS, N_CLS), predicted class labels, one-hot
                                        encoded
    returns:
        seg_loss: (); mean cross-entropy loss
        per_instance_rgb_loss: (BS,); instance-wise loss
        per_instance_rgb_pred_res: (BS, N_PTS,); predicted class
                                                    labels

    '''
    per_instance_seg_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=seg_pred, labels=seg_gt), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)
    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res


def get_seg_metrics(gt_pcl, gt_labels, pred_pcl, pred_labels, batch_size=10,
                    num_classes=4, freq_wt=False):
    # cross entropy
    pts_match_fwd, pts_match_bwd = get_labels_seg(gt_pcl, pred_pcl, 'chamfer')
    pred_labels_match = tf.stack([tf.gather(
        pred_labels[k], pts_match_fwd[k], axis=0) for k in range(batch_size)],
        axis=0)
    gt_labels_match = tf.stack([tf.gather(
        gt_labels[k], pts_match_bwd[k], axis=0) for k in range(batch_size)],
        axis=0)
    seg_loss_fwd, per_inst_fwd, per_inst_fwd_lbl = get_seg_loss(
        pred_labels_match, gt_labels)
    seg_loss_bwd, per_inst_bwd, per_inst_bwd_lbl = get_seg_loss(
        pred_labels, gt_labels_match)
    per_instance_seg_loss = per_inst_fwd + per_inst_bwd
    seg_loss = (seg_loss_fwd + seg_loss_bwd)/2.

    # IoU
    labels_pred_idx = tf.argmax(pred_labels, axis=2)
    per_part_iou_fwd, per_part_iou_bwd = get_iou(
        gt_pcl, pred_pcl, gt_labels, labels_pred_idx, num_classes,
        'chamfer', freq_wt)  # dict
    per_instance_iou_fwd = get_mIoU(per_part_iou_fwd, freq_wt)
    per_instance_iou_bwd = get_mIoU(per_part_iou_bwd, freq_wt)
    per_instance_iou = (per_instance_iou_fwd + per_instance_iou_bwd)/2.

    return (per_instance_seg_loss, per_inst_fwd, per_inst_bwd,
            per_instance_iou, per_instance_iou_fwd, per_instance_iou_bwd)


def get_mIoU(iou_dict, freq_wt):
    '''
    Obtain mean IOU values from dictionary.

    Args:
        iou_dict: keys are [0...n_cls]; values are of dim (n_cls, bs)
    Returns:
        mIoU: (bs,)
    '''
    mIoU = []
    for key in sorted(iou_dict.keys()):
        mIoU.append(iou_dict[key])
    mIoU = tf.stack(mIoU)
    if not freq_wt:
        mIoU = tf.reduce_mean(mIoU, axis=0)

    return mIoU


def get_iou(pcl_gt, pcl_pred, labels_gt, labels_pred, n_cls, metric,
            freq_wt=False):
    '''
    Obtain iou metrics for part segmentation.

    Wrapper for calc_iou method. Calculates point correspondences before
    using the calc_iou method to obtain IOU metrics.
    '''
    batch_size, num_points = labels_gt.get_shape()
    if metric == 'chamfer':
        pts_match_fwd, pts_match_bwd = get_labels_seg(pcl_gt, pcl_pred,
                                                      'chamfer')
    elif metric == 'emd':
        pts_match = get_labels_seg(pcl_gt, pcl_pred, 'emd')

    idx, _ = tf.meshgrid(range(batch_size), range(num_points),
                         indexing='ij')
    labels_pred_match = tf.gather_nd(
        labels_pred, tf.stack([idx, pts_match_fwd], -1))
    labels_gt_match = tf.gather_nd(
        labels_gt, tf.stack([idx, pts_match_bwd], -1))

    iou_fwd, _dict = calc_iou(labels_gt, labels_pred_match, n_cls, freq_wt)
    iou_bwd, _ = calc_iou(labels_gt_match, labels_pred, n_cls, freq_wt)

    _dict['lbprm'] = labels_pred_match
    _dict['lbgtm'] = labels_gt_match

    return iou_fwd, iou_bwd


def calc_iou(labels_gt, labels_pred, n_cls, freq_wt=False):
    '''
    Calculate iou metrics for part segmentation.

    '''
    _dict = {}
    gt_onehot = {}
    pred_onehot = {}
    tp = {}
    union = {}
    iou = {}
    cls_cnt = []
    for cls in range(n_cls):
        gt_onehot[cls] = tf.to_float(tf.equal(labels_gt, cls))
        cls_cnt.append(tf.reduce_sum(gt_onehot[cls], axis=1))
        pred_onehot[cls] = tf.to_float(tf.equal(labels_pred, cls))
        tp[cls] = tf.reduce_sum(tf.to_float(tf.equal(
            gt_onehot[cls], pred_onehot[cls]))*pred_onehot[cls], axis=1)
        union[cls] = tf.reduce_sum(gt_onehot[cls], axis=1) +\
            tf.reduce_sum(pred_onehot[cls], axis=1) - tp[cls]
        if freq_wt:
            iou[cls] = cls_cnt[cls]*(tp[cls] / (1.*union[cls] + 1e-8))
        else:
            iou[cls] = tf.where(union[cls] > 0,
                                tp[cls] / (1.*union[cls]),
                                tf.ones_like(tp[cls]/(1.*union[cls]))
                                )
    tot_cnt = tf.reduce_sum(tf.stack(cls_cnt, axis=1), axis=1)
    if freq_wt:
        for cls in range(n_cls):
            iou[cls] = iou[cls] / tot_cnt

    _dict['gt'] = gt_onehot
    _dict['pred'] = pred_onehot
    _dict['tp'] = tp
    _dict['un'] = union
    _dict['iou'] = iou
    _dict['cnt'] = cls_cnt

    return iou, _dict


def get_averages(csv_path):
    column_sums = None
    with open(csv_path) as f:
        lines = f.readlines()[1:]
        rows_of_numbers = [map(float, line.split(';')[1:]) for line in lines]
        sums = map(sum, zip(*rows_of_numbers))
        averages = [sum_item / len(lines) for sum_item in sums]

        return averages


def load_previous_checkpoint(snapshot_folder, saver, sess, exp, snapshot):
    if snapshot == 'best':  # only seg training
        ckpt_path = join(exp, 'best', 'best')
        print ('loading ' + ckpt_path + '  ....')
        saver.restore(sess, ckpt_path)

    elif snapshot == 'best_emd':
        ckpt_path = join(exp, 'best_emd', 'best')
        print ('loading ' + ckpt_path + '  ....')
        saver.restore(sess, ckpt_path)

    elif snapshot == 'best_chamfer':
        ckpt_path = join(exp, 'best_chamfer', 'best')
        print ('loading ' + ckpt_path + '  ....')
        saver.restore(sess, ckpt_path)

    elif snapshot == 'best_seg':
        ckpt_path = join(exp, 'best_seg', 'best')
        print ('loading ' + ckpt_path + '  ....')
        saver.restore(sess, ckpt_path)

    elif snapshot == 'best_joint':
        ckpt_path = join(exp, 'best_joint', 'best')
        print ('loading ' + ckpt_path + '  ....')
        saver.restore(sess, ckpt_path)
    else:
        try:
            epoch_num = int(snapshot)
        except:
            print 'Check the snapshot entered'
            sys.exit(1)
        ckpt = tf.train.get_checkpoint_state(snapshot_folder)
        print snapshot_folder
        if ckpt is not None:
            ckpt_path = os.path.abspath(ckpt.model_checkpoint_path)
            ckpt_path = join(snapshot_folder, 'model-%d' % epoch_num)
            print ('loading '+ckpt_path + '  ....')
            saver.restore(sess, ckpt_path)
        else:
            print ckpt
            print 'Failed to load checkpoint'
            sys.exit(1)

    return


def save_screenshots(_gt_scaled, _pr_scaled, img, screenshot_dir, fid,
                     eval_set, ballradius, FLAGS, partseg=False,
                     gt_labels=None, pr_labels=None, seg=False):
    '''
    Save image and gif outputs from different view-points.

    '''
    if not seg:
        mask = np.sum(img, axis=-1, keepdims=True)
        mask = ((mask > 0).astype(img.dtype))*img.max()
        img = np.concatenate((img, mask), axis=-1)
        cv2.imwrite(join(screenshot_dir, '%s_%s_inp.png' %
                         (eval_set, fid)), img)

    if FLAGS.save_screenshots:
        # clock, front, anticlock, side, back, top
        xangles = np.array([-50, 0, 50, 90, 180, 0]) * np.pi / 180.
        yangles = np.array([20, 20, 20, 20, 20, 90]) * np.pi / 180.
        gts = []
        results = []
        for xangle, yangle in zip(xangles, yangles):
            gt_rot = show3d_balls.get2D(np_rotate(
                _gt_scaled, xangle=xangle, yangle=yangle), partseg=partseg,
                labels=gt_labels, ballradius=ballradius)
            result_rot = show3d_balls.get2D(np_rotate(
                _pr_scaled, xangle=xangle, yangle=yangle), partseg=partseg,
                labels=pr_labels, ballradius=ballradius)
            gts.append(gt_rot)
            results.append(result_rot)
        gt = np.concatenate(gts, 1)
        result = np.concatenate(results, 1)
        final = np.concatenate((gt, result), 0)
        mask = np.sum(final, axis=-1, keepdims=True)
        mask = ((mask > 0).astype(final.dtype))*final.max()
        final = np.concatenate((final, mask), axis=-1)
        cv2.imwrite(join(
            screenshot_dir, '%s_%s.png' % (eval_set, fid)), final)

    if FLAGS.save_gifs:
        import imageio
        xangles = np.linspace(0, 360, 30) * np.pi / 180.
        yangles = np.ones_like(xangles) * 20 * np.pi / 180.
        final = []
        for xangle, yangle in zip(xangles, yangles):
            gt_rot = np_rotate(_gt_scaled, xangle=xangle, yangle=yangle)
            gt_rot = show3d_balls.get2D(
                gt_rot, partseg=partseg, labels=gt_labels,
                ballradius=ballradius, background=(255, 255, 255),
                showsz=400)
            gt_rot = cv2.cvtColor(gt_rot, cv2.COLOR_BGR2RGB)

            result_rot = np_rotate(_pr_scaled, xangle=xangle, yangle=yangle)
            result_rot = show3d_balls.get2D(
                result_rot, partseg=partseg, labels=gt_labels,
                ballradius=ballradius, background=(255, 255, 255),
                showsz=400)
            result_rot = cv2.cvtColor(result_rot, cv2.COLOR_BGR2RGB)
            frame = np.concatenate((gt_rot, result_rot), 1)
            final.append(frame)
        imageio.mimsave(join(screenshot_dir, '%s_%s.gif' %
                             (eval_set, fid)), final, 'GIF', duration=0.7)

    return
