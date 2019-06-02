import os
from os.path import join
import sys

import numpy as np
from itertools import product
import cv2
import scipy.io as sio

from shapenet_taxonomy import shapenet_category_to_id

sys.path.append('src')


def read_image(img_path, normalize=True):
    '''
    Read grayscale and color images from directory.

    args:
        img_path: str; path for input image to be read
        normalize: Bool, (); True if image has to be normalized to 
                                [0, 1] range
    returns:
        img_rgb: float; output image, in rgb format if color image
    '''
    img = cv2.imread(rgb_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = np.astype(img_rgb)
    if normalize:
        img_rgb = img_rgb / 255.

    return img_rgb


def read_angles(angles_path, to_rad=True):
    '''
    Read values of pose angles from file, convert to radians.

    args: 
        angles_path: str; path for file from which angles are read
        normalize: Bool, (); True if angles have to be converted to 
                                radians
    returns:
        angle_x, angle_y: float, (); rotation angles wrt x and y axes
    '''
    with open(angles_path, 'r') as fp:
        angles = [item.split('\n')[0] for item in fp.readlines()]
    angle = angles[i % 10]
    angle_x = float(angle.split(' ')[0])
    angle_y = float(angle.split(' ')[1])
    # Convert from degrees to radians
    if to_rad:
        angle_x = angle_x * np.pi/180.
        angle_y = angle_y * np.pi/180.

    return angle_x, angle_y


def fetch_batch_drc(models, indices, batch_num, batch_size, args=None):
    '''
    Obtain batch data for training.

    args:
        models: list of all ids/names of input image models
        indices: indices to be chosen from models for the current
                    batch
        batch_num: index of the current batch
        args: input arguments while running the train.py file
    returns:
        All outputs are lists of length N_VIEWS.
        Properties of each element in the list:
        batch_ip: uint8, (BS,IMG_H,IMG_W,3); input rgb images
        batch_gt: float, (BS,IMG_H,IMG_W); GT foreground masks
        batch_gt_rgb: float, (BS,IMG_H,IMG_W,3); GT rgb images from
                                different views
        batch_names: str; names of pcl models corresponding to
                            input images
        batch_x: float, (); rotation angle along x-axis for the
                                view point in radians
        batch_y: float, (); rotation angle along y-axis for the
                                view point in radians
    '''
    batch_ip, batch_gt, batch_x, batch_y, batch_names = [[] for i in range(5)]
    batch_data = [batch_gt, batch_x, batch_y]
    if args.rgb:
        batch_rgb = []
        batch_data.append(batch_rgb)
    if args.partseg:
        batch_partseg = []
        batch_data.append(batch_partseg)

    for ind in indices[(batch_num*batch_size):
                       (batch_num*batch_size + batch_size)]:
        model_gt, model_x, model_y = [[] for i in range(3)]
        model_data = [model_gt, model_x, model_y]
        if args.rgb:
            model_rgb = []
            model_data.append(model_rgb)
        if args.partseg:
            model_partseg = []
            model_data.append(model_partseg)
        model_path = models[ind[0]]
        model_name = model_path.split('/')[-1]
        img_path = join(model_path, 'render_%d.png' % ind[1])

        ip_image = read_image(img_path)
        ip_image = cv2.resize(ip_image, (args.IMG_W, args.IMG_H))
        batch_ip.append(ip_image)
        batch_names.append(model_name)


        for i in range(0, args.N_VIEWS):
            if args.CORR:
                # To obtain outputs corresponding to input view
                i = ind[1]
            proj_path = join(model_path, 'depth_%d.png' % (i % 10))
            rgb_path = join(model_path, 'render_%d.png' % (i % 10))
            view_path = join(model_path, 'camera_%d.mat' % (i % 10))
            angles_path = join(model_path, 'view.txt')
            ip_proj = read_image(proj_path, normalize=False)[:, :, 0]
            # Read rotation angles in radians from file
            angle_x, angle_y = read_angles(angles_path)
            model_vals = [ip_proj, angle_x, angle_y]
            if args.rgb:
                # Read inputs and normalize to [0,1] range
                ip_rgb = read_image(rgb_path, normalize=True)
                model_vals.append(ip_rgb)
            if args.partseg:
                labels_path = join(model_path, 'labels_%d.npy' % (
                    i % 10))
                proj_labels = np.load(labels_path)
                model_vals.append(proj_labels)
            
            # Append the data from multiple views into the instance-wise list
            _ = [model_item.append(vals_item) for model_item, vals_item in
                 zip(model_data, model_vals)]

        # Append the instance-wise data into the batch list
        _ = [batch_item.append(model_item) for batch_item, model_item in
             zip(batch_data, model_data)]
    batch_out = [batch_ip, batch_rgb, batch_partseg, batch_gt,
                 batch_x, batch_y]
    batch_out = [np.array(item) for item in batch_out]
    batch_out.append(batch_names)

    return batch_out


def fetch_labels(model_path):
    '''
    Obtain part segmentation class labels for point cloud.

    args:
        model_path: str; directory of the pcl model
    returns:
        labels_gt: int, (N_PTS); class labels
    '''
    label_path = join(model_path, 'pointcloud_labels.npy')
    labels_gt = np.load(label_path)
    labels_gt -= 1

    return labels_gt


def fetch_labels_rgb(model_path):
    '''
    Obtain RGB color values for point cloud.

    args:
        model_path: str; directory of the pcl model
    returns:
        colors_gt: int, (N_PTS,3); rgb values
    '''
    pcl_filename = 'pcl_1024_fps_trimesh_colors.npy'
    pcl_path = join(model_path, pcl_filename)
    colors_gt = np.load(pcl_path)
    colors_gt = colors_gt[:, 3:]

    return colors_gt


def fetch_batch_pcl_rgb(models, indices, batch_num, batch_size):
    batch_ip = []
    batch_gt = []
    batch_rgb = []
    for ind in indices[(batch_num*batch_size):
                       (batch_num*batch_size + batch_size)]:
        model_path = models[ind[0]]
        pcl_path = join(model_path, 'pcl_1024_fps_trimesh.npy')
        pcl_gt = np.load(pcl_path)
        pcl_rgb = fetch_labels_rgb(model_path)
        rotate = False
        if rotate:
            metadata_path = join(
                model_path, 'rendering', 'rendering_metadata.txt'
                )
            with open(metadata_path, 'r') as f:
                metadata = f.readlines()
                metadata = [i.strip() for i in metadata]
                x = [float(i.split(' ')[0]) for i in metadata]
                y = [float(i.split(' ')[1]) for i in metadata]
                xangle = np.pi/180. * x[ind[1]]
                yangle = np.pi/180. * y[ind[1]]
                pcl_gt = rotate(pcl_gt, xangle=xangle, yangle=yangle)
        try:
            batch_gt.append(pcl_gt)
            batch_rgb.append(pcl_rgb)
        except:
            pass
    batch_gt = np.array(batch_gt)
    batch_rgb = np.array(batch_rgb)

    return batch_ip, batch_gt, batch_rgb


def fetch_batch_seg(dataset, models, indices, batch_num, batch_size):
    batch_ip = []
    batch_gt = []
    batch_label_wts = []
    for ind in indices[(batch_num*batch_size):
                       (batch_num*batch_size + batch_size)]:
        model_path = models[ind]
        pcl_inp = fetch_pcl(dataset, model_path)
        labels_gt = fetch_labels(model_path)
        batch_ip.append(pcl_inp)
        batch_gt.append(labels_gt)
    batch_ip = np.array(batch_ip)
    batch_gt = np.array(batch_gt)
    return batch_ip, batch_gt


def get_shapenet_drc_models(data_dir, categs=['03001627'],
                            num_views=10):
    '''
    Obtain indices and names of all point cloud models, for train and
    validation sets
    args:
        data_dir: str; root directory containing data of all
                        categories
        categs: list of str; list of category ids for which indices
                                and names have to be returned
        num_views: number of view points from which images are
                    rendered for each model
    returns:
        train_pair_indices, val_pair_indices: list of tuples of
                    model index and rendered image index
        train_models, val_models: list of str; names of training
                    and validation pcl models respectively
    '''
    train_models = []
    val_models = []

    for cat in categs:
        cat_train_model = np.load(
            data_dir+'/splits/%s_train_list.npy' % cat
            )
        cat_val_model = np.load(
            data_dir + '/splits/%s_val_list.npy' % cat)
        cat_train_model = [
            join(data_dir, cat, item) for item in cat_train_model
            ]
        cat_val_model = [
            join(data_dir, cat, item) for item in cat_val_model
            ]
        train_models.extend(cat_train_model)
        val_models.extend(cat_val_model)

    train_pair_indices = list(
        product(xrange(len(train_models)), xrange(num_views))
        )
    val_pair_indices = list(
        product(xrange(len(val_models)), xrange(num_views))
        )

    print 'TRAINING: models={}  samples={}'.format(
        len(train_models), len(train_models)*num_views
        )
    print 'VALIDATION: models={}  samples={}'.format(
        len(val_models), len(val_models)*num_views
        )

    return (
            train_models, val_models,
            train_pair_indices, val_pair_indices
            )


def get_drc_models_util(data_dir, category, eval_set):
    models = []
    if category == 'all':
        cats = ['chair', 'car', 'aero']
    else:
        cats = [category]
    for cat in cats:
        category_id = shapenet_category_to_id[cat]
        splits_file_path = join(
            data_dir, 'splits', category_id +
            '_%s_list.txt' % eval_set)
        with open(splits_file_path, 'r') as f:
            for model in f.readlines():
                models.append(
                    join(data_dir, category_id, model.strip()))

    return models


def get_shapenet_drc_models_partseg(data_dir, category, NUM_VIEWS,
                                    eval_set):
    models = get_drc_models_util(data_dir, category, eval_set)
    pair_indices = list(
        product(xrange(len(models)), xrange(NUM_VIEWS))
        )
    print '{}: models={}  samples={}'.format(
        eval_set, len(models), len(models)*NUM_VIEWS)

    return models, pair_indices


def get_drc_models_seg(dataset, data_dir, category, eval_set):
    models = get_drc_models_util(data_dir, category, eval_set)
    indices = list(xrange(len(models)))
    print '{}: models={}'.format(eval_set, len(models))

    return models, indices


def get_feed_dict(models, indices, models_pcl, b, args):
    '''
    Obtain batch data for training
    args:
        models: list of all ids/names of input image models
        indices: indices to be chosen from models for the current
                    batch
        models_pcl: list of all ids/names of pcl models
        b: index of the current batch
        args: input arguments while running the train.py file
    returns:
        All outputs are lists of length N_VIEWS.
        Properties of each element in the list:
        batch_ip: uint8, (BS,IMG_H,IMG_W,3); input rgb images
        batch_gt: float, (BS,IMG_H,IMG_W); GT foreground masks
        batch_gt_rgb: float, (BS,IMG_H,IMG_W,3); GT rgb images from
                            different views
        batch_names: str; names of pcl models corresponding to
                            input images
        batch_x: float, (); rotation angle along x-axis for the
                                view point in radians
        batch_y: float, (); rotation angle along y-axis for the
                                view point in radians
    '''
    batch = fetch_batch_drc(models, indices, b, args.batch_size, args)
    (batch_ip, batch_rgb, _, batch_gt_mask,
        batch_x, batch_y, model_names) = batch

    # pcl gt
    _, batch_pcl_gt, _ = fetch_batch_pcl_rgb(
        models_pcl, indices, b, args.batch_size
        )

    # Align axes according to the co-ordinate system of renderer
    batch_pcl_gt = preprocess_pcl_gt(batch_pcl_gt)

    return (batch_ip, batch_rgb, batch_gt_mask, batch_pcl_gt,
            batch_x, batch_y, model_names)


def preprocess_pcl_gt(pcl):
    '''
    To align the GT pcl according to the axes of the GT image renderer
    (i.e.  the co-ordinate system used while rendering the images from
    GT PCL), interchange the axes and change axes directions
    args:
        pcl: float, (BS,N_PTS,3), numpy array; input point cloud
    returns:
        pcl: float, (BS,N_PTS,3), numpy array; transformed point cloud
    '''
    pcl[:, :, [0, 2]] = pcl[:, :, [2, 0]]
    pcl[:, :, [0, 1]] = pcl[:, :, [1, 0]]
    pcl[:, :, 1] = -pcl[:, :, 1]
    pcl[:, :, 0] = -pcl[:, :, 0]

    return pcl
