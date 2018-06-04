# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:29:17 2018

@author: jakpo_000
"""
from copy import deepcopy
import os
import cv2
import numpy as np

import nibabel as nib
import SimpleITK as sitk

from keras.preprocessing import image


def cv_load(img_path, target_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (target_size), )
    img /= 255.
    img = img[:, :, 0]
    return img


def load_2d(img_path, t_size):
    if not isinstance(t_size, tuple):
        t_size = t_size, t_size
    img = image.load_img(img_path, target_size=(t_size))
    img = image.img_to_array(img)
    img /= 255.
    img = img[:, :, 0]
    return img

def pil_load(path, target_size):
     if not isinstance(t_size, tuple):
        t_size = t_size, t_size

def load(path, t_size):
    if not isinstance(t_size, tuple):
        t_size = t_size, t_size

    if os.path.isfile(path):
        img = image.load_img(path, target_size=(t_size))
        img = image.img_to_array(img)
        img /= 255.
        img = img[:, :, 0]
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        path_str = path

    if os.path.isdir(path):
        path_names = os.listdir(path)
        path_str = [path + i for i in path_names]
        n_name = len(path_names)

        cube = np.zeros((n_name, t_size[0], t_size[1]))
        c = 0
        # make/load picture matrix
        for i in path_str:
            img = image.load_img(i, target_size=(t_size))
            img = image.img_to_array(img)
            img /= 255.
            cube[c, :, :] = deepcopy(img[:, :, 0])
            c += 1
        img = cube
        img = np.expand_dims(img, axis=-1)

    return img, path_str


def to_tensor(img):
    tens = deepcopy(img)
    tens = np.expand_dims(tens, axis=0)
    tens = np.expand_dims(tens, axis=-1)
    return tens

def mha(path):
    img = sitk.ReadImage(path, sitk.sitk.Float32)
    arr = sitk.GetImageFromArray(img)
    arr /= 255.
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def nii(path):
    img = nib.load(path)
    arr = np.array(img.dataobj)
    arr /= 255.
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def nii_batch(path, batch_size='all'):
    """
    load dir names from path -> load pictures in big Matrix --> calculate mean
    std and normed matrices --
    """
    # target size; value against true divide
    size = 128
    # truedivider = 1e-6

    # path = 'OASIS/Test/train/normal/'
    # path loading; length
    path_names = os.listdir(path)
    path_str = [path + i for i in path_names]
    n_name = len(path_names)

    if batch_size is not 'all':
        n_name = batch_size
    # M allocations; Cube = (Batch, m, n)
    cube = np.zeros((n_name, size, size, size))

    # std1 = np.zeros((size,size))
    c = 0

    # make/load picture matrix
    for i in path_str:
        img = nib.load(i)
        a = np.array(img.dataobj)

        img /= 255.
        cube[c] = deepcopy(a)
        c += 1
        if c == n_name: break

    cube = np.expand_dims(cube, axis=-1)
    return cube
