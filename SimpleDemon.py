#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy.ndimage
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates


def gaussian_filter(dim1, dim2):
    siz = (np.array(dim1) - 1) / 2
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]), np.arange(-siz[0], siz[0]))
    arg = -(x * x + y * y) / (2 * dim2 ** 2)
    h = np.exp(arg)
    eps = np.finfo(np.float).eps
    h[h < eps * np.amax(h)] = 0
    sum_h = np.sum(np.sum(h))
    if sum_h != 0:
        h = h / sum_h
    return h


def simple_demon(im_moving, im_static, times, gaussian_size, alpha):
    s = np.array(im_static.convert('L')) / 255
    m = np.array(im_moving.convert('L')) / 255
    field = np.meshgrid(np.linspace(0, s.shape[0], s.shape[0]), np.linspace(0, s.shape[1], s.shape[1]))
    h_smooth = gaussian_filter(gaussian_size[0:2], gaussian_size[2])
    dsdy, dsdx = np.gradient(s)
    for ii in range(times):
        image_diff = m - s
        dmdy, dmdx = np.gradient(m)
        u_cols = -image_diff * (dsdy * ((dsdy ** 2 + dsdx ** 2) + (alpha ** 2) * (image_diff ** 2)) + dmdy * (
                    (dmdy ** 2 + dmdx ** 2) + (alpha ** 2) * (image_diff ** 2)))
        u_rows = -image_diff * (dsdx * ((dsdy ** 2 + dsdx ** 2) + (alpha ** 2) * (image_diff ** 2)) + dmdx * (
                    (dmdy ** 2 + dmdx ** 2) + (alpha ** 2) * (image_diff ** 2)))
        u_cols[np.isnan(u_cols)] = 0
        u_rows[np.isnan(u_rows)] = 0
        u_cols = 3 * scipy.ndimage.correlate(u_cols, h_smooth, mode='constant')
        u_rows = 3 * scipy.ndimage.correlate(u_rows, h_smooth, mode='constant')
        field[0] += u_rows
        field[1] += u_cols
        m = map_coordinates(m, field, mode='nearest')
    return field
