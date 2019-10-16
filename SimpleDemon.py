#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy.ndimage
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
import matplotlib.pyplot as plt
from PIL import Image


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


def simple_demon(im_moving, im_static, times, gaussian_size):
    s = np.array(im_static.convert('L')) / 255
    m = np.array(im_moving.convert('L')) / 255
    rows, cols = s.shape[0], s.shape[1]
    src_cols = np.linspace(0, cols, cols)
    src_rows = np.linspace(0, rows, rows)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    alpha = 2.5
    h_smooth = gaussian_filter(gaussian_size[0:2], gaussian_size[2])
    t_rows = src_rows
    t_cols = src_cols
    dsdy, dsdx = np.gradient(s)
    for ii in range(times):
        image_diff = m - s
        dmdy, dmdx = np.gradient(m)
        d1 = (dsdy ** 2 + dsdx ** 2) + (alpha ** 2) * (image_diff ** 2)
        d2 = (dmdy ** 2 + dmdx ** 2) + (alpha ** 2) * (image_diff ** 2)
        temp1 = np.array(np.where(d1 != 0))
        temp2 = np.array(np.where(d2 != 0))
        d1[temp1[0], temp1[1]] = 1 / d1[temp1[0], temp1[1]]
        d2[temp2[0], temp2[1]] = 1 / d2[temp2[0], temp2[1]]
        u_cols = -image_diff * (dsdy * d1 + dmdy * d2)
        u_rows = -image_diff * (dsdx * d1 + dmdx * d2)
        u_cols = 3 * scipy.ndimage.correlate(u_cols, h_smooth, mode='constant')
        u_rows = 3 * scipy.ndimage.correlate(u_rows, h_smooth, mode='constant')
        t_rows += 100 * u_rows
        t_cols += 100 * u_cols
        dst = np.dstack([t_cols.flat, t_rows.flat])[0]
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        out = warp(im_moving, tform)[:, :, 0:3]
        out[:, :, 0] = out[:, :, 0].T
        out[:, :, 1] = out[:, :, 1].T
        out[:, :, 2] = out[:, :, 2].T
        m = np.array(Image.fromarray(np.uint8(out)).convert('L'))
        fig, ax = plt.subplots()
        ax.imshow(m - s)
        ax.axis((0, rows, cols, 0))
        plt.show()
    fig, ax = plt.subplots()
    ax.imshow(out)
    ax.axis((0, rows, cols, 0))
    plt.show()
