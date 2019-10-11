#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy.ndimage
from skimage.transform import PiecewiseAffineTransform, warp


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


def demon(im_moving, im_static, times, gaussian_size):
    s = im_static
    m = im_moving
    rows, cols = s.shape[0], s.shape[1]
    src_cols = np.linspace(0, cols, cols)
    src_rows = np.linspace(0, rows, rows)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    alpha = 2.5
    h_smooth = gaussian_filter(gaussian_size[0:2], gaussian_size[2])
    t_x = np.zeros(np.shape(m))
    t_y = np.zeros(np.shape(m))
    s_x, s_y = np.gradient(s)
    for ii in range(times):
        image_diff = m - s
        m_x, m_y = np.gradient(m)
        d1 = (s_x ** 2 + s_y ** 2) + (alpha ** 2) * (image_diff ** 2)
        d2 = (m_x ** 2 + m_y ** 2) + (alpha ** 2) * (image_diff ** 2)
        temp1 = np.array(np.where(d1 != 0))
        temp2 = np.array(np.where(d2 != 0))
        d1[temp1[0], temp1[1]] = 1 / d1[temp1[0], temp1[1]]
        d2[temp2[0], temp2[1]] = 1 / d2[temp2[0], temp2[1]]
        u_x = -image_diff * (s_x * d1 + m_x * d2)
        u_y = -image_diff * (s_y * d1 + m_y * d2)
        u_xs = 3 * scipy.ndimage.correlate(u_x, h_smooth, mode='constant')
        u_ys = 3 * scipy.ndimage.correlate(u_y, h_smooth, mode='constant')
        t_x = t_x+u_xs
        t_y += u_ys
        tform = PiecewiseAffineTransform()
        dst = np.dstack([t_x.flat,t_y.flat])[0]
        tform.estimate(src,dst)
        out_rows = s.shape[0] - 1.5 * 50
        out_cols = cols
        out = warp(m, tform, output_shape=(out_rows, out_cols))
        print(out)

