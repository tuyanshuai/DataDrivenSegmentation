#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy.ndimage
from skimage.transform import warp


def gaussian_filter(dim1, dim2):
    pt = np.arange(-0.5*(dim1-1), 0.5*(dim1-1)+1)
    [x, y] = np.meshgrid(pt, pt)
    arg = -(x*x + y*y)/(2 * dim2 ** 2)
    h = np.exp(arg)
    eps = np.finfo(float).eps
    h[np.where(h < eps*np.amax(h))] = 0
    sum_h = sum(sum(h))
    if sum_h != 0:
        h = h/sum_h
    return h


def demon(im_moving, im_static, times, gaussian_size):
    s = im_static
    m = im_moving
    alpha = 2.5
    h_smooth = gaussian_filter(gaussian_size[0], gaussian_size[1])
    t_x = np.zeros(np.shape(m))
    t_y = np.zeros(np.shape(m))
    [s_x, s_y] = np.gradient(s)
    for ii in range(times-1):
        image_diff = m - s
        [m_x, m_y] = np.gradient(m)
        d1 = (s_x**2+s_y**2) + (alpha**2)*(image_diff**2)
        d2 = (m_x**2+m_y**2) + (alpha**2)*(image_diff**2)
        d1[np.where(d1 != 0)] = 1/d1
        d2[np.where(d2 != 0)] = 1/d2
        u_x = -image_diff * (s_x*d1 + m_x*d2)
        u_y = -image_diff * (s_y*d1 + m_y*d2)
        u_xs = scipy.ndimage.correlate(u_x, h_smooth, mode='constant').transpose()
        u_ys = scipy.ndimage.correlate(u_y, h_smooth, mode='constant').transpose()
        t_x += u_xs
        t_y += u_ys
        # move pixel here
