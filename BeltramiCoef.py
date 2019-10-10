#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cmath
import scipy
import numpy as np
from package_functions import MeshOperator


def mu_metric(v, f, mapping, dimension):
    s = MeshOperator.f2v(v, f)
    v = np.array(v)
    f = np.array(f)
    mapping = np.array(mapping)
    if len(v) > len(v[0]):
        v = v.transpose()
    if len(f) > len(f[0]):
        f = f.transpose()
    (dx, dy, dz, dc) = MeshOperator.diff_operator(v, f)
    if len(mapping) > len(mapping[0]):
        mapping = mapping.transpose()
    if dimension == 2:
        f = mapping[0, :] + 1j * mapping[1, :]
        mu = (dc * np.transpose([f])) / (dz * np.transpose([f]))
    elif dimension == 3:
        dXdu = np.concatenate(dx * np.transpose([mapping[0, :]]))
        dXdv = np.concatenate(dy * np.transpose([mapping[0, :]]))
        dYdu = np.concatenate(dx * np.transpose([mapping[1, :]]))
        dYdv = np.concatenate(dy * np.transpose([mapping[1, :]]))
        dZdu = np.concatenate(dx * np.transpose([mapping[2, :]]))
        dZdv = np.concatenate(dy * np.transpose([mapping[2, :]]))
        E = dXdu**2+dYdu**2+dZdu**2
        G = dXdv**2+dYdv**2+dZdv**2
        F = dXdu*dXdv + dYdu*dYdv + dZdu*dZdv
        mu = np.transpose([(E-G+2*1j*F)/(E+G+2*np.sqrt(E*G-F**2))])
    else:
        print('Dimension should either be 2 or 3. Please check again.')
        mu = []
    return mu


def mu_chop(mu, bound):
    for ii in range(len(mu)):
        if abs(mu[ii]) > bound:
            mu[ii] = bound * cmath.cos(cmath.phase(mu[ii])) + 1j * bound * cmath.sin(cmath.phase(mu[ii]))
    return mu


def mu_smooth(mu, operator, p_lambda, delta):
    smooth_operator = (1 + delta) * scipy.speye(len(mu)) - 0.5 * p_lambda * operator.laplacian
    return smooth_operator
