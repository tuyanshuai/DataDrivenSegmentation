import numpy as np
import scipy
import scipy.sparse
import trimesh


def laplace_beltrami(v, f):
    v = np.array(v)
    f = np.array(f)
    if len(v) > len(v[0]):
        v = v.transpose()
    if len(f) > len(f[0]):
        f = f.transpose()
    l = np.stack([[np.sqrt(np.sum((v[:, f[1, :]] - v[:, f[2, :]]) ** 2, axis=0))],
                  [np.sqrt(np.sum((v[:, f[2, :]] - v[:, f[0, :]]) ** 2, axis=0))],
                  [np.sqrt(np.sum((v[:, f[0, :]] - v[:, f[1, :]]) ** 2, axis=0))]], axis=0)
    f1 = f[0, :]
    f2 = f[1, :]
    f3 = f[2, :]
    l1 = l[0, :]
    l2 = l[1, :]
    l3 = l[2, :]
    s = (l1 + l2 + l3) * 0.5
    area = 2 * np.sqrt(s * (s - l1) * (s - l2) * (s - l3))
    area[area < 1e-5] = 1e-5
    cot12 = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (4 * area)
    cot23 = (l2 ** 2 + l3 ** 2 - l1 ** 2) / (4 * area)
    cot31 = (l1 ** 2 + l3 ** 2 - l2 ** 2) / (4 * area)
    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23
    ii = np.concatenate([f1, f2, f2, f3, f3, f1, f1, f2, f3], axis=0)
    jj = np.concatenate([f2, f1, f3, f2, f1, f3, f1, f2, f3], axis=0)
    v_new = np.concatenate((cot12.T, cot12.T, cot23.T, cot23.T, cot31.T, cot31.T, diag1.T, diag2.T, diag3.T), axis=1)
    lb_operator = scipy.sparse.bsr_matrix((v_new.flatten('F'), (ii, jj)), shape=(len(v[0]), len(v[0])))
    return lb_operator


def diff_operator(v, f):
    v = np.array(v)
    f = np.array(f)
    if len(v) > len(v[0]):
        v = v.transpose()
    if len(f) > len(f[0]):
        f = f.transpose()
    n = len(f[0])
    Mi = np.reshape([[np.arange(n)], [np.arange(n)], [np.arange(n)]], [1, 3 * n], 'F')
    Mj = np.reshape(f, [1, 3 * n], 'F')
    (e1, e2, e3) = get_edge(v, f)
    area = get_signed_area_edge(e1, e2)
    area = np.concatenate([[area], [area], [area]], axis=0)
    mx = np.reshape(np.concatenate([[e1[1, :]], [e2[1, :]], [e3[1, :]]], axis=0) / (area * 2), [1, 3 * n], 'F')
    my = -np.reshape(np.concatenate([[e1[0, :]], [e2[0, :]], [e3[0, :]]], axis=0) / (area * 2), [1, 3 * n], 'F')
    dx = scipy.sparse.csr_matrix((np.concatenate(mx), (np.concatenate(Mi), np.concatenate(Mj))), dtype=np.float)
    dy = scipy.sparse.csr_matrix((np.concatenate(my), (np.concatenate(Mi), np.concatenate(Mj))), dtype=np.float)
    dz = (dx - 1j * dy) / 2
    dc = (dx + 1j * dy) / 2
    return dx, dy, dz, dc


def get_edge(v, f):
    v = np.array(v)
    f = np.array(f)
    if len(v) > len(v[0]):
        v = v.transpose()
    if len(f) > len(f[0]):
        f = f.transpose()
    e1 = v[0:2, f[2, :]] - v[0:2, f[1, :]]
    e2 = v[0:2, f[0, :]] - v[0:2, f[2, :]]
    e3 = v[0:2, f[1, :]] - v[0:2, f[0, :]]
    return e1, e2, e3


def get_signed_area_edge(e1, e2):
    xb = -e1[0, :]
    yb = -e1[1, :]
    xa = e2[0, :]
    ya = e2[1, :]
    area = (xa * yb - xb * ya) / 2
    return area


def f2v(v, f):
    v = np.array(v)
    f = np.array(f)
    if len(v) < len(v[0]):
        v = v.transpose()
    if len(f) < len(f[0]):
        f = f.transpose()
    ring = trimesh.base.Trimesh(vertices=v, faces=f, process=False).vertex_faces
    if len(v) > len(v[0]):
        v = v.transpose()
    if len(f) > len(f[0]):
        f = f.transpose()
    nv = len(v[0])
    nf = len(f[0])
    ii = np.zeros(sum(ring[0] > -1))
    jj = np.sort(ring[0][ring[0] > -1])
    avg = [sum(ring[0] > -1)]
    for kk in range(1, nv):
        ii = np.concatenate([ii, kk * np.ones(sum(ring[kk] > -1))], axis=0)
        jj = np.concatenate([jj, np.sort(ring[kk][ring[kk] > -1])], axis=0)
        avg = np.concatenate([avg, [sum(ring[kk] > -1)]], axis=0)
    s = scipy.sparse.csr_matrix((np.ones(len(jj)), (ii.astype(int), jj.astype(int))), shape=(nv, nf), dtype=np.float)
    s = scipy.sparse.csr_matrix((1 / avg, (np.arange(nv), np.arange(nv))), dtype=np.float) * s
    return s


def v2f(v, f):
    v = np.array(v)
    f = np.array(f)
    if len(v) > len(v[0]):
        v = v.transpose()
    if len(f) > len(f[0]):
        f = f.transpose()
    nv = len(v[0])
    nf = len(f[0])
    ii = np.concatenate([[np.arange(nf)], [np.arange(nf)], [np.arange(nf)]], axis=0)
    jj = f.flatten('F')
    s = scipy.sparse.csr_matrix((np.ones(len(jj)), (ii.flatten('F').astype(int), jj.astype(int))), shape=(nf, nv),
                                dtype=np.float) / 3
    return s
