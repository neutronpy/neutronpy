# -*- coding: utf-8 -*-
import numpy as np


def gram_schmidt(in_vecs, row_vecs=True, normalize=True):
    r"""

    """
    if not row_vecs:
        in_vecs = in_vecs.T

    out_vecs = in_vecs[0:1, :]

    for i in range(1, in_vecs.shape[0]):
        proj = np.diag((in_vecs[i, :].dot(out_vecs.T) / np.linalg.norm(out_vecs, axis=1) ** 2).flat).dot(out_vecs)
        out_vecs = np.vstack((out_vecs, in_vecs[i, :] - proj.sum(0)))

    if normalize:
        out_vecs = np.diag(1 / np.linalg.norm(out_vecs, axis=1)).dot(out_vecs)

    if not row_vecs:
        out_vecs = out_vecs.T

    return out_vecs
