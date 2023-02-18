# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 13:41:57 2022

@author: Giulio Rossetti
giulio.rossetti94@gmail.com
"""

import numpy as np

def nw(h, lag=None, prewhite=False):
    """
    Parameters
    ----------
    h : np.ndarray
        T x N matrix of data.
    lag : int
        number of lags.
    prewhite : boolean
        do pre-whitening. The default is False.

    Returns
    -------
    V : np.dfarray
        covariance matrix of h.

    """
    if not isinstance(h, np.ndarray):
        h = np.array(h)
        
    if prewhite:
        h0 = h[:-1]
        h1 = h[1:]
        A = np.linalg.lstsq(h0, h1, rcond=None)[0]
        he = h1 - h0 @ A
    else:
        he = h
    T, r = he.shape
    if lag is not None and lag >= 0:
        V = he.T @ he / T
        for i in range(1, lag+1):
            V1 = he[i:T].T @ he[:T-i] / T
            V += (1 - i / (lag+1)) * (V1 + V1.T)
    else:
        T1 = len(he)
        n = int(np.fix(12 * (0.01 * T) ** (2/9)))
        w = np.ones((r, 1))
        hw = he @ w
        sigmah = np.zeros(n)
        for i in range(1, n+1):
            sigmah[i-1] = hw[:T1-i].T @ hw[i:T1] / T1
        sigmah0 = hw.T @ hw / T1
        s0 = sigmah0 + 2 * np.sum(sigmah)
        s1 = 2 * np.sum(np.arange(1, n+1) * sigmah)
        gam = 1.1447 * np.abs(s1 / s0) ** (2/3)
        m = int(np.fix(gam * T ** (1/3)))
        V = he.T @ he / T1
        for i in range(1, m+1):
            V1 = he[i:T1].T @ he[:T1-i] / T1
            V += (1 - i / (m+1)) * (V1 + V1.T)
        if prewhite:
            IA = np.linalg.inv(np.eye(r) - A.T)
            V = IA @ V @ IA.T
    return V