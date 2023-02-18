# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 13:41:57 2022

@author: Giulio Rossetti
giulio.rossetti94@gmail.com
"""

import numpy as np
import scipy.linalg as sp
from scipy.special import gammainc

def RankTestKP(R, F):
    """
    Kleibergen and Paap (2006) rank test
    Parameters. Null is that D is of reduced rank
    D = E[R_t F_t']
    
    H0: rank(D) = G - 1
    
    ----------
    R : np.ndarray
        T x N matrix of returns.
    F : np.ndarray
        T x K matrix of factors.

    Returns
    -------
    pval : Float
        pvalues of the test.

    """
    
    
    if not isinstance(R, np.ndarray):
        R = np.array(R)
    if not isinstance(F, np.ndarray):
        F = np.array(F)

    T, K = F.shape
    _, N = R.shape

    Vrr = (R.T @ R)/T
    Vff = (F.T @ F)/T

    Sx = sp.sqrtm(sp.inv(Vrr))
    Sy = sp.sqrtm(sp.inv(Vff))

    R1 = R @ Sx
    F1 = F @ Sy
    D = (R1.T @ F1)/T

    U, S, V = sp.svd(D)

    U2 = U[:,K-1:]
    V2 = V[:,-1]

    R2 = R1 @ U2
    F2 = F1 @ V2

    m = (F2[:,np.newaxis] @ np.ones((1,N - K + 1))) * R2

    W1 = m.T @ m / T
    W11 = W1[0,0]
    W12 = W1[0,1:][np.newaxis,:]
    W22 = W1[1:,1:]
    
    stat = T * S[K-1]**2 / (W11 - W12 @ sp.inv(W22) @ W12.T)
    pval = gammainc(stat/2 , (N - K + 1)/2)
    
    return pval.item()





