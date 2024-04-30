'''
    NMF module
'''

import numpy as np

def nndsvd_initialization(a, rank):
    '''
        Init func
    '''
    u, s, v = np.linalg.svd(a, full_matrices=False)
    v = v.T
    w = np.zeros((a.shape[0], rank))
    h = np.zeros((rank, a.shape[1]))

    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0].T)

    for i in range(1, rank):
        ui = u[:, i]
        vi = v[:, i]
        ui_pos = (ui >= 0) * ui
        ui_neg = (ui < 0) * -ui
        vi_pos = (vi >= 0) * vi
        vi_neg = (vi < 0) * -vi

        ui_pos_norm = np.linalg.norm(ui_pos, 2)
        ui_neg_norm = np.linalg.norm(ui_neg, 2)
        vi_pos_norm = np.linalg.norm(vi_pos, 2)
        vi_neg_norm = np.linalg.norm(vi_neg, 2)

        norm_pos = ui_pos_norm * vi_pos_norm
        norm_neg = ui_neg_norm * vi_neg_norm

        if norm_pos >= norm_neg:
            w[:, i] = np.sqrt(s[i] * norm_pos) / ui_pos_norm * ui_pos
            h[i, :] = np.sqrt(s[i] * norm_pos) / vi_pos_norm * vi_pos.T
        else:
            w[:, i] = np.sqrt(s[i] * norm_neg) / ui_neg_norm * ui_neg
            h[i, :] = np.sqrt(s[i] * norm_neg) / vi_neg_norm * vi_neg.T

    return w, h



def divergence(V, W, H):
    '''
        Divergence func
    '''
    return (1 / 2) * np.linalg.norm(W @ H - V)


def NMF(V, S, MAXITER = 5000, threshold = 1e-12): 
    ''''
        NMF func
    '''
    counter = 0
    cost_function = []
    beta_divergence = 1

    W, H = nndsvd_initialization(V, S)

    while beta_divergence >= threshold and counter <= MAXITER:

        H *= (W.T @ V) / (W.T @ (W @ H) + 10e-12)
        H[H < 0] = 0
        W *= (V @ H.T) / ((W @ H) @ H.T + 10e-12)
        W[W < 0] = 0

        beta_divergence =  divergence(V, W, H)
        cost_function.append(beta_divergence)
        counter += 1

    return W, H, cost_function
