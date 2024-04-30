import numpy as np
from mnmf import mnmf

def nndsvd_initialization(a, rank):

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

def divergence(V,W,H, beta = 2):
    import math
    
    """
    beta = 2 : Euclidean cost function
    beta = 1 : Kullback-Leibler cost function
    beta = 0 : Itakura-Saito cost function
    """ 
    
    if beta == 0 : return np.sum( V/np.dot(W, H) - np.log10(V/np.dot(W, H)) -1 )
    
    if beta == 1 : return np.sum( V*np.log10(V/np.dot(W, H)) + (np.dot(W, H) - V))
    
    if beta == 2 : return 1/2*np.linalg.norm(np.dot(W, H)-V)


def NMF(V, S, beta = 2,  threshold = 0.05, MAXITER = 5000): 
    
    """
    inputs : 
    --------
        V         : Mixture signal : |TFST|
        S         : The number of sources to extract
        beta      : Beta divergence considered, default=2 (Euclidean)
        threshold : Stop criterion 
        MAXITER   : The number of maximum iterations, default=1000                                                     
    
    outputs :
    ---------
        W : dictionary matrix [KxS], W>=0
        H : activation matrix [SxN], H>=0
        cost_function : the optimised cost function over iterations
       
   Algorithm : 
   -----------
   
    1) Randomly initialize W and H matrices
    2) Multiplicative update of W and H 
    3) Repeat step (2) until convergence or after MAXITER   
    """
    
    counter = 0
    cost_function = []
    beta_divergence = 1
    
    K, N = np.shape(V)
    
    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))

    while beta_divergence >= threshold and counter <= MAXITER:
        
        # Update of W and H
        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
        
        # Compute cost function
        beta_divergence =  divergence(V,W,H, beta = 2)
        cost_function.append( beta_divergence )
        counter += 1
       
    return W,H, cost_function



if __name__ == '__main__':
    max_iter = 100
    import librosa
    V, sr = librosa.load('./data/sound1_mix.mp3')
    V = librosa.stft(V)
    V = np.abs(V)
    import matplotlib.pyplot as plt

    W, H, cf = NMF(V, 2, MAXITER=max_iter)

    V_rec = np.dot(W, H)
    plt.plot(np.arange(0, max_iter + 1), cf)
    plt.xlabel('Iteration')
    plt.ylabel('Divergence')
    plt.title('Divergences from NMF')
    plt.show()  
