import numpy as np

EPS = 1e-12
THRESHOLD = 1e+12

def mnmf(input, n_basis = 10, n_sources = None, iteration = 100):
    n_channels, n_bins, n_frames = input.shape

    # Initializing step 
    n_sources = n_channels

    G = np.ones((n_sources, n_bins, n_channels)) * 1e-2
    for m in range(n_channels):
        G[m % n_sources, :, m] = 1

    basis = np.random.rand(n_sources, n_bins, n_basis)    
    activation = np.random.rand(n_sources, n_basis, n_frames)
    diagonalizer = np.tile(np.eye(n_channels, dtype=np.complex128), (n_bins, 1, 1))
    spatial_covariance = G

    # Updating w, h, g & q

    for _ in range(iteration):
        X = input.transpose(1, 2, 0)
        g = spatial_covariance
        Q = diagonalizer
        W, H = basis, activation

        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis=3)
        x_tilde = np.abs(QX) ** 2

        # Update W & H
        L = W @ H
        Y_tidle = np.sum(L[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        Y_tidle[Y_tidle < EPS] = EPS
        xy = x_tilde / (Y_tidle ** 2)
        g_y = np.sum(g[:, :, np.newaxis] * xy[np.newaxis], axis = 3)
        gy_ = np.sum(g[:, :, np.newaxis] / Y_tidle[np.newaxis], axis = 3)

        num = np.sum(H[:, np.newaxis, :, :] * g_y[:, :, np.newaxis], axis = 3)
        den = np.sum(H[:, np.newaxis, :, :] * gy_[:, :, np.newaxis], axis = 3)
        den[den < EPS] = EPS
        W = W * np.sqrt(num / den)

        L = W @ H
        Y_tidle = np.sum(L[...,np.newaxis] * g[:, :, np.newaxis], axis = 0)
        Y_tidle[Y_tidle < EPS] = EPS
        xy = x_tilde / (Y_tidle ** 2)
        g_y = np.sum(g[:, :, np.newaxis] * xy[np.newaxis], axis = 3)
        gy_ = np.sum(g[:, :, np.newaxis] / Y_tidle[np.newaxis], axis = 3)

        num = np.sum(W[:, :, :, np.newaxis] * g_y[:, :, np.newaxis], axis = 1)
        den = np.sum(W[:, :, :, np.newaxis] * gy_[:, :, np.newaxis], axis = 1)
        den[den < EPS] = EPS
        H = H * np.sqrt(num / den)

        basis, activation = W, H

        # Update Q

        g = spatial_covariance
        W, H = basis, activation
        X = input.transpose(1, 2, 0)
        Q = diagonalizer

        L = W @ H

        Y_tidle = np.sum(L[..., np.newaxis] * g[:, :, np.newaxis], axis = 0)
        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis = 3)
        x_tilde = np.abs(QX) ** 2

        Y_tidle[Y_tidle < EPS] = EPS
        xy = x_tilde / (Y_tidle ** 2)
        
        A = np.sum(L[..., np.newaxis] * xy[np.newaxis], axis = 2)
        B = np.sum(L[..., np.newaxis] / Y_tidle[np.newaxis], axis = 2)
        B[B < EPS] = EPS
        g = g * np.sqrt(A / B)

        spatial_covariance = g

        X = input.transpose(1, 2, 0)
        Q = diagonalizer
        g = spatial_covariance
        XX = X[:, :, :, np.newaxis] @ X[:, :, np.newaxis, :].conj()

        W, H = basis, activation
        L = W @ H

        Y_tidle = np.sum(L[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        Y_tidle[Y_tidle < EPS] = EPS
        hot_vec = np.eye(n_channels)
        hot_vec = np.tile(hot_vec, reps=(n_bins, 1, 1))

        for channel_idx in range(n_channels):
            q_m_Hermite = Q[:, channel_idx, :]
            V = (XX / Y_tidle[:, :, channel_idx, np.newaxis, np.newaxis]).mean(axis=1)
            QV = Q @ V
            # condition = np.linalg.cond(QV) < THRESHOLD
            # condition = condition[:,np.newaxis]
            e_m = hot_vec[:, channel_idx, :]
            q_m = np.linalg.solve(QV, e_m)
            qVq = q_m.conj()[:, np.newaxis, :] @ V @ q_m[:, :, np.newaxis]
            den = np.sqrt(qVq[...,0])
            den[den < EPS] = EPS
            # q_m_Hermite = np.where(condition, q_m.conj() / den, q_m_Hermite)
            Q[:, channel_idx, :] = q_m_Hermite
        diagonalizer = Q


        Q = diagonalizer
        g = spatial_covariance
        W, H = basis, activation

        Q_Q = Q * Q.conj()
        Q_Q_s = np.real(Q_Q.sum(axis=2).mean(axis=1))
        Q_Q_s[Q_Q_s < EPS] = EPS
        Q /= np.sqrt(Q_Q_s)[:, np.newaxis, np.newaxis]
        g /= Q_Q_s[np.newaxis, :, np.newaxis]

        g_sum = g.sum(axis=2)
        g_sum[g_sum < EPS] = EPS
        g /= g_sum[:, :, np.newaxis]
        W *= g_sum[:, :, np.newaxis]

        Wsum = W.sum(axis = 1)
        Wsum[Wsum < EPS] = EPS
        W /= Wsum[:, np.newaxis]
        H *= Wsum[:, :, np.newaxis]

        basis, activation = W, H
        diagonalizer = Q
        spatial_covariance = g

    # Separate G

    X = input.transpose(1, 2, 0)
    Q = diagonalizer
    g = spatial_covariance

    W, H = basis, activation
    L = W @ H
    
    L_G = L[..., np.newaxis] * g[:, :, np.newaxis, :]
    y_tilde = np.sum(L_G, axis=0)
    Q_inv = np.linalg.inv(Q)
    QX = np.sum(Q[:, np.newaxis, :] * X[:, :, np.newaxis], axis=3)
    y_tilde[y_tilde < EPS] = EPS
    QXLambdaGy = QX * (L_G / y_tilde)
    
    X_hat = np.sum(Q_inv[:, np.newaxis, :, :] * QXLambdaGy[:, :, :, np.newaxis, :], axis=4)
    X_hat = X_hat.transpose(0, 3, 1, 2)
    
    return X_hat[:, 0, :, :]
