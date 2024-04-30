import numpy as np

EPS = 1e-12
THRESHOLD = 1e-12

def mnmf(input, n_basis = 10, n_sources = None, iteration = 100):
    n_channels, n_bins, n_frames = input.shape

    # Initializing step 

    if n_sources is None:
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
        Lambda = W @ H
        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        R[R < EPS] = EPS
        xR = x_tilde / (R ** 2)
        gxR = np.sum(g[:, :, np.newaxis] * xR[np.newaxis], axis = 3)
        gR = np.sum(g[:, :, np.newaxis] / R[np.newaxis], axis = 3)

        numerator = np.sum(H[:, np.newaxis, :, :] * gxR[:, :, np.newaxis], axis = 3)
        denominator = np.sum(H[:, np.newaxis, :, :] * gR[:, :, np.newaxis], axis = 3)
        denominator[denominator < EPS] = EPS
        W = W * np.sqrt(numerator / denominator)

        Lambda = W @ H
        R = np.sum(Lambda[...,np.newaxis] * g[:, :, np.newaxis], axis = 0)
        R[R < EPS] = EPS
        xR = x_tilde / (R ** 2)
        gxR = np.sum(g[:, :, np.newaxis] * xR[np.newaxis], axis = 3)
        gR = np.sum(g[:, :, np.newaxis] / R[np.newaxis], axis = 3)

        numerator = np.sum(W[:, :, :, np.newaxis] * gxR[:, :, np.newaxis], axis = 1)
        denominator = np.sum(W[:, :, :, np.newaxis] * gR[:, :, np.newaxis], axis = 1)
        denominator[denominator < EPS] = EPS
        H = H * np.sqrt(numerator / denominator)

        basis, activation = W, H

        # Update Q

        g = spatial_covariance
        W, H = basis, activation
        X = input.transpose(1, 2, 0)
        Q = diagonalizer

        Lambda = W @ H

        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis = 0)
        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis = 3)
        x_tilde = np.abs(QX) ** 2

        R[R < EPS] = EPS
        xR = x_tilde / (R ** 2)
        
        A = np.sum(Lambda[..., np.newaxis] * xR[np.newaxis], axis = 2)
        B = np.sum(Lambda[..., np.newaxis] / R[np.newaxis], axis = 2)
        B[B < EPS] = EPS
        g = g * np.sqrt(A / B)

        spatial_covariance = g

        X = input.transpose(1, 2, 0)
        Q = diagonalizer
        g = spatial_covariance
        XX = X[:, :, :, np.newaxis] @ X[:, :, np.newaxis, :].conj()

        W, H = basis, activation
        Lambda = W @ H

        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        R[R < EPS] = EPS
        E = np.eye(n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1))

        for channel_idx in range(n_channels):
            q_m_Hermite = Q[:, channel_idx, :]
            V = (XX / R[:, :, channel_idx, np.newaxis, np.newaxis]).mean(axis=1)
            QV = Q @ V
            condition = np.linalg.cond(QV) < THRESHOLD
            condition = condition[:,np.newaxis]
            e_m = E[:, channel_idx, :]
            q_m = np.linalg.solve(QV, e_m)
            qVq = q_m.conj()[:, np.newaxis, :] @ V @ q_m[:, :, np.newaxis]
            denominator = np.sqrt(qVq[...,0])
            denominator[denominator < EPS] = EPS
            q_m_Hermite = np.where(condition, q_m.conj() / denominator, q_m_Hermite)
            Q[:, channel_idx, :] = q_m_Hermite
        diagonalizer = Q


        Q = diagonalizer
        g = spatial_covariance
        W, H = basis, activation

        QQ = Q * Q.conj()
        QQsum = np.real(QQ.sum(axis=2).mean(axis=1))
        QQsum[QQsum < EPS] = EPS
        Q /= np.sqrt(QQsum)[:, np.newaxis, np.newaxis]
        g /= QQsum[np.newaxis, :, np.newaxis]

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
    Lambda = W @ H
    
    LambdaG = Lambda[..., np.newaxis] * g[:, :, np.newaxis, :]
    y_tilde = np.sum(LambdaG, axis=0)
    Q_inverse = np.linalg.inv(Q)
    QX = np.sum(Q[:, np.newaxis, :] * X[:, :, np.newaxis], axis=3)
    y_tilde[y_tilde < EPS] = EPS
    QXLambdaGy = QX * (LambdaG / y_tilde)
    
    x_hat = np.sum(Q_inverse[:, np.newaxis, :, :] * QXLambdaGy[:, :, :, np.newaxis, :], axis=4)
    x_hat = x_hat.transpose(0, 3, 1, 2)
    
    return x_hat[:, 0, :, :]
