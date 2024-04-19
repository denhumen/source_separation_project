
if 1:
    import numpy as np
    from nmf import *

    A = np.random.rand(3, 3)

    print('A', A)
    rank = 3
    max_iter = int(1e5)
    W_mu_H_mu = multiplicative_update(A, rank, max_iter)
    print('WH', W_mu_H_mu)

if 1:
    import numpy as np
    X = A
    from sklearn.decomposition import NMF
    model = NMF(n_components=2, init='nndsvd', random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    print('WH built-in', W@H)
