import numpy as np
from nmf import *


# Generate a random matrix A
A = np.random.rand(10, 10)

# Test random_initialization
rank = 3
W_rand, H_rand = random_initialization(A, rank)
print("Random Initialization:")
print("W:\n", W_rand)
print("H:\n", H_rand)
print()

# Test nndsvd_initialization
W_nndsvd, H_nndsvd = nndsvd_initialization(A, rank)
print("NNDSVD Initialization:")
print("W:\n", W_nndsvd)
print("H:\n", H_nndsvd)
print()

# Test multiplicative_update
max_iter = 100
W_mu, H_mu, norms = multiplicative_update(A, rank, max_iter)
print("Multiplicative Update:")
print("W:\n", W_mu)
print("H:\n", H_mu)
print("Frobenius Norms:\n", norms)
