# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a simple script to demonstrate the use of timing functions.
"""

import numpy as np
# timing (default_timer chooses the right timer for the version of Python in use)
from timeit import default_timer as timer

""" build some large array """

L = 10**2
vec = np.ones([L,L], dtype=float)

""" test einstein summation function vs dot product function """

print("performing timing tests...")

start = timer()
t1 = np.einsum('ji,jk,kl', vec, vec, vec)
elapsed = timer() - start
print("einsum (no broadcasting): ",elapsed)

start = timer()
t2 = np.dot(np.transpose(vec),np.dot(vec,vec))
elapsed = timer() - start
print("dot (broadcasting): ",elapsed)

print("compare operations: " , (t2 - t1).max())