"""
Part C - Interview Ready
Week 04, Day 21 - NumPy Assignment
"""

import numpy as np

# ─────────────────────────────────────────────
# Q1 - What is NumPy broadcasting? Why is it useful?
# ─────────────────────────────────────────────

print("=" * 55)
print("Q1. NumPy Broadcasting — Explanation")
print("=" * 55)

print("""
Broadcasting means NumPy can perform arithmetic on arrays of
different shapes without actually copying data to make them the
same size.

Rules:
  1. Compare shapes from the trailing (rightmost) dimension.
  2. Dimensions are compatible if they are equal OR one of them is 1.
  3. The smaller dimension is 'stretched' to match the larger one.

Why is it useful?
  - Avoids writing explicit loops, so code is shorter and faster.
  - No extra memory is used to duplicate data.
  - Makes operations like 'add a bias vector to every row of a matrix'
    very easy to write.
""")

# Quick demo
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])        # shape (2, 3)
bias = np.array([10, 20, 30])        # shape (3,)
result = matrix + bias               # bias broadcast to (2, 3)
print("Matrix (2x3) + bias vector (3,) — broadcasting in action:")
print(result)


# ─────────────────────────────────────────────
# Q2 - Coding: normalize(X)
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q2. normalize(X) — Scale values between 0 and 1")
print("=" * 55)

def normalize(X):
    """
    Min-max normalization: scales every value in X to [0, 1].
    Formula: (x - min) / (max - min)
    Works for 1D and multi-dimensional NumPy arrays.
    """
    X = np.array(X, dtype=float)
    x_min = X.min()
    x_max = X.max()

    # Edge case: all values are the same → return zeros
    if x_max == x_min:
        return np.zeros_like(X)

    return (X - x_min) / (x_max - x_min)


# Test 1: 1D array
arr1 = np.array([2, 5, 1, 8, 3])
print("\nInput  :", arr1)
print("Output :", normalize(arr1))

# Test 2: 2D array
arr2 = np.array([[10, 20], [30, 40]])
print("\nInput  :\n", arr2)
print("Output :\n", normalize(arr2))

# Test 3: Edge case — all same values
arr3 = np.array([7, 7, 7])
print("\nEdge case (all 7s):", normalize(arr3))


# ─────────────────────────────────────────────
# Q3 - Vectorisation vs Loops — Why is NumPy faster?
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q3. Vectorisation vs Loops")
print("=" * 55)

print("""
Loops (Python):
  - Python executes one instruction at a time.
  - Each element is a Python object carrying type info and reference
    counts — a lot of overhead per element.
  - The interpreter is involved at every iteration.

Vectorisation (NumPy):
  - Operations are applied to the whole array at once.
  - Implemented in compiled C/Fortran under the hood.
  - CPU SIMD instructions (SSE/AVX) handle multiple elements in a
    single CPU instruction.
  - No per-element Python overhead.

Bottom line: for large arrays, NumPy can be 100x–1000x faster than
a plain Python loop.

Example comparison:
""")

import time

n = 5_000_000
data = np.random.rand(n)

# Loop
t0 = time.time()
sq_loop = [x**2 for x in data]
t_loop = time.time() - t0

# Vectorised
t0 = time.time()
sq_vec = data ** 2
t_vec = time.time() - t0

print(f"  Squaring {n:,} elements:")
print(f"    Python list comprehension : {t_loop:.4f}s")
print(f"    NumPy vectorised          : {t_vec:.6f}s")
print(f"    Speed-up                  : ~{t_loop/t_vec:.0f}x")
