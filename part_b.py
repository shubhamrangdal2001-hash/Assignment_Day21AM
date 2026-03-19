"""
Part B - Stretch Problem
Week 04, Day 21 - NumPy Assignment
"""

import numpy as np
import time

# ─────────────────────────────────────────────
# Q1. Matrix operations
# ─────────────────────────────────────────────

print("=" * 55)
print("Q1. Matrix Operations")
print("=" * 55)

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

print("\nMatrix A:\n", A)
print("\nMatrix B:\n", B)

# Matrix multiplication
print("\nMatrix Multiplication (A @ B):\n", A @ B)

# Transpose
print("\nTranspose of A:\n", A.T)

# Determinant
# Using a non-singular matrix for meaningful determinant
C = np.array([[2, 1, 3],
              [0, 4, 1],
              [5, 2, 8]])
print("\nMatrix C (for determinant):\n", C)
det_C = np.linalg.det(C)
print(f"Determinant of C: {det_C:.4f}")


# ─────────────────────────────────────────────
# Q2. Solve a system of linear equations
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q2. System of Linear Equations")
print("=" * 55)

# Equations:
#   2x + y - z  = 8
#  -3x - y + 2z = -11
#  -2x + y + 2z = -3

coeff = np.array([[ 2,  1, -1],
                  [-3, -1,  2],
                  [-2,  1,  2]], dtype=float)

constants = np.array([8, -11, -3], dtype=float)

print("\nEquations:")
print("  2x  +  y - z  =  8")
print(" -3x  -  y + 2z = -11")
print(" -2x  +  y + 2z = -3")

# Solve using numpy
solution = np.linalg.solve(coeff, constants)
x, y, z = solution
print(f"\nSolution: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}")

# Verify
print("Verification (coeff @ solution):", coeff @ solution)
print("Expected constants              :", constants)


# ─────────────────────────────────────────────
# Q3. Performance: Python loop vs NumPy
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q3. Performance Comparison: Loop vs NumPy")
print("=" * 55)

N = 10_000_000
large_array = np.random.rand(N)
large_list  = large_array.tolist()   # plain Python list for loop test

# Python loop
start = time.time()
total_loop = 0.0
for val in large_list:
    total_loop += val
loop_time = time.time() - start

# NumPy sum
start = time.time()
total_numpy = np.sum(large_array)
numpy_time = time.time() - start

print(f"\nArray size   : {N:,} elements")
print(f"Python loop  : sum = {total_loop:.4f}  | time = {loop_time:.4f}s")
print(f"NumPy sum    : sum = {total_numpy:.4f}  | time = {numpy_time:.6f}s")
print(f"Speed-up     : NumPy is ~{loop_time / numpy_time:.0f}x faster")

print("\nWhy NumPy is faster:")
print("  - NumPy operations run in compiled C code, not interpreted Python.")
print("  - It uses vectorised SIMD instructions to process multiple elements at once.")
print("  - No Python object overhead per element — data is stored as raw C floats.")
