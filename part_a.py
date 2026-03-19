"""
Part A - Concept Application
Week 04, Day 21 - NumPy Assignment
"""

import numpy as np

# ─────────────────────────────────────────────
# Q1. Arrays of different dimensions + indexing
# ─────────────────────────────────────────────

print("=" * 55)
print("Q1. Creating 1D, 2D, 3D arrays and indexing/slicing")
print("=" * 55)

# 1D array
arr_1d = np.array([10, 20, 30, 40, 50, 60])
print("\n1D array:", arr_1d)
print("  Element at index 2   :", arr_1d[2])
print("  Slice [1:4]          :", arr_1d[1:4])
print("  Last two elements    :", arr_1d[-2:])

# 2D array
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print("\n2D array:\n", arr_2d)
print("  Row 1 (index 1)      :", arr_2d[1])
print("  Column 2             :", arr_2d[:, 2])
print("  Subarray [0:2, 1:3]  :\n", arr_2d[0:2, 1:3])
print("  Element [2][3]       :", arr_2d[2, 3])

# 3D array
arr_3d = np.arange(1, 25).reshape(2, 3, 4)
print("\n3D array (2x3x4):\n", arr_3d)
print("  First block [0]:\n", arr_3d[0])
print("  Second block, row 1  :", arr_3d[1, 1])
print("  Element [1][2][3]    :", arr_3d[1, 2, 3])
print("  Subarray [0, 0:2, 1:3]:\n", arr_3d[0, 0:2, 1:3])


# ─────────────────────────────────────────────
# Q2. Basic operations without loops
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q2. Element-wise operations, mean, variance, std dev")
print("=" * 55)

a = np.array([4, 8, 15, 16, 23, 42])
b = np.array([2, 4,  5,  6,  7,  8])

print("\nArray a:", a)
print("Array b:", b)
print("  Addition       :", a + b)
print("  Subtraction    :", a - b)
print("  Multiplication :", a * b)

print("\nStatistics on array a:")
print("  Mean           :", np.mean(a))
print("  Variance       :", np.var(a))
print("  Std deviation  :", np.std(a))


# ─────────────────────────────────────────────
# Q3. Broadcasting
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q3. Broadcasting demonstrations")
print("=" * 55)

# Case 1: Add 1D array to 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vec = np.array([10, 20, 30])

print("\nMatrix (3x3):\n", matrix)
print("1D array:", row_vec)
result1 = matrix + row_vec
print("Matrix + 1D array (row_vec added to each row):\n", result1)
print("  → row_vec shape (3,) is broadcast to (3,3) by repeating across rows.")

# Case 2: Multiply matrix by scalar
scalar = 5
result2 = matrix * scalar
print(f"\nMatrix * scalar ({scalar}):\n", result2)
print("  → Scalar is broadcast to match (3,3) shape, every element gets multiplied.")

# Case 3: Multiply matrix by column vector
col_vec = np.array([[2], [3], [4]])   # shape (3,1)
result3 = matrix * col_vec
print("\nMatrix * column vector (shape 3x1):\n", result3)
print("  → col_vec shape (3,1) is broadcast to (3,3) by repeating across columns.")


# ─────────────────────────────────────────────
# Q4. Vectorised operations
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q4. Vectorised operations")
print("=" * 55)

data = np.array([-3, -1, 0, 2, 4, 6, 8])
print("\nOriginal array:", data)

# Square and cube
print("  Square         :", data ** 2)
print("  Cube           :", data ** 3)

# Replace negatives with 0
clipped = np.where(data < 0, 0, data)
print("  Negatives → 0  :", clipped)

# Normalize (min-max scaling to [0, 1])
def normalize(arr):
    """Scale array values to [0, 1] range."""
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

print("  Normalized     :", normalize(data))


# ─────────────────────────────────────────────
# Q5. Dataset operations
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("Q5. Dataset operations")
print("=" * 55)

np.random.seed(42)
dataset = np.random.randint(0, 100, size=(5, 6))
print("\nDataset (5x6):\n", dataset)

# Top 5 maximum values
flat_top5 = np.sort(dataset.flatten())[-5:][::-1]
print("\nTop 5 maximum values:", flat_top5)

# Row-wise and column-wise sums
print("Row-wise sums   :", dataset.sum(axis=1))
print("Column-wise sums:", dataset.sum(axis=0))

# Indices where value > threshold
threshold = 70
indices = np.argwhere(dataset > threshold)
print(f"\nIndices where value > {threshold}:")
for idx in indices:
    print(f"  Row {idx[0]}, Col {idx[1]} → value = {dataset[idx[0], idx[1]]}")
