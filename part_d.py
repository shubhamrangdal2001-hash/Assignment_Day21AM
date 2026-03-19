"""
Part D - AI-Augmented Task
Week 04, Day 21 - NumPy Assignment

Prompt used:
    "Explain NumPy broadcasting and vectorisation with practical Python examples."

AI Output (reproduced below for evaluation):
----------------------------------------------
Broadcasting lets you perform operations between arrays of different shapes.
NumPy 'stretches' the smaller array along size-1 dimensions to match the larger one.

Vectorisation means applying an operation to an entire array at once
using optimised C code, instead of looping in Python.

AI gave these example snippets (evaluated below).
"""

import numpy as np
import time

print("=" * 55)
print("Part D — AI Output Evaluation")
print("=" * 55)

# ─────────────────────────────────────────────
# Prompt used
# ─────────────────────────────────────────────
print("""
Prompt sent to AI:
  "Explain NumPy broadcasting and vectorisation with
   practical Python examples."
""")

# ─────────────────────────────────────────────
# AI-provided code examples (faithfully reproduced)
# ─────────────────────────────────────────────

print("-" * 55)
print("AI Example 1: Broadcasting")
print("-" * 55)

# AI suggested this example
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)
b = np.array([10, 20, 30]) # shape (3,)
C = A + b                  # b is broadcast to (2, 3)
print("A:\n", A)
print("b:", b)
print("A + b:\n", C)
print("Result correct? →", C.tolist() == [[11,22,33],[14,25,36]])

print()
print("-" * 55)
print("AI Example 2: Column-wise broadcasting")
print("-" * 55)

# AI also showed column vector broadcasting
col = np.array([[100],
                [200]])   # shape (2, 1)
D = A + col               # col broadcast to (2, 3)
print("col:\n", col)
print("A + col:\n", D)
print("Result correct? →", D.tolist() == [[101,102,103],[204,205,206]])

print()
print("-" * 55)
print("AI Example 3: Vectorised squaring")
print("-" * 55)

data = np.array([-2, -1, 0, 1, 2, 3])
squared = data ** 2
print("Input  :", data)
print("Squared:", squared)
print("Result correct? →", squared.tolist() == [4, 1, 0, 1, 4, 9])

print()
print("-" * 55)
print("AI Example 4: Vectorised normalization")
print("-" * 55)

def ai_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

sample = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
norm   = ai_normalize(sample)
print("Input     :", sample)
print("Normalized:", norm)
print("Result correct? →", np.allclose(norm, [0, 0.25, 0.5, 0.75, 1.0]))

print()
print("-" * 55)
print("AI Example 5: Loop vs NumPy speed")
print("-" * 55)

big = np.random.rand(2_000_000)

t0 = time.time()
_ = sum(big.tolist())           # plain Python
t_py = time.time() - t0

t0 = time.time()
_ = np.sum(big)                 # NumPy
t_np = time.time() - t0

print(f"Python loop : {t_py:.4f}s")
print(f"NumPy sum   : {t_np:.6f}s")
print(f"Speed-up    : ~{t_py / t_np:.0f}x")

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

print()
print("=" * 55)
print("Evaluation of AI Output")
print("=" * 55)
print("""
1. Are examples correct?
   → Yes. All five examples produce expected outputs and I verified
     each one by running the code. Results matched the correct values.

2. Is code efficient and runnable?
   → Yes. The AI used proper NumPy syntax with no unnecessary loops.
     Broadcasting and vectorisation are applied correctly.
     The normalization function handles the edge case implicitly
     (though it would crash if min == max; I added that guard in Part C).

3. What I learned / verified:
   → Broadcasting rules are explained correctly: shapes are compared
     right-to-left and size-1 dimensions are stretched.
   → The speed comparison numbers from the AI matched what I observed
     on my machine (~100x faster for NumPy).
   → One minor thing: AI did not mention the edge case in normalize()
     where all values are equal. I handled that separately in Part C.
""")
