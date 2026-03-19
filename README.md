# Week 04 · Day 21 AM — NumPy Assignment

**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**
Gitlink : https://github.com/shubhamrangdal2001-hash/Assignment_Day21AM.git
## Overview

This assignment covers NumPy fundamentals including array operations, broadcasting, vectorisation, matrix math, and performance comparison between loops and NumPy.

---

## Folder Structure

```
numpy_assignment/
├── part_a.py       # Concept Application — arrays, slicing, stats, broadcasting, vectorisation
├── part_b.py       # Stretch Problem — matrix ops, linear equations, loop vs NumPy speed
├── part_c.py       # Interview Ready — broadcasting explanation + normalize() function
├── part_d.py       # AI-Augmented Task — AI prompt, output, and evaluation
└── README.md
```

---

## How to Run

**Requirements:** Python 3.8+, NumPy

Install NumPy if not already installed:
```bash
pip install numpy
```

Run each part separately:
```bash
python part_a.py
python part_b.py
python part_c.py
python part_d.py
```

---

## Topics Covered

| Part | Topics |
|------|--------|
| A    | 1D/2D/3D arrays, indexing, slicing, broadcasting, vectorisation, dataset operations |
| B    | Matrix multiplication, transpose, determinant, linear equations, performance benchmark |
| C    | Broadcasting explanation, `normalize()` function, vectorisation vs loops |
| D    | AI prompt engineering, output verification, evaluation |

---

## Key Concepts

**Broadcasting** — NumPy can operate on arrays of different shapes by implicitly expanding size-1 dimensions. No data is copied; it is just viewed differently.

**Vectorisation** — Applying operations to full arrays using compiled C code instead of Python loops. Much faster due to SIMD CPU instructions and no Python object overhead.

**normalize(X)** — Min-max scaling: `(X - X.min()) / (X.max() - X.min())` — scales all values to [0, 1].
