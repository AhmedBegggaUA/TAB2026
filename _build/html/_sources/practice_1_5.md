# SoftAssign

## Practical Session 1.5: Graph Matching with SoftAssign

## Introduction

In the previous sessions we built a pipeline for graph matching that ended with the **Hungarian Algorithm**: given a cost matrix of pairwise Euclidean distances, it finds the optimal one-to-one assignment between keypoints in $O(n^3)$.

The limitation is structural: the Hungarian Algorithm only sees **where** points are, not **how they are connected**. In this session we implement **SoftAssign**, which explicitly maximizes structural compatibility between the two graphs through the *rectangle* criterion.

---

## Theoretical Background

### The Rectangle Criterion

A rectangle is formed when four conditions hold simultaneously:

- Edge $a$–$b$ exists in $G_1$ → $X_{ab} = 1$
- Edge $i$–$j$ exists in $G_2$ → $Y_{ij} = 1$
- Node $a$ is matched to $i$ → $M_{ai} = 1$
- Node $b$ is matched to $j$ → $M_{bj} = 1$

The objective function that counts all rectangles is:

$$F(\mathbf{M}) = \frac{1}{2} \sum_{a,i,b,j} M_{ai}\, M_{bj}\, X_{ab}\, Y_{ij}$$

### The Affinity Matrix Q

For each candidate match $(a, i)$, $Q_{ai}$ counts how many rectangles it could close given the current soft matching $\mathbf{M}$:

$$Q_{ai} = \sum_{b}\sum_{j} X_{ab} \cdot M_{bj} \cdot Y_{ij} \quad \Longleftrightarrow \quad \mathbf{Q} = \mathbf{X} \cdot \mathbf{M} \cdot \mathbf{Y}$$

### Sinkhorn Normalization

After updating $\mathbf{M} \leftarrow \exp(\beta \cdot \mathbf{Q})$, rows and columns no longer sum to 1. Sinkhorn fixes this by alternating row and column normalization until convergence.

### The β Parameter and Annealing

- **Low $\beta$**: assignments are soft → the algorithm explores freely
- **High $\beta$**: assignments peak sharply → the algorithm commits to a solution

SoftAssign starts at $\beta_0$ and multiplies by $\beta_r > 1$ at each outer iteration. This is **deterministic annealing**.

---

## Algorithm Pseudocode

### Algorithm 1 — Sinkhorn

```
INPUT:  Matrix M (m×n), max_iter, ε
OUTPUT: Doubly stochastic matrix M

1. FOR t = 1, 2, …, max_iter:
    a. M_old ← M
    b. FOR each row a:   M[a,:] ← M[a,:] / sum(M[a,:])
    c. FOR each col i:   M[:,i] ← M[:,i] / sum(M[:,i])
    d. IF ||M − M_old||₁ < ε:  BREAK
2. RETURN M
```

### Algorithm 2 — SoftAssign

```
INPUT:  Adjacency matrices X (m×m), Y (n×n)
        β₀, β_f, β_r, I₀, I₁, ε
OUTPUT: Soft matching matrix M (m×n)

1. M ← ones(m,n) + 0.1·rand(m,n)
2. M ← Sinkhorn(M)
3. β ← β₀

4. WHILE β < β_f:
    a. FOR t = 1, …, I₀:
        i.   M_old ← M
        ii.  Q ← X · M · Y
        iii. M ← exp(β · Q)
        iv.  M ← Sinkhorn(M)
        v.   IF ||M − M_old||₁ < ε: BREAK
    b. β ← β · β_r

5. RETURN M
```

### Algorithm 3 — Cleanup

```
INPUT:  Soft matching matrix M (m×n)
OUTPUT: Binary matching matrix M_binary (m×n)

1. M_binary ← zeros(m,n)
2. M_work   ← copy(M)

3. FOR k = 1, …, min(m,n):
    a. (a*, i*) ← argmax(M_work)
    b. M_binary[a*, i*] ← 1
    c. M_work[a*, :] ← −∞
    d. M_work[:, i*] ← −∞

4. RETURN M_binary
```

---

## Exercises

### Exercise 1 — Implement SoftAssign

Complete the following skeleton. The utility functions `load_and_preprocess_images`, `delaunay_triangulation` and `visualize_matching_full` are the same as in Session 1.3 and are already provided.

```python
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# ── Hyperparameters ───────────────────────────────────────────────────────────
BETA_0    = 0.5
BETA_F    = 10.0
BETA_RATE = 1.075
I0        = 4
I1        = 30
EPS       = 0.5
OBJ_RESIZE = (256, 256)

# ── Provided utilities ────────────────────────────────────────────────────────
def load_and_preprocess_images(img_path1, img_path2, mat_path1, mat_path2):
    """Load images and scale keypoints to obj_resize."""
    return img1, img2, kpts1, kpts2

def delaunay_triangulation(kpt):
    """Build adjacency matrix from Delaunay triangulation."""
    return A

def visualize_matching_full(img1, img2, kpts1, kpts2, adj1, adj2, matching):
    """Visualize matching: yellow edges, green correct, red incorrect."""
    pass

# ── TO IMPLEMENT ──────────────────────────────────────────────────────────────
def sinkhorn(M, max_iter=30, eps=1e-3):
    """
    Alternate row/column normalization until M is doubly stochastic.

    Args:
        M        : (m×n) non-negative matrix
        max_iter : maximum iterations
        eps      : convergence threshold on ||M - M_old||₁

    Returns:
        M : doubly stochastic matrix (rows and columns sum to 1)

    Hints:
        - Normalize rows:    M = M / M.sum(axis=1, keepdims=True)
        - Normalize columns: M = M / M.sum(axis=0, keepdims=True)
        - Add 1e-10 in the denominator to avoid division by zero
        - Stop early if ||M - M_old||₁ < eps
    """
    return M


def softassign(X_adj, Y_adj,
               beta0=BETA_0, beta_f=BETA_F, beta_rate=BETA_RATE,
               i0=I0, i1=I1, eps=EPS):
    """
    Graph matching by maximizing structural rectangles (Gold & Rangarajan 1996).

    Args:
        X_adj     : (m×m) adjacency matrix of G1
        Y_adj     : (n×n) adjacency matrix of G2

    Returns:
        M : (m×n) soft matching matrix

    Hints:
        - Initialize: M = ones(m,n) + 0.1*rand(m,n), then Sinkhorn
        - Outer loop: while beta < beta_f, multiply beta by beta_rate at the end
        - Inner loop (i0 iterations):
            · Q = X_adj @ M @ Y_adj
            · M = exp(beta * Q)
            · M = sinkhorn(M)
            · break if ||M - M_old||₁ < eps
    """
    return M


def cleanup(M_soft):
    """
    Convert soft matching to binary via greedy assignment.

    Args:
        M_soft : (m×n) soft matching matrix

    Returns:
        M_binary : (m×n) binary matching matrix

    Hints:
        - Repeatedly find argmax of M_work
        - Assign that entry, then set its row and column to -inf
    """
    return M_binary
```

### Exercise 2 — Visualize the Result

Run SoftAssign on one Duck pair and save the result. Compare visually with the Hungarian result from Session 1.3.

```python
img1, img2, kpts1, kpts2 = load_and_preprocess_images(
    './data/WillowObject/WILLOW-ObjectClass/Duck/060_0081.png',
    './data/WillowObject/WILLOW-ObjectClass/Duck/060_0066.png',
    './data/WillowObject/WILLOW-ObjectClass/Duck/060_0081.mat',
    './data/WillowObject/WILLOW-ObjectClass/Duck/060_0066.mat'
)

A1 = delaunay_triangulation(kpts1)
A2 = delaunay_triangulation(kpts2)

M_soft   = softassign(A1, A2)
matching = cleanup(M_soft)

correct = int(np.trace(matching))
total   = int(matching.sum())
print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

fig = visualize_matching_full(img1, img2, kpts1, kpts2, A1, A2, matching)
plt.savefig('./images/duck_softassign.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Exercise 3 — Hyperparameter Tuning

For each of the 5 Willow categories, select **one image pair** where the default hyperparameters give a poor result. Tune the hyperparameters until you achieve an accuracy gain of at least **+40 percentage points** on that pair.

**Hyperparameters to explore:**

| Parameter | Default | What it controls |
|---|---|---|
| `BETA_0` | 0.5 | Initial softness — how uniform the starting assignments are |
| `BETA_F` | 10.0 | How binary the final solution is |
| `BETA_RATE` | 1.075 | Speed of annealing — how fast β grows |
| `I0` | 4 | Inner iterations per β step |

**For each of the 5 categories, deliver:**

1. The two image filenames used.
2. A side-by-side figure: matching **before** tuning (default params) and **after** tuning.
3. The following table filled in:

| | Default | Tuned |
|---|---|---|
| `BETA_0` | 0.5 | |
| `BETA_F` | 10.0 | |
| `BETA_RATE` | 1.075 | |
| `I0` | 4 | |
| **Accuracy** | % | % |

> **Where to start:** vary `BETA_F` first — it has the largest effect. If the result is already sharp but wrong, slow down the annealing by reducing `BETA_RATE`.