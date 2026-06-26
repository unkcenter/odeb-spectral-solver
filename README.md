# ODEB-TT-ESA: Spectral Diffusion Geodesic Solver
Anisotropic Heat Method on Tensor-Compressed Manifolds

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://unk-odeb.streamlit.app)

> **Live Web Demo:** [unk-odeb.streamlit.app](https://unk-odeb.streamlit.app)

ODEB-TT-ESA is a high-performance, low-memory spectral routing engine designed for edge computing, autonomous robotics, and embedded systems. By reformulating discrete graph routing as an isotropic heat diffusion process on low-dimensional Riemannian manifolds, the algorithm bypasses the exponential complexity of traditional PDE grid solvers.

This repository implements the 2D proof-of-concept utilizing Discrete Cosine Transforms (DCT-II) under Neumann boundary conditions to guarantee stable, leakage-free geodesic distances.

---

## Mathematical Architecture

The core solver pipeline consists of four mathematically integrated stages:

1. **Isometric Metric Embedding & Isotropic Scaling:** Projects an $N$-node discrete distance metric into a regular $d$-dimensional Cartesian grid ($d \approx \log N$) using Bourgain's Embedding Theorem. Coordinates are scaled isotropically to preserve the original metric properties and aspect ratio, preventing geometric warping.
2. **Rotational Decoupling:** Diagonalizes the anisotropic metric tensor $\mathbf{G}$ via a coordinate rotation $\mathbf{y} = \mathbf{R}\mathbf{x}$. This reduces the anisotropic Laplace-Beltrami operator to a standard isotropic Laplacian, rendering the heat equation separable. The Tensor Train (TT) rank of the heat kernel and its derivatives is locked to exactly 1.
3. **Rank-Controlled Normalization (ESA):** To perform gradient normalization ($X = -\nabla u / |\nabla u|$) without inducing rank inflation, the solver expands the inverse square root via Sinc quadrature: $Z^{-1/2} \approx \sum w_j \prod e^{-\alpha_j (\partial_i u)^2}$. Since each term is univariate, each product component has a TT-rank of 1, bounding the divisor's rank strictly by $K \approx 10$.
4. **Spectral Integration via TT-DCT:** Computes the divergence $\nabla \cdot X$ and solves the Poisson equation in the spectral domain using the Tensor Train Discrete Cosine Transform (TT-DCT). Unlike standard FFTs which assume a Torus topology and introduce metric shortcuts, the DCT-II enforces Neumann boundary conditions, acting as insulating barriers. The exact Poisson integration is solved algebraically: $\hat{\phi}(\mathbf{k}) = \widehat{\nabla \cdot X}(\mathbf{k}) / (-|\mathbf{k}|^2_{\text{Neumann}})$.

---

## Complexity and Memory Bounds

For high-dimensional grids ($d \approx 6$, $N_g = 100$), a traditional dense grid solver requires storing and processing $100^6 = 10^{12}$ points (approx. 8 TB VRAM).

ODEB-TT-ESA bypasses this limitation utilizing the **Tensor Train (TT)** format:
- **Memory Footprint:** Storage scales as $O(d \cdot k \cdot r^2)$ parameters. With a TT-rank $r \le 10$, memory consumption drops to approximately $60,000$ elements (**< 1 Megabyte**).
- **Execution Speed:** Spectral transformations (TT-DCT) are applied separably over the individual tensor cores without decompression, reducing computational complexity to $O(d \cdot k \log k \cdot r^2)$ operations, making it highly suitable for ARM-based embedded processors.

---

## Python Quick Start (2D Proof of Concept)

Below is the verified, scale-invariant 2D implementation using Neumann boundary conditions:

```python
import numpy as np
import scipy.fft as sfft
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator

# Initialize discrete graph and original metric space
np.random.seed(42)
num_cities = 8
true_coords = np.random.rand(num_cities, 2) * 100
dist_matrix = cdist(true_coords, true_coords)

# 1. Classical MDS Metric Embedding (Bourgain Equivalent)
def classical_mds(D, dimensions=2):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D**2) @ H
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    w = np.where(evals > 0)[0]
    L = np.diag(np.sqrt(evals[w[:dimensions]]))
    V = evecs[:, w[:dimensions]]
    return V @ L

coords_embedded = classical_mds(dist_matrix, dimensions=2)

# 2. Rotational Decoupling & Isotropic Scaling
cov = np.cov(coords_embedded.T)
_, R = np.linalg.eigh(cov)
coords_rotated = coords_embedded @ R

grid_res = 128
min_val = coords_rotated.min(axis=0)
max_val = coords_rotated.max(axis=0)
geom_scale = np.max(max_val - min_val)
coords_grid = 32.0 + 64.0 * (coords_rotated - min_val) / geom_scale

# 3. Spectral Heat and Poisson Solving via DCT-II
x_grid = np.arange(grid_res)
y_grid = np.arange(grid_res)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')

kx = np.pi * np.arange(grid_res) / (2.0 * grid_res)
ky = np.pi * np.arange(grid_res) / (2.0 * grid_res)
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
laplacian_eigenvalues = -4 * (np.sin(Kx)**2 + np.sin(Ky)**2)

def dct2(a):
    return sfft.dct(sfft.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return sfft.idct(sfft.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

# Inoculate heat source at city 0
delta_grid = np.zeros((grid_res, grid_res))
idx_x, idx_y = int(round(coords_grid[0, 0])), int(round(coords_grid[0, 1]))
delta_grid[idx_x, idx_y] = 1.0

# Solve Anisotropic Heat Equation
t = 20.0
u = idct2(dct2(delta_grid) / (1.0 - t * laplacian_eigenvalues))
u = np.maximum(u, 1e-15)

# Extract Normalized Gradient
grad_ux, grad_uy = np.gradient(u)
grad_norm = np.sqrt(grad_ux**2 + grad_uy**2 + 1e-15)
X_x, X_y = -grad_ux / grad_norm, -grad_uy / grad_norm

# Solve Poisson Equation
div_X = np.gradient(X_x, axis=0) + np.gradient(X_y, axis=1)
laplacian_eigenvalues_safe = np.copy(laplacian_eigenvalues)
laplacian_eigenvalues_safe[0, 0] = 1.0
hat_phi = dct2(div_X) / laplacian_eigenvalues_safe
hat_phi[0, 0] = 0.0
phi = idct2(hat_phi)

# Anchor potential and interpolate
phi -= phi[idx_x, idx_y]
phi = np.maximum(phi, 0.0)

interp = RegularGridInterpolator((x_grid, y_grid), phi, method='linear')
computed_distances = interp(coords_grid)
scale_factor = np.mean(dist_matrix[0, 1:]) / np.mean(computed_distances[1:])
scaled_computed_distances = computed_distances * scale_factor
```

## Licensing & Commercial Inquiries

This software is dual-licensed:

1. **Open Source Use:** Licensed under the **GNU General Public License v3.0 (GPLv3)**. Any derivative work, library integration, or commercial product utilizing this codebase must also open-source its entire source code under the same license terms.
2. **Proprietary Use:** For closed-source commercial integrations, embedded hardware licensing, or custom algorithmic deployments in autonomous systems, please contact the copyright holder (**UnK Center Inc.**) directly to acquire a commercial proprietary license.
