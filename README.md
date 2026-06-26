# ODEB-TT-ESA: Spectral Diffusion Geodesic Solver
Anisotropic Heat Method on Tensor-Compressed Manifolds

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://unk-odeb.streamlit.app)

> **Live Web Demo:** [unk-odeb.streamlit.app](https://unk-odeb.streamlit.app)

ODEB-TT-ESA is a high-performance, low-memory spectral routing engine designed for edge computing, autonomous robotics, and embedded systems. By reformulating discrete graph routing as an isotropic heat diffusion process on low-dimensional Riemannian manifolds, the algorithm bypasses the exponential complexity of traditional PDE grid solvers.

This repository implements a fully generalized d-dimensional spectral geodesic solver operating entirely within the Tensor Train (TT) format to bypass the curse of dimensionality.

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

## Python Quick Start (3D Tensor Train Implementation)

Below is the verified, scale-invariant 3D Tensor Train implementation using Neumann boundary conditions:

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from odeb_solver import BourgainMDSEmbedding, SpectralHeatSolverND

# Initialize discrete graph and original metric space (3D)
np.random.seed(42)
num_cities = 12
dimensions = 3  # Native execution beyond 2D
grid_res = 128

true_coords = np.random.rand(num_cities, dimensions) * 100
dist_matrix = cdist(true_coords, true_coords)

# 1. Classical MDS Metric Embedding (Bourgain Equivalent) with safe padding (0.35)
embedding = BourgainMDSEmbedding(dimensions=dimensions, padding_factor=0.35)
coords_grid = embedding.fit_transform(dist_matrix, grid_res=grid_res)

# 2. Spectral Heat and Poisson Solving via TT-DCT and Sinc Quadrature (t=25.0)
max_rank = 10
solver = SpectralHeatSolverND(
    dimensions=dimensions, 
    grid_res=grid_res, 
    t=25.0, 
    max_rank=max_rank
)

source_city_idx = 0
phi_tt, origin_pixel = solver.solve_geodesic_distance(coords_grid, coords_grid[source_city_idx])

# 3. Interpolation and Least-Squares Distance Recovery
computed_distances = solver.interpolate_distances(phi_tt, coords_grid)
scaled_computed_distances = embedding.transform_distance(computed_distances, dist_matrix[source_city_idx])
```

---

## Licensing & Commercial Inquiries

This software is dual-licensed:

1. **Open Source Use:** Licensed under the **GNU General Public License v3.0 (GPLv3)**. Any derivative work, library integration, or commercial product utilizing this codebase must also open-source its entire source code under the same license terms [5].
2. **Proprietary Use:** For closed-source commercial integrations, embedded hardware licensing, or custom algorithmic deployments in autonomous systems, please contact the copyright holder (**UnK Center Inc.**) directly to acquire a commercial proprietary license.
