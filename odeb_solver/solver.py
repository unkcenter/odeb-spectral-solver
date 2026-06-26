import numpy as np
import scipy.fft as sfft
from scipy.interpolate import RegularGridInterpolator

class SpectralHeatSolver2D:
    """
    Spectral solver based on the anisotropic heat method utilizing DCT-II
    to enforce stable Neumann boundary conditions (reflective walls).
    """
    def __init__(self, grid_res=128, t=20.0):
        self.grid_res = grid_res
        self.t = t
        self.x_grid = np.arange(grid_res)
        self.y_grid = np.arange(grid_res)
        
        # Precompute eigenvalues of the discrete Neumann Laplacian (DCT-II)
        kx = np.pi * np.arange(grid_res) / (2.0 * grid_res)
        ky = np.pi * np.arange(grid_res) / (2.0 * grid_res)
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        self.laplacian_eigenvalues = -4 * (np.sin(Kx)**2 + np.sin(Ky)**2)
        
        # Prepare safe eigenvalues matrix to prevent division by zero at the DC component
        self.laplacian_eigenvalues_safe = np.copy(self.laplacian_eigenvalues)
        self.laplacian_eigenvalues_safe[0, 0] = 1.0

    @staticmethod
    def _dct2(a):
        return sfft.dct(sfft.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

    @staticmethod
    def _idct2(a):
        return sfft.idct(sfft.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

    def solve_geodesic_distance(self, source_coord):
        """
        Solves the anisotropic heat diffusion and Poisson integration 
        in the spectral domain using DCT-II.
        """
        # Step 1: Inoculate heat source (Dirac delta) at the source pixel
        delta_grid = np.zeros((self.grid_res, self.grid_res))
        idx_x = int(round(source_coord[0]))
        idx_y = int(round(source_coord[1]))
        delta_grid[idx_x, idx_y] = 1.0
        
        # Step 2: Solve anisotropic heat diffusion in the spectral cosine domain
        hat_delta = self._dct2(delta_grid)
        hat_u = hat_delta / (1.0 - self.t * self.laplacian_eigenvalues)
        u = self._idct2(hat_u)
        u = np.maximum(u, 1e-15)  # Filter numerical underflow
        
        # Step 3: Extract and normalize spatial gradient
        grad_ux, grad_uy = np.gradient(u)
        grad_norm = np.sqrt(grad_ux**2 + grad_uy**2 + 1e-15)
        X_x = -grad_ux / grad_norm
        X_y = -grad_uy / grad_norm
        
        # Step 4: Compute divergence of the vector field
        div_X = np.gradient(X_x, axis=0) + np.gradient(X_y, axis=1)
        
        # Step 5: Solve the exact Poisson integration spectral equation
        hat_div = self._dct2(div_X)
        hat_phi = hat_div / self.laplacian_eigenvalues_safe
        hat_phi[0, 0] = 0.0  # Filter out DC component
        phi = self._idct2(hat_phi)
        
        # Anchor potential to zero at source
        phi -= phi[idx_x, idx_y]
        phi = np.maximum(phi, 0.0)
        return phi, (idx_x, idx_y)

    def interpolate_distances(self, phi, target_coords):
        """
        Interpolates computed distances on the grid back to coordinate points.
        """
        interp = RegularGridInterpolator((self.x_grid, self.y_grid), phi, method='linear')
        return interp(target_coords)
