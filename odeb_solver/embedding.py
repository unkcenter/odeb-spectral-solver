import numpy as np

class BourgainMDSEmbedding:
    """
    Manages the projection of discrete distance matrices into isotropic continuous coordinates.
    """
    def __init__(self, dimensions=2):
        self.dimensions = dimensions
        self.R = None  # Decoupling rotation matrix
        self.min_val = None
        self.max_val = None
        self.geom_scale = None

    def fit_transform(self, dist_matrix):
        """
        Computes classical MDS and aligns coordinates using eigenvectors
        of the covariance matrix (PCA decoupling) while strictly preserving
        the isotropic metric ratios.
        """
        n = dist_matrix.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (dist_matrix**2) @ H
        
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        w = np.where(evals > 0)[0]
        L = np.diag(np.sqrt(evals[w[:self.dimensions]]))
        V = evecs[:, w[:self.dimensions]]
        coords_embedded = V @ L
        
        # Rotational Decoupling (Diagonalization of the Metric Tensor)
        cov = np.cov(coords_embedded.T)
        _, self.R = np.linalg.eigh(cov)
        coords_rotated = coords_embedded @ self.R
        
        # Isotropic Scaling (aspect ratio preservation)
        self.min_val = coords_rotated.min(axis=0)
        self.max_val = coords_rotated.max(axis=0)
        self.geom_scale = np.max(self.max_val - self.min_val)
        
        # Centralize in the grid's safe zone [32, 96] for a 128 resolution grid
        coords_grid = 32.0 + 64.0 * (coords_rotated - self.min_val) / self.geom_scale
        return coords_grid

    def transform_distance(self, computed_distances, original_distances_to_source):
        """
        Scales computed distances back to the physical unit (e.g. km).
        """
        scale_factor = np.mean(original_distances_to_source[1:]) / np.mean(computed_distances[1:])
        return computed_distances * scale_factor
