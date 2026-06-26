import numpy as np

class BourgainMDSEmbedding:
    """
    Manages the projection of discrete distance matrices into isotropic,
    multi-dimensional continuous coordinates on a regular grid.
    """
    def __init__(self, dimensions=2, padding_factor=0.35):
        self.dimensions = dimensions
        self.padding_factor = padding_factor
        self.R = None  # Decoupling rotation matrix (PCA)
        self.min_val = None
        self.max_val = None
        self.geom_scale = None

    def fit_transform(self, dist_matrix, grid_res=128):
        """
        Computes multi-dimensional Classical MDS, aligns coordinates using 
        eigenvectors of the covariance matrix (PCA decoupling), and scales 
        them isotropically to a safe padded region of a regular grid.
        """
        n = dist_matrix.shape[0]
        # Cap dimensions dynamically to avoid numerical instabilities with zero or negative eigenvalues
        d_eff = min(self.dimensions, n - 1)
        
        # Double centering the distance matrix
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (dist_matrix**2) @ H
        
        # Spectral decomposition
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # Keep only the top positive eigenvalues
        pos_idx = np.where(evals > 1e-12)[0]
        d_actual = min(d_eff, len(pos_idx))
        
        if d_actual == 0:
            raise ValueError("Distance matrix has no positive eigenvalues.")
            
        L = np.diag(np.sqrt(evals[pos_idx[:d_actual]]))
        V = evecs[:, pos_idx[:d_actual]]
        coords_embedded = V @ L
        
        # Rotational Decoupling (Diagonalization of the Metric Tensor via PCA)
        if d_actual > 1:
            cov = np.cov(coords_embedded.T)
            _, self.R = np.linalg.eigh(cov)
            coords_rotated = coords_embedded @ self.R
        else:
            self.R = np.eye(1)
            coords_rotated = coords_embedded

        # Isotropic Scaling to preserve absolute aspect ratios of the metric space
        self.min_val = coords_rotated.min(axis=0)
        self.max_val = coords_rotated.max(axis=0)
        self.geom_scale = np.max(self.max_val - self.min_val)
        
        if self.geom_scale < 1e-12:
            self.geom_scale = 1.0  # Avoid division by zero for single-point grids
            
        # Dynamically calculate safe boundaries to avoid boundary reflection artifacts
        lower_bound = self.padding_factor * grid_res
        upper_bound = (1.0 - self.padding_factor) * grid_res
        span = upper_bound - lower_bound
        
        # Isotropically map coordinates into the safe centered region
        coords_grid = lower_bound + span * (coords_rotated - self.min_val) / self.geom_scale
        return coords_grid

    def transform_distance(self, computed_distances, original_distances_to_source):
        """
        Computes the mathematically exact Least-Squares scale factor 
        to minimize L2 error propagation.
        """
        # Exclude the source point (distance = 0) from the scale factor average to prevent division by zero
        mask = original_distances_to_source > 1e-12
        if not np.any(mask):
            return np.zeros_like(computed_distances)
            
        numerator = np.sum(original_distances_to_source * computed_distances)
        denominator = np.sum(computed_distances**2) + 1e-15
        scale_factor = numerator / denominator
        return computed_distances * scale_factor
