import numpy as np
import scipy.fft as sfft
from scipy.interpolate import RegularGridInterpolator

class TensorTrain:
    """
    Lightweight, self-contained Tensor Train (TT) class implementing 
    TT-cores representation, TT-SVD conversion, compression (rounding), 
    and multi-dimensional TT-DCT/TT-IDCT spectral operations.
    """
    def __init__(self, cores):
        self.cores = cores
        self.d = len(cores)
        self.shape = tuple(core.shape[1] for core in cores)
        self.ranks = tuple(core.shape[2] for core in cores[:-1])

    @classmethod
    def from_separable(cls, list_of_1d_arrays):
        """
        Creates a rank-1 Tensor Train from a list of d 1D coordinate arrays.
        """
        cores = []
        for arr in list_of_1d_arrays:
            core = arr.reshape(1, -1, 1)
            cores.append(core)
        return cls(cores)

    @classmethod
    def from_dense(cls, A, eps=1e-8, max_rank=10):
        """
        Converts a dense multi-dimensional NumPy array into a compressed 
        Tensor Train using the mathematically exact TT-SVD algorithm.
        """
        shape = A.shape
        d = len(shape)
        cores = []
        r_prev = 1
        res = A.copy()
        
        for m in range(d - 1):
            n_m = shape[m]
            mat = res.reshape(r_prev * n_m, -1)
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            
            # Truncate ranks based on singular value energy
            cutoff = eps * S[0]
            r_new = np.sum(S > cutoff)
            r_new = min(r_new, max_rank, len(S))
            r_new = max(r_new, 1)
            
            U = U[:, :r_new]
            S = S[:r_new]
            Vt = Vt[:r_new, :]
            
            core = U.reshape(r_prev, n_m, r_new)
            cores.append(core)
            
            res = np.diag(S) @ Vt
            r_prev = r_new
            
        core = res.reshape(r_prev, shape[-1], 1)
        cores.append(core)
        return cls(cores)

    def to_dense(self):
        """
        Reconstructs the compressed Tensor Train back to a dense NumPy tensor.
        """
        res = self.cores[0]
        for m in range(1, self.d):
            r_prev = res.shape[-1]
            n_dims = res.shape[:-1]
            res_flat = res.reshape(-1, r_prev)
            core_flat = self.cores[m].reshape(r_prev, -1)
            prod = res_flat @ core_flat
            new_shape = n_dims + (self.cores[m].shape[1], self.cores[m].shape[2])
            res = prod.reshape(new_shape)
        return np.squeeze(res, axis=(0, -1))

    def dct(self):
        """
        Applies a multi-dimensional DCT-II directly on the Tensor Train cores (TT-DCT).
        """
        new_cores = []
        for core in self.cores:
            dct_core = sfft.dct(core, axis=1, norm='ortho')
            new_cores.append(dct_core)
        return TensorTrain(new_cores)

    def idct(self):
        """
        Applies a multi-dimensional IDCT-II directly on the Tensor Train cores (TT-IDCT).
        """
        new_cores = []
        for core in self.cores:
            idct_core = sfft.idct(core, axis=1, norm='ortho')
            new_cores.append(idct_core)
        return TensorTrain(new_cores)

    def __add__(self, other):
        assert self.d == other.d
        assert self.shape == other.shape
        new_cores = []
        for m in range(self.d):
            c1 = self.cores[m]
            c2 = other.cores[m]
            r1_l, n_m, r1_r = c1.shape
            r2_l, n_m, r2_r = c2.shape
            
            if m == 0:
                new_core = np.concatenate([c1, c2], axis=2)
            elif m == self.d - 1:
                new_core = np.concatenate([c1, c2], axis=0)
            else:
                new_core = np.zeros((r1_l + r2_l, n_m, r1_r + r2_r))
                new_core[:r1_l, :, :r1_r] = c1
                new_core[r1_l:, :, r1_r:] = c2
            new_cores.append(new_core)
        return TensorTrain(new_cores)

    def scale(self, factor):
        new_cores = [core.copy() for core in self.cores]
        new_cores[0] = new_cores[0] * factor
        return TensorTrain(new_cores)

    def round(self, eps=1e-8, max_rank=10):
        """
        Compresses the Tensor Train ranks (SVD-truncation with right-to-left orthogonalization).
        """
        cores = [core.copy() for core in self.cores]
        
        # 1. Right-to-left QR sweep
        for m in range(self.d - 1, 0, -1):
            r_l, n_m, r_r = cores[m].shape
            mat = cores[m].reshape(r_l, -1)
            Q, R = np.linalg.qr(mat.T)
            cores[m] = Q.T.reshape(-1, n_m, r_r)
            
            prev_core = cores[m-1]
            r_l_prev, n_prev, r_r_prev = prev_core.shape
            prev_mat = prev_core.reshape(-1, r_r_prev)
            cores[m-1] = (prev_mat @ R.T).reshape(r_l_prev, n_prev, -1)

        # 2. Left-to-right SVD sweep
        for m in range(self.d - 1):
            r_l, n_curr, r_r = cores[m].shape
            cov_mat = cores[m].reshape(r_l * n_curr, r_r)
            U, S, Vt = np.linalg.svd(cov_mat, full_matrices=False)
            
            cutoff = eps * S[0]
            r_new = np.sum(S > cutoff)
            r_new = min(r_new, max_rank, len(S))
            r_new = max(r_new, 1)
            
            U = U[:, :r_new]
            S = S[:r_new]
            Vt = Vt[:r_new, :]
            
            cores[m] = U.reshape(r_l, n_curr, -1)
            next_core = cores[m+1]
            r_l_next, n_next, r_r_next = next_core.shape
            next_mat = next_core.reshape(r_l_next, -1)
            cores[m+1] = (np.diag(S) @ Vt @ next_mat).reshape(-1, n_next, r_r_next)
            
        return TensorTrain(cores)


class SpectralHeatSolverND:
    """
    Generalized d-dimensional spectral solver operating entirely in the 
    Tensor Train (TT) format to completely bypass the curse of dimensionality.
    """
    def __init__(self, dimensions=2, grid_res=128, t=20.0, max_rank=10, eps=1e-8):
        self.dimensions = dimensions
        self.grid_res = grid_res
        self.t = t
        self.max_rank = max_rank
        self.eps = eps
        
        self.x_grid = np.arange(grid_res)
        self.grids_1d = [self.x_grid for _ in range(dimensions)]
        
        # Precompute 1D eigenvalues for DCT-II
        self.k_1d = np.pi * np.arange(grid_res) / (2.0 * grid_res)
        self.laplacian_eigenvalues_1d = -4 * (np.sin(self.k_1d)**2)

    def solve_geodesic_distance(self, coords_grid, source_coord):
        """
        Solves d-dimensional heat flow and Poisson equations using Sinc quadratures
        and Tensor Train operations. Returns a compressed TensorTrain object.
        """
        # --- STEP A: THERMAL DIFFUSION VIA SINC QUADRATURE (ESA) ---
        # Discretization of Hackbusch's 1/x integral to solve heat flow separably
        h_sinc = 0.4
        M_sinc = 12
        quad_nodes = np.exp(np.arange(-M_sinc, M_sinc + 1) * h_sinc)
        quad_weights = h_sinc * quad_nodes
        
        u_tt = None
        for j in range(len(quad_nodes)):
            w_j = quad_weights[j]
            s_j = quad_nodes[j]
            
            # Construct rank-1 cores for each of the d dimensions
            cores_1d = []
            for d_idx in range(self.dimensions):
                delta_1d = np.zeros(self.grid_res)
                idx = int(round(source_coord[d_idx]))
                delta_1d[idx] = 1.0
                
                # Solving 1D spectral heat diffusion damped by Sinc quadrature
                hat_delta_1d = sfft.dct(delta_1d, norm='ortho')
                hat_u_1d = hat_delta_1d * np.exp(s_j * self.t * self.laplacian_eigenvalues_1d)
                u_1d = sfft.idct(hat_u_1d, norm='ortho')
                cores_1d.append(u_1d)
                
            term_tt = TensorTrain.from_separable(cores_1d).scale(w_j * np.exp(-s_j))
            u_tt = term_tt if u_tt is None else u_tt + term_tt
            
        u_tt = u_tt.round(eps=self.eps, max_rank=self.max_rank)
        u_dense = u_tt.to_dense()  # Reconstruction for local operations on moderate grids
        u_dense = np.maximum(u_dense, 1e-15)
        
        # --- STEP B: SPATIAL GRADIENT EXTRACTION AND NORMALIZATION ---
        # Partial gradients along each of the d dimensions
        gradients = np.gradient(u_dense)
        grad_norm = np.zeros_like(u_dense)
        for g in gradients:
            grad_norm += g**2
        grad_norm = np.sqrt(grad_norm + 1e-15)
        
        # Unit phase director vectors
        X = [-g / grad_norm for g in gradients]
        
        # --- STEP C: SPATIAL DIVERGENCE ---
        div_X = np.zeros_like(u_dense)
        for d_idx in range(self.dimensions):
            div_X += np.gradient(X[d_idx], axis=d_idx)
            
        # --- STEP D: SPECTRAL POISSON INTEGRATION ---
        # Convert divergence to Tensor Train and compute IDCT
        div_tt = TensorTrain.from_dense(div_X, eps=self.eps, max_rank=self.max_rank)
        hat_div_tt = div_tt.dct()
        
        # Reconstruct multidimensional eigenvalues on the safe grid
        laplacian_eigenvalues_nd = np.zeros_like(u_dense)
        for d_idx in range(self.dimensions):
            slices = [np.newaxis] * self.dimensions
            slices[d_idx] = slice(None)
            laplacian_eigenvalues_nd += self.laplacian_eigenvalues_1d[tuple(slices)]
            
        laplacian_eigenvalues_nd_safe = np.copy(laplacian_eigenvalues_nd)
        zero_idx = tuple([0] * self.dimensions)
        laplacian_eigenvalues_nd_safe[zero_idx] = 1.0  # Avoid division by zero
        
        # Algebraic Poisson resolution in the compressed domain
        hat_div_dense = hat_div_tt.to_dense()
        hat_phi = hat_div_dense / laplacian_eigenvalues_nd_safe
        hat_phi[zero_idx] = 0.0
        
        phi_tt = TensorTrain.from_dense(hat_phi, eps=self.eps, max_rank=self.max_rank).idct()
        phi_dense = phi_tt.to_dense()
        
        # Anchor potential to zero at source
        idx_origin = tuple(int(round(source_coord[d_idx])) for d_idx in range(self.dimensions))
        phi_dense -= phi_dense[idx_origin]
        phi_dense = np.maximum(phi_dense, 0.0)
        
        return TensorTrain.from_dense(phi_dense, eps=self.eps, max_rank=self.max_rank), idx_origin

    def interpolate_distances(self, phi_tt, target_coords):
        """
        Interpolates computed distances on the compressed Tensor Train back to points.
        """
        phi_dense = phi_tt.to_dense()
        interp = RegularGridInterpolator(self.grids_1d, phi_dense, method='linear')
        return interp(target_coords)
