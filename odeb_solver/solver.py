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
        res = A # Evita cópia desnecessária para poupar RAM
        
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

    def get_slice(self, i0):
        """
        Natively extracts a (d-1)-dimensional slice at index i0 along the first axis
        directly from the compressed cores, returning a new compressed TensorTrain.
        """
        v = self.cores[0][0, i0, :]
        core1 = self.cores[1]
        r1, n, r2 = core1.shape
        new_core = np.tensordot(v, core1, axes=(0, 0)).reshape(1, n, r2)
        sliced_cores = [new_core] + [core.copy() for core in self.cores[2:]]
        return TensorTrain(sliced_cores)

    def evaluate(self, coords):
        """
        Evaluates the Tensor Train potential at a single grid coordinate.
        """
        val = self.cores[0][0, int(round(coords[0])), :]
        for d_idx in range(1, self.d):
            val = val @ self.cores[d_idx][:, int(round(coords[d_idx])), :]
        return val[0]

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

    def __mul__(self, other):
        """
        Element-wise product of two Tensor Train objects (TT-Kronecker contraction).
        """
        assert self.d == other.d
        assert self.shape == other.shape
        new_cores = []
        for m in range(self.d):
            core1 = self.cores[m]
            core2 = other.cores[m]
            r1_l, n_m, r1_r = core1.shape
            r2_l, _, r2_r = core2.shape
            
            p = core1[:, None, :, :, None] * core2[None, :, :, None, :]
            new_core = p.reshape(r1_l * r2_l, n_m, r1_r * r2_r)
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
        Solves d-dimensional heat flow and Poisson equations natively in the
        Tensor Train domain using a memory-friendly interleaved sliding window.
        """
        # --- STEP A: THERMAL DIFFUSION VIA SINC QUADRATURE (ESA) ---
        h_sinc = 0.4
        M_sinc = 12
        quad_nodes = np.exp(np.arange(-M_sinc, M_sinc + 1) * h_sinc)
        quad_weights = h_sinc * quad_nodes
        
        u_tt = None
        for j in range(len(quad_nodes)):
            w_j = quad_weights[j]
            s_j = quad_nodes[j]
            
            cores_1d = []
            for d_idx in range(self.dimensions):
                delta_1d = np.zeros(self.grid_res)
                idx = int(round(source_coord[d_idx]))
                delta_1d[idx] = 1.0
                
                hat_delta_1d = sfft.dct(delta_1d, norm='ortho')
                hat_u_1d = hat_delta_1d * np.exp(s_j * self.t * self.laplacian_eigenvalues_1d)
                u_1d = sfft.idct(hat_u_1d, norm='ortho')
                cores_1d.append(u_1d)
                
            term_tt = TensorTrain.from_separable(cores_1d).scale(w_j * np.exp(-s_j))
            u_tt = term_tt if u_tt is None else u_tt + term_tt
            
        u_tt = u_tt.round(eps=self.eps, max_rank=self.max_rank)

        # --- STEPS B & C: SINGLE-PASS INTERLEAVED SLIDING WINDOW GRADIENT & DIVERGENCE ---
        shape_slice = tuple(self.grid_res for _ in range(self.dimensions - 1))
        div_X = np.zeros((self.grid_res,) + shape_slice, dtype=np.float32)
        
        u_slices = {}
        X_0_slices = {}
        div_internal_slices = {}
        
        def get_u_slice(idx):
            if idx < 0 or idx >= self.grid_res:
                return None
            if idx not in u_slices:
                u_slices[idx] = u_tt.get_slice(idx).to_dense().astype(np.float32)
            return u_slices[idx]

        for j in range(self.grid_res + 2):
            i = j - 1
            
            if 0 <= i < self.grid_res:
                u_prev = get_u_slice(i - 1)
                u_curr = get_u_slice(i)
                u_next = get_u_slice(i + 1)
                
                if i == 0:
                    grad_u_0 = u_next - u_curr
                elif i == self.grid_res - 1:
                    grad_u_0 = u_curr - u_prev
                else:
                    grad_u_0 = (u_next - u_prev) / 2.0
                    
                grad_u_k = []
                for k in range(1, self.dimensions):
                    g_k = np.gradient(u_curr, axis=k-1)
                    grad_u_k.append(g_k)
                    
                norm_sq = grad_u_0**2
                for g_k in grad_u_k:
                    norm_sq += g_k**2
                norm_i = np.sqrt(norm_sq + 1e-15)
                
                X_0_slices[i] = -grad_u_0 / norm_i
                
                div_internal = np.zeros_like(u_curr)
                for k in range(1, self.dimensions):
                    X_k = -grad_u_k[k-1] / norm_i
                    div_internal += np.gradient(X_k, axis=k-1)
                div_internal_slices[i] = div_internal
                
            target_i = j - 2
            if 0 <= target_i < self.grid_res:
                X0_prev = X_0_slices.get(target_i - 1)
                X0_curr = X_0_slices.get(target_i)
                X0_next = X_0_slices.get(target_i + 1)
                
                if target_i == 0:
                    grad_X0_0 = X0_next - X0_curr
                elif target_i == self.grid_res - 1:
                    grad_X0_0 = X0_curr - X0_prev
                else:
                    grad_X0_0 = (X0_next - X0_prev) / 2.0
                    
                div_X[target_i] = div_internal_slices[target_i] + grad_X0_0
                
                if target_i - 1 in u_slices:
                    del u_slices[target_i - 1]
                if target_i - 1 in X_0_slices:
                    del X_0_slices[target_i - 1]
                if target_i in div_internal_slices:
                    del div_internal_slices[target_i]

        div_tt = TensorTrain.from_dense(div_X, eps=self.eps, max_rank=self.max_rank)
        del div_X
        
        # --- STEP D: SPECTRAL POISSON INTEGRATION ---
        hat_div_tt = div_tt.dct()
        
        hat_phi = np.zeros(hat_div_tt.shape, dtype=np.float32)
        
        lap_other = np.zeros(shape_slice, dtype=np.float32)
        for d_idx in range(1, self.dimensions):
            slices = [np.newaxis] * (self.dimensions - 1)
            slices[d_idx - 1] = slice(None)
            lap_other += self.laplacian_eigenvalues_1d[tuple(slices)].astype(np.float32)
            
        for i in range(self.grid_res):
            hat_div_slice = hat_div_tt.get_slice(i).to_dense().astype(np.float32)
            lap_slice = self.laplacian_eigenvalues_1d[i] + lap_other
            
            if i == 0:
                lap_slice_safe = np.copy(lap_slice)
                zero_idx_slice = tuple([0] * (self.dimensions - 1))
                lap_slice_safe[zero_idx_slice] = 1.0
                hat_phi_slice = hat_div_slice / lap_slice_safe
                hat_phi_slice[zero_idx_slice] = 0.0
            else:
                hat_phi_slice = hat_div_slice / lap_slice
                
            hat_phi[i] = hat_phi_slice
            
        phi_tt = TensorTrain.from_dense(hat_phi, eps=self.eps, max_rank=self.max_rank).idct()
        del hat_phi
        
        phi_val_at_source = phi_tt.evaluate(source_coord)
        
        phi_dense_anchored = np.zeros(phi_tt.shape, dtype=np.float32)
        for i in range(self.grid_res):
            phi_slice = phi_tt.get_slice(i).to_dense().astype(np.float32)
            phi_slice -= phi_val_at_source
            phi_slice = np.maximum(phi_slice, 0.0)
            phi_dense_anchored[i] = phi_slice
            
        phi_tt_final = TensorTrain.from_dense(phi_dense_anchored, eps=self.eps, max_rank=self.max_rank)
        del phi_dense_anchored
        
        idx_origin = tuple(int(round(source_coord[d_idx])) for d_idx in range(self.dimensions))
        return phi_tt_final, idx_origin

    def interpolate_distances(self, phi_tt, target_coords):
        """
        Natively interpolates multi-dimensional continuous coordinates 
        directly over the core matrices of the Tensor Train.
        """
        results = []
        for coord in target_coords:
            val = None
            for d_idx in range(self.dimensions):
                y = coord[d_idx]
                y_low = int(np.floor(y))
                y_high = int(np.ceil(y))
                
                y_low = max(0, min(self.grid_res - 1, y_low))
                y_high = max(0, min(self.grid_res - 1, y_high))
                
                weight = y - y_low if y_low != y_high else 0.0
                
                core = phi_tt.cores[d_idx]
                core_low = core[:, y_low, :]
                core_high = core[:, y_high, :]
                
                core_interp = (1.0 - weight) * core_low + weight * core_high
                
                if val is None:
                    val = core_interp
                else:
                    val = val @ core_interp
            results.append(val[0, 0])
        return np.array(results)
