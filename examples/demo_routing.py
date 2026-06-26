import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from odeb_solver import BourgainMDSEmbedding, SpectralHeatSolverND

def main():
    # 1. Generate discrete graph (12 random cities in 3D space)
    np.random.seed(42)
    num_cities = 12
    dimensions = 3  # Native execution beyond 2D
    grid_res = 128

    true_coords = np.random.rand(num_cities, dimensions) * 100  # 100km x 100km x 100km space
    dist_matrix = cdist(true_coords, true_coords)       # Ground Truth distances
    
    # 2. Initialize and fit the Isotropic Bourgain/MDS Embedding with safe padding (0.35)
    embedding = BourgainMDSEmbedding(dimensions=dimensions, padding_factor=0.35)
    coords_grid = embedding.fit_transform(dist_matrix, grid_res=grid_res)
    
    # 3. Initialize the d-dimensional Spectral Heat Solver in Tensor Train format (t=25.0)
    max_rank = 10
    solver = SpectralHeatSolverND(dimensions=dimensions, grid_res=grid_res, t=25.0, max_rank=max_rank)
    
    # 4. Solve the geodesic distance field starting from City 0
    source_city_idx = 0
    phi_tt, origin_pixel = solver.solve_geodesic_distance(coords_grid, coords_grid[source_city_idx])
    
    # 5. Interpolate distances back to the city grid coordinates
    computed_distances = solver.interpolate_distances(phi_tt, coords_grid)
    
    # 6. Scale the calculated distances back to the original physical unit (km) using Least-Squares
    scaled_computed_distances = embedding.transform_distance(computed_distances, dist_matrix[source_city_idx])
    
    # 7. Compute comparative metrics
    real_dists = dist_matrix[source_city_idx, :]
    calc_dists = scaled_computed_distances

    mask_non_source = np.arange(num_cities) != source_city_idx
    mean_error_pct = np.mean(np.abs(real_dists[mask_non_source] - calc_dists[mask_non_source]) / real_dists[mask_non_source]) * 100
    correlation, _ = spearmanr(real_dists, calc_dists)

    # Volumetric memory footprint comparison (Dense vs Tensor Train)
    dense_elements = grid_res ** dimensions
    tt_elements = sum(core.size for core in phi_tt.cores)
    compression_ratio = dense_elements / tt_elements

    # Print comparative results in English
    print("="*75)
    print(" CONVERGENCE METRICS: OPTIMIZED MULTIDIMENSIONAL ODEB-TT-ESA SOLVER")
    print("="*75)
    print(f"Projection Space Dimensions (d):            {dimensions}")
    print(f"Regular Grid Resolution per Axis:           {grid_res}")
    print(f"Mean Geodesic Distance Error:               {mean_error_pct:.2f}%")
    print(f"Ranking Correlation (Spearman rho):         {correlation:.4f}")
    print("-"*75)
    print(" MEMORY FOOTPRINT COMPARISON")
    print("-"*75)
    print(f"Dense Grid Elements (Tridimensional):       {dense_elements:,}")
    print(f"Tensor Train (TT) Grid Elements:            {tt_elements:,}")
    print(f"Achieved Data Compression Ratio:            {compression_ratio:.1f}x less VRAM/RAM")
    print("="*75)

    print("\n" + "="*75)
    print(" COMPARATIVE TABLE OF OPTIMIZED PATHS (km)")
    print("="*75)
    for idx in range(num_cities):
        real_d = real_dists[idx]
        calc_d = calc_dists[idx]
        erro_pct = abs(real_d - calc_d) / (real_d + 1e-9) * 100 if idx != source_city_idx else 0.0
        print(f"City {idx:02d}: Real = {real_d:6.2f} km | Calculated (TT) = {calc_d:6.2f} km | Error = {erro_pct:5.2f}%")
    print("="*65)
    
    # 8. Optional Plotting (Visualizing a 2D slice for 3D coordinate space)
    try:
        print("\nPlotting a 2D slice of the 3D distance field...")
        phi_dense = phi_tt.to_dense()
        slice_z = int(round(coords_grid[source_city_idx, 2]))
        phi_slice = phi_dense[:, :, slice_z]
        
        x_grid = np.arange(grid_res)
        y_grid = np.arange(grid_res)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        plt.figure(figsize=(11, 9))
        plt.contourf(X_mesh, Y_mesh, phi_slice, levels=40, cmap='viridis')
        plt.colorbar(label='Geodesic Distance Potential Slice (Z-index)')
        
        # Plot cities
        plt.scatter(coords_grid[:, 0], coords_grid[:, 1], color='red', s=120, zorder=5, label='Cities')
        for idx in range(num_cities):
            plt.text(coords_grid[idx, 0]+1.5, coords_grid[idx, 1]+1.5, f"C{idx}", color='white', weight='bold', fontsize=12, zorder=10)
            
        # Highlight source city
        source_coord = coords_grid[source_city_idx]
        plt.scatter(source_coord[0], source_coord[1], color='cyan', s=200, edgecolors='black', linewidth=2, zorder=6, label=f'Origin (C{source_city_idx})')
        
        plt.title(f'ODEB-TT - 2D Slice of 3D Geodesic Field (Z={slice_z})', fontsize=14, pad=15)
        plt.xlabel('Grid Coordinate X', fontsize=12)
        plt.ylabel('Grid Coordinate Y', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"Could not render plot: {e}")

if __name__ == "__main__":
    main()
