import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from odeb_solver import BourgainMDSEmbedding, SpectralHeatSolver2D

def main():
    # 1. Generate discrete graph (8 random cities and their distance matrix)
    np.random.seed(42)
    num_cities = 8
    true_coords = np.random.rand(num_cities, 2) * 100  # 100km x 100km space
    dist_matrix = cdist(true_coords, true_coords)       # Ground Truth distances
    
    # 2. Initialize and fit the Isotropic Bourgain/MDS Embedding
    embedding = BourgainMDSEmbedding(dimensions=2)
    coords_grid = embedding.fit_transform(dist_matrix)
    
    # 3. Initialize the Spectral Heat Solver (Grid 128x128, t=20.0)
    solver = SpectralHeatSolver2D(grid_res=128, t=20.0)
    
    # 4. Solve the geodesic distance field starting from City 0
    source_city_idx = 0
    phi, (idx_x, idx_y) = solver.solve_geodesic_distance(coords_grid[source_city_idx])
    
    # 5. Interpolate distances back to the city grid coordinates
    computed_distances = solver.interpolate_distances(phi, coords_grid)
    
    # 6. Scale the calculated distances back to the original physical unit (km)
    scaled_distances = embedding.transform_distance(computed_distances, dist_matrix[source_city_idx])
    
    # 7. Print comparative results
    print("="*65)
    print(f" SPECTRALLY COMPUTED DISTANCES FROM ORIGIN (CITY {source_city_idx})")
    print("="*65)
    for idx in range(num_cities):
        real_d = dist_matrix[source_city_idx, idx]
        calc_d = scaled_distances[idx]
        error_pct = abs(real_d - calc_d) / (real_d + 1e-9) * 100 if idx != source_city_idx else 0.0
        print(f"City {idx}: Real = {real_d:6.2f} km | ODEB Solver = {calc_d:6.2f} km | Error = {error_pct:5.2f}%")
    print("="*65)
    
    # 8. Plot the continuous geodesic distance field
    grid_res = 128
    x_grid = np.arange(grid_res)
    y_grid = np.arange(grid_res)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    plt.figure(figsize=(11, 9))
    plt.contourf(X_mesh, Y_mesh, phi, levels=40, cmap='viridis')
    plt.colorbar(label='Geodesic Distance Potential (Pixels)')
    
    # Plot cities
    plt.scatter(coords_grid[:, 0], coords_grid[:, 1], color='red', s=120, zorder=5, label='Cities')
    for idx in range(num_cities):
        plt.text(coords_grid[idx, 0]+1.5, coords_grid[idx, 1]+1.5, f"C{idx}", color='white', weight='bold', fontsize=12, zorder=10)
        
    # Highlight source city
    source_coord = coords_grid[source_city_idx]
    plt.scatter(source_coord[0], source_coord[1], color='cyan', s=200, edgecolors='black', linewidth=2, zorder=6, label=f'Origin (C{source_city_idx})')
    
    plt.title('ODEB - Distance Field via DCT (Neumann) and Isotropic Scaling', fontsize=14, pad=15)
    plt.xlabel('Grid Coordinate X', fontsize=12)
    plt.ylabel('Grid Coordinate Y', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
