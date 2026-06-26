import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from odeb_solver import BourgainMDSEmbedding, SpectralHeatSolverND

# Set up page configurations for a clean, modern dashboard
st.set_page_config(
    page_title="ODEB-TT-ESA: Spectral Routing Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header
st.title("ODEB-TT-ESA")
st.subheader("Anisotropic Spectral Heat Diffusion on Tensor-Compressed Manifolds")
st.write(
    "This interactive dashboard demonstrates the real-time execution of the "
    "multidimensional Spectral Geodesic Solver, utilizing Tensor Train (TT) "
    "compression to bypass the Curse of Dimensionality."
)

# Sidebar Configuration Controls
st.sidebar.header("Execution Parameters")

num_cities = st.sidebar.slider("Number of Cities (N)", min_value=5, max_value=30, value=12, step=1)
dimensions = st.sidebar.selectbox("Manifold Dimensions (d)", options=[3, 4, 5, 6], index=0)
grid_res = st.sidebar.select_slider("Grid Resolution per Axis", options=[64, 128], value=128)
t_val = st.sidebar.slider("Diffusion Time (t)", min_value=5.0, max_value=100.0, value=25.0, step=5.0)
max_rank = st.sidebar.slider("Tensor Train Max Rank (r)", min_value=2, max_value=20, value=10, step=1)

# Execution Trigger
if st.sidebar.button("Run Spectral Optimization", type="primary"):
    
    # 1. Generate Synthetic Metric Space
    np.random.seed(42)
    true_coords = np.random.rand(num_cities, dimensions) * 100
    dist_matrix = cdist(true_coords, true_coords)
    
    # 2. Run Bourgain MDS Metric Embedding
    embedding = BourgainMDSEmbedding(dimensions=dimensions, padding_factor=0.35)
    with st.spinner("Computing Isotropic Metric Embedding..."):
        coords_grid = embedding.fit_transform(dist_matrix, grid_res=grid_res)
    
    # 3. Initialize and Execute the N-Dimensional Tensor Train Solver
    solver = SpectralHeatSolverND(
        dimensions=dimensions, 
        grid_res=grid_res, 
        t=t_val, 
        max_rank=max_rank
    )
    
    source_city_idx = 0
    with st.spinner("Solving Multidimensional Spectral Heat & Poisson Equations..."):
        phi_tt, origin_pixel = solver.solve_geodesic_distance(coords_grid, coords_grid[source_city_idx])
    
    # 4. Interpolate and Scale Geodesic Distances
    computed_distances = solver.interpolate_distances(phi_tt, coords_grid)
    scaled_computed_distances = embedding.transform_distance(computed_distances, dist_matrix[source_city_idx])
    
    # 5. Extract Evaluation Metrics
    real_dists = dist_matrix[source_city_idx, :]
    calc_dists = scaled_computed_distances

    mask_non_source = np.arange(num_cities) != source_city_idx
    mean_error_pct = np.mean(np.abs(real_dists[mask_non_source] - calc_dists[mask_non_source]) / real_dists[mask_non_source]) * 100
    correlation, _ = spearmanr(real_dists, calc_dists)

    # Calculate Compression Footprint
    dense_elements = grid_res ** dimensions
    tt_elements = sum(core.size for core in phi_tt.cores)
    compression_ratio = dense_elements / tt_elements
    
    # --- UI DISPLAY ---
    # Metric KPI Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Mean Geodesic Error", 
            value=f"{mean_error_pct:.2f}%", 
            delta="Target: < 5.0%" if mean_error_pct < 5.0 else "Needs Parameter Tuning",
            delta_color="normal" if mean_error_pct < 5.0 else "inverse"
        )
    with col2:
        st.metric(
            label="Spearman Rank Correlation (rho)", 
            value=f"{correlation:.4f}", 
            delta="Target: > 0.95" if correlation > 0.95 else "Low Correlation",
            delta_color="normal" if correlation > 0.95 else "inverse"
        )
    with col3:
        st.metric(
            label="RAM/VRAM Compression Factor", 
            value=f"{compression_ratio:.1f}x", 
            delta="Bypassing Dense Grid Limit",
            delta_color="off"
        )
        
    # Graphical and Tabular Results
    layout_col1, layout_col2 = st.columns([3, 2])
    
    with layout_col1:
        st.write("### Geodesic Distance Field Visualization")
        try:
            slice_z = int(round(coords_grid[source_city_idx, 2])) if dimensions >= 3 else 0
            
            if dimensions >= 3:
                # Extrai a fatia 2D diretamente dos núcleos do Trem de Tensores de forma nativa e ultra-rápida (Z = slice_z)
                fixed_coords = [int(round(coords_grid[source_city_idx, k])) for k in range(dimensions)]
                fixed_coords[2] = slice_z # Garante a fatia no nível Z do ponto de origem
                
                # Multiplica todos os eixos fixados da direita para a esquerda (k >= 2)
                v_right = phi_tt.cores[-1][:, fixed_coords[-1], :]  # shape: (r_{d-1}, 1)
                for k in range(dimensions - 2, 1, -1):
                    core_val = phi_tt.cores[k][:, fixed_coords[k], :]  # shape: (r_k, r_{k+1})
                    v_right = core_val @ v_right  # shape: (r_{k-1}, 1)
                
                v_right = v_right.squeeze(axis=1)  # shape: (r_1,)
                
                # Contrai o núcleo do segundo eixo (Y) com o vetor direito consolidado
                C1 = phi_tt.cores[1]  # shape: (r_1, N_g, r_2)
                C1_reduced = np.tensordot(C1, v_right, axes=(2, 0))  # shape: (r_1, N_g)
                
                # Multiplica o núcleo do primeiro eixo (X) com a representação contraída de Y
                C0 = phi_tt.cores[0][0, :, :]  # shape: (N_g, r_1)
                phi_slice = C0 @ C1_reduced  # shape: (N_g, N_g)
            else:
                phi_slice = phi_tt.to_dense()
                
            x_grid = np.arange(grid_res)
            y_grid = np.arange(grid_res)
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing='ij')
            
            fig, ax = plt.subplots(figsize=(8, 6.5))
            contour = ax.contourf(X_mesh, Y_mesh, phi_slice, levels=40, cmap='viridis')
            fig.colorbar(contour, ax=ax, label='Geodesic Distance Potential (Grid Pixels)')
            
            # Scatter coordinates projected to the slice plane
            ax.scatter(coords_grid[:, 0], coords_grid[:, 1], color='red', s=80, zorder=5, label='Nodes')
            for idx in range(num_cities):
                ax.text(coords_grid[idx, 0]+1.5, coords_grid[idx, 1]+1.5, f"C{idx}", color='white', weight='bold', fontsize=10, zorder=10)
                
            source_coord = coords_grid[source_city_idx]
            ax.scatter(source_coord[0], source_coord[1], color='cyan', s=150, edgecolors='black', linewidth=1.5, zorder=6, label='Source (C0)')
            
            ax.set_title(f'2D Slice of the d-Dimensional Distance Field (Slice plane Z={slice_z})')
            ax.set_xlabel('Grid Coordinate X')
            ax.set_ylabel('Grid Coordinate Y')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Visualization rendering error: {e}")
            
    with layout_col2:
        st.write("### Comparative Routing Table")
        st.write("Verification of scaled geodesic distances against exact Euclidean distances:")
        
        # Build pandas dataframe for clean UI presentation
        results_data = []
        for idx in range(num_cities):
            real_val = real_dists[idx]
            calc_val = calc_dists[idx]
            error_val = abs(real_val - calc_val) / (real_val + 1e-9) * 100 if idx != source_city_idx else 0.0
            results_data.append({
                "City Index": f"City {idx:02d}",
                "Real Distance (km)": f"{real_val:.2f}",
                "Computed (TT-Solver)": f"{calc_val:.2f}",
                "Relative Error": f"{error_val:.2f}%"
            })
            
        st.table(results_data)
        
    # Memory and Algorithmic Explanation
    st.write("---")
    st.write("### Data Compression Breakdown")
    st.info(
        f"A standard dense grid representation for d={dimensions} with a resolution of {grid_res} "
        f"requires the evaluation of **{dense_elements:,}** spatial elements. By compressing the distance field "
        f"using the **Tensor Train** format, the active parameter space is reduced to just **{tt_elements:,}** elements "
        f"in the core matrices. This prevents exponential memory explosion and ensures O(d * k log k * r^2) computational scalability."
    )
else:
    st.info("Configure your parameters in the sidebar and click 'Run Spectral Optimization' to start.")
