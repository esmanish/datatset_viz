import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

def load_label_files(directory):
    """Load all label files from a directory"""
    all_objects = []
    frame_ids = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.startswith('.'):
            file_path = os.path.join(directory, filename)
            frame_id = filename.split('.')[0]
            
            try:
                with open(file_path, 'r') as f:
                    objects = json.load(f)
                    
                # Add frame_id to each object
                for obj in objects:
                    obj['frame_id'] = frame_id
                
                all_objects.extend(objects)
                frame_ids.append(frame_id)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return all_objects, frame_ids

def calculate_spatial_niche_overlap(df):
    """Calculate Pianka's Overlap Index between different vehicle types based on spatial distribution"""
    print("Calculating spatial niche overlap between vehicle types...")
    
    vehicle_types = df['obj_type'].unique()
    overlap_matrix = pd.DataFrame(0.0, index=vehicle_types, columns=vehicle_types, dtype=np.float64)
    
    # Create spatial distribution vectors for each type
    # Divide space into grid cells and count objects per cell
    x_bins = np.linspace(df['pos_x'].min(), df['pos_x'].max(), 20)
    y_bins = np.linspace(df['pos_y'].min(), df['pos_y'].max(), 20)
    
    # Store distributions for visualization
    all_distributions = {}
    
    # Calculate distribution for each vehicle type
    for v_type in vehicle_types:
        type_df = df[df['obj_type'] == v_type]
        hist, x_edges, y_edges = np.histogram2d(
            type_df['pos_x'], type_df['pos_y'], 
            bins=[x_bins, y_bins]
        )
        # Normalize to get probability distribution
        if hist.sum() > 0:
            hist = hist / hist.sum()
        all_distributions[v_type] = hist
        
    # Calculate Pianka's overlap index for each pair
    for type1 in vehicle_types:
        for type2 in vehicle_types:
            p1 = all_distributions[type1].flatten()
            p2 = all_distributions[type2].flatten()
            
            # Pianka's formula: overlap = sum(p1i * p2i) / sqrt(sum(p1i^2) * sum(p2i^2))
            numerator = np.sum(p1 * p2)
            denominator = np.sqrt(np.sum(p1**2) * np.sum(p2**2))
            
            if denominator > 0:
                overlap_matrix.loc[type1, type2] = float(numerator / denominator)
            else:
                overlap_matrix.loc[type1, type2] = 0.0
    
    # Debug: Check matrix data type
    print(f"Matrix data type: {overlap_matrix.dtypes.iloc[0]}")
    
    # Ensure all values are numeric
    overlap_matrix = overlap_matrix.astype(float)
    
    return overlap_matrix, all_distributions, x_bins, y_bins

def visualize_results(overlap_matrix, all_distributions, x_bins, y_bins, df):
    """Create visualizations from the ecological analysis"""
    
    # 1. Overlap heatmap
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy array to ensure it's numeric
    overlap_array = overlap_matrix.values.astype(float)
    
    # Check for any non-finite values
    if not np.isfinite(overlap_array).all():
        overlap_array = np.nan_to_num(overlap_array)
    
    # Create heatmap with numpy array
    sns.heatmap(overlap_array, annot=True, cmap='viridis', vmin=0, vmax=1,
               xticklabels=overlap_matrix.columns, 
               yticklabels=overlap_matrix.index)
    
    plt.title("Spatial Niche Overlap Between Vehicle Types (Pianka's Index)")
    plt.tight_layout()
    plt.savefig('spatial_niche_overlap.png', dpi=300)
    
    # 2. Spatial distribution heatmaps for each vehicle type
    n_types = len(all_distributions)
    n_cols = min(3, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    for i, (v_type, hist) in enumerate(all_distributions.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(hist.T, origin='lower', aspect='auto', 
                   extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                   cmap='hot')
        plt.colorbar(label='Probability')
        plt.title(f'{v_type} Spatial Distribution')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
    
    plt.tight_layout()
    plt.savefig('vehicle_type_distributions.png', dpi=300)
    
    # 3. Calculate and visualize niche breadth (spatial area used)
    niche_breadth = {}
    for v_type, hist in all_distributions.items():
        # Shannon's diversity index as measure of niche breadth
        flat_hist = hist.flatten()
        nonzero = flat_hist[flat_hist > 0]
        if len(nonzero) > 0:
            shannon = -np.sum(nonzero * np.log(nonzero))
            niche_breadth[v_type] = shannon
        else:
            niche_breadth[v_type] = 0
    
    # Sort by breadth
    niche_breadth = {k: v for k, v in sorted(niche_breadth.items(), 
                                            key=lambda item: item[1], reverse=True)}
    
    plt.figure(figsize=(10, 6))
    plt.bar(niche_breadth.keys(), niche_breadth.values())
    plt.title('Spatial Niche Breadth by Vehicle Type')
    plt.ylabel("Shannon's Diversity Index")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('niche_breadth.png', dpi=300)
    
    # 4. Calculate and visualize niche position (centroid)
    centroids = {}
    for v_type in all_distributions.keys():
        type_df = df[df['obj_type'] == v_type]
        centroids[v_type] = (type_df['pos_x'].mean(), type_df['pos_y'].mean())
    
    plt.figure(figsize=(10, 8))
    # Plot all points as background
    plt.scatter(df['pos_x'], df['pos_y'], c='lightgray', alpha=0.3, s=10)
    
    # Plot centroids
    for v_type, (x, y) in centroids.items():
        plt.scatter(x, y, s=100, label=v_type)
    
    plt.legend()
    plt.title('Spatial Niche Centroids by Vehicle Type')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('niche_centroids.png', dpi=300)
    
    return niche_breadth, centroids

def calculate_competition_indices(overlap_matrix, df):
    """Calculate competition indices for each vehicle type"""
    # Get counts of each type
    type_counts = df['obj_type'].value_counts()
    
    # Calculate competition pressure on each type
    competition_indices = {}
    for v_type in overlap_matrix.index:
        # Weight overlap by abundance of other types
        competition = 0
        for other_type in overlap_matrix.columns:
            if other_type != v_type and other_type in type_counts.index:
                competition += float(overlap_matrix.loc[v_type, other_type]) * float(type_counts[other_type])
                
        competition_indices[v_type] = competition
    
    # Sort by competition index
    competition_indices = {k: v for k, v in sorted(competition_indices.items(), 
                                                  key=lambda item: item[1], reverse=True)}
    
    plt.figure(figsize=(10, 6))
    plt.bar(competition_indices.keys(), competition_indices.values())
    plt.title('Spatial Competition Index by Vehicle Type')
    plt.ylabel('Competition Index (Higher = More Competition)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('competition_indices.png', dpi=300)
    
    return competition_indices

def main():
    # Path to your label files directory
    label_directory = "/home/shuttle_01/Pictures/Lidar-Tiand/TIAND_LiDAR-object detection/2nd_september/Scene4/label"
    
    # Load data
    print("Loading label files...")
    all_objects, frame_ids = load_label_files(label_directory)
    
    # Convert to DataFrame
    df = pd.json_normalize(all_objects)
    
    # Extract position for easier access
    df['pos_x'] = df.apply(lambda row: row['psr.position.x'], axis=1)
    df['pos_y'] = df.apply(lambda row: row['psr.position.y'], axis=1)
    df['pos_z'] = df.apply(lambda row: row['psr.position.z'], axis=1)
    
    # Calculate spatial niche overlap
    overlap_matrix, all_distributions, x_bins, y_bins = calculate_spatial_niche_overlap(df)
    
    # Visualize results
    print("Creating visualizations...")
    niche_breadth, centroids = visualize_results(overlap_matrix, all_distributions, x_bins, y_bins, df)
    
    # Calculate competition indices
    competition_indices = calculate_competition_indices(overlap_matrix, df)
    
    # Print results
    print("\n--- Ecological Traffic Analysis Results ---")
    print("\nSpatial Niche Overlap (Higher values indicate more spatial overlap):")
    print(overlap_matrix.round(3))
    
    print("\nSpatial Niche Breadth (Higher values indicate wider spatial distribution):")
    for v_type, breadth in niche_breadth.items():
        print(f"{v_type}: {breadth:.4f}")
        
    print("\nSpatial Competition Index (Higher values indicate more competition for space):")
    for v_type, comp in competition_indices.items():
        print(f"{v_type}: {comp:.4f}")
    
    print("\nAnalysis complete! Visualizations saved to current directory.")

if __name__ == "__main__":
    main()