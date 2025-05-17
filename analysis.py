import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

def load_label_files(directory):
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

def advanced_analysis(df):
    """Perform advanced analysis on the dataset"""
    print("\n--- Advanced Spatial Analysis ---")
    
    # Extract positions as numpy array
    positions = df[['pos_x', 'pos_y']].values
    
    # 1. Proximity Analysis: Calculate distances between all objects
    distance_matrix = squareform(pdist(positions))
    np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances
    
    # Find closest neighbor for each object
    min_distances = np.min(distance_matrix, axis=1)
    min_indices = np.argmin(distance_matrix, axis=1)
    
    # Create dataframe with object pairs
    proximity_data = []
    for i in range(len(df)):
        obj1 = df.iloc[i]
        obj2 = df.iloc[min_indices[i]]
        
        proximity_data.append({
            'obj1_id': obj1['obj_id'],
            'obj1_type': obj1['obj_type'],
            'obj2_id': obj2['obj_id'],
            'obj2_type': obj2['obj_type'],
            'distance': min_distances[i],
            'frame_id': obj1['frame_id']
        })
    
    proximity_df = pd.DataFrame(proximity_data)
    
    # Calculate average closest distance by object type
    avg_distance_by_type = proximity_df.groupby('obj1_type')['distance'].mean().sort_values()
    print("\nAverage closest neighbor distance by object type:")
    print(avg_distance_by_type)
    
    # Calculate the most common neighbor type for each object type
    type_pairs = proximity_df.groupby(['obj1_type', 'obj2_type']).size().reset_index(name='count')
    most_common_neighbors = type_pairs.sort_values(['obj1_type', 'count'], ascending=[True, False])
    most_common_neighbors = most_common_neighbors.groupby('obj1_type').first().reset_index()
    print("\nMost common neighbor type for each object type:")
    for _, row in most_common_neighbors.iterrows():
        print(f"{row['obj1_type']} → {row['obj2_type']} ({row['count']} instances)")
    
    # Visualize proximity relationships
    plt.figure(figsize=(10, 8))
    plt.hist(proximity_df['distance'], bins=20, alpha=0.7)
    plt.axvline(proximity_df['distance'].mean(), color='r', linestyle='--', label=f'Mean: {proximity_df["distance"].mean():.2f}m')
    plt.axvline(proximity_df['distance'].median(), color='g', linestyle='--', label=f'Median: {proximity_df["distance"].median():.2f}m')
    plt.xlabel('Distance to Closest Neighbor (m)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Closest Neighbor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('closest_neighbor_histogram.png', dpi=300)
    
    # Boxplot of distances by object type
    plt.figure(figsize=(12, 8))
    type_order = avg_distance_by_type.index
    boxplot_data = [proximity_df[proximity_df['obj1_type'] == t]['distance'] for t in type_order]
    
    plt.boxplot(boxplot_data, labels=type_order)
    plt.xticks(rotation=45)
    plt.ylabel('Distance to Closest Neighbor (m)')
    plt.title('Closest Neighbor Distance by Object Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('closest_neighbor_by_type.png', dpi=300)
    
    # 2. Lane Detection using DBSCAN Clustering
    print("\n--- Lane Detection Analysis ---")
    
    # Filter for cars as they most likely follow lane patterns
    cars_df = df[df['obj_type'] == 'Car']
    car_positions = cars_df[['pos_x', 'pos_y']].values
    
    # Apply DBSCAN for lane clustering
    # Adjust eps (max distance) and min_samples based on your data
    dbscan = DBSCAN(eps=2.5, min_samples=5)
    car_clusters = dbscan.fit_predict(car_positions)
    
    # Add cluster information to cars dataframe
    cars_df = cars_df.copy()
    cars_df['cluster'] = car_clusters
    
    # Count objects per cluster
    cluster_counts = pd.Series(car_clusters).value_counts().sort_index()
    print("\nCars per cluster:")
    print(cluster_counts)
    
    # Visualize clusters (potential lanes)
    plt.figure(figsize=(14, 10))
    
    # First plot all non-car objects as background
    non_cars = df[df['obj_type'] != 'Car']
    for obj_type in non_cars['obj_type'].unique():
        subset = non_cars[non_cars['obj_type'] == obj_type]
        plt.scatter(subset['pos_x'], subset['pos_y'], marker='x', label=obj_type, alpha=0.4)
    
    # Then plot car clusters
    unique_clusters = np.unique(car_clusters)
    colormap = plt.cm.get_cmap('tab10', len(unique_clusters))
    
    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id == -1:  # Noise points in DBSCAN
            cluster_cars = cars_df[cars_df['cluster'] == cluster_id]
            plt.scatter(cluster_cars['pos_x'], cluster_cars['pos_y'], 
                       color='black', marker='o', label='Noise', alpha=0.5)
        else:
            cluster_cars = cars_df[cars_df['cluster'] == cluster_id]
            plt.scatter(cluster_cars['pos_x'], cluster_cars['pos_y'],
                       color=colormap(i), marker='o', label=f'Lane {cluster_id}', alpha=0.7)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Potential Lane Detection using DBSCAN Clustering')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.savefig('lane_clusters.png', dpi=300)
    
    # Fit lines to significant clusters to represent lanes
    lane_models = {}
    min_cluster_size = 5  # Minimum number of cars to consider a valid lane
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_cars = cars_df[cars_df['cluster'] == cluster_id]
        
        if len(cluster_cars) >= min_cluster_size:
            X = cluster_cars['pos_x'].values.reshape(-1, 1)
            y = cluster_cars['pos_y'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            lane_models[cluster_id] = {
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'cars': len(cluster_cars),
                'r2_score': model.score(X, y)
            }
    
    print("\nLane model parameters:")
    for lane_id, params in lane_models.items():
        print(f"Lane {lane_id}: slope={params['slope']:.4f}, intercept={params['intercept']:.4f}, " +
              f"cars={params['cars']}, R²={params['r2_score']:.4f}")
    
    # Visualize lane lines
    plt.figure(figsize=(14, 10))
    
    # Plot all objects by type
    for obj_type in df['obj_type'].unique():
        subset = df[df['obj_type'] == obj_type]
        plt.scatter(subset['pos_x'], subset['pos_y'], label=obj_type, alpha=0.5)
    
    # Plot lane lines
    x_min, x_max = df['pos_x'].min(), df['pos_x'].max()
    x_range = np.array([x_min, x_max])
    
    for lane_id, params in lane_models.items():
        y_range = params['slope'] * x_range + params['intercept']
        plt.plot(x_range, y_range, 'r-', linewidth=2, 
                label=f"Lane {lane_id}" if lane_id == list(lane_models.keys())[0] else "")
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Detected Lanes with All Objects')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.savefig('detected_lanes_with_objects.png', dpi=300)
    
    # 3. Object Size vs. Position Analysis
    print("\n--- Object Size vs. Position Analysis ---")
    
    # Calculate distance from origin
    df['distance_from_origin'] = np.sqrt(df['pos_x']**2 + df['pos_y']**2)
    
    # Calculate volume
    df['volume'] = df['scale_x'] * df['scale_y'] * df['scale_z']
    
    # Scatter plot of volume vs. distance
    plt.figure(figsize=(12, 8))
    
    for obj_type in df['obj_type'].unique():
        subset = df[df['obj_type'] == obj_type]
        plt.scatter(subset['distance_from_origin'], subset['volume'], label=obj_type, alpha=0.7)
    
    plt.xlabel('Distance from Origin (m)')
    plt.ylabel('Volume (m³)')
    plt.title('Object Volume vs. Distance from Origin')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('volume_vs_distance.png', dpi=300)
    
    # Calculate correlation between distance and volume
    correlation = df.groupby('obj_type').apply(
        lambda x: np.corrcoef(x['distance_from_origin'], x['volume'])[0, 1]
    )
    
    print("\nCorrelation between distance from origin and object volume:")
    print(correlation)
    
    # 4. Object Orientation Analysis
    print("\n--- Object Orientation Analysis ---")
    
    # Visualize orientation patterns by object type
    plt.figure(figsize=(15, 10))
    
    for i, obj_type in enumerate(df['obj_type'].unique()):
        ax = plt.subplot(2, 4, i+1, projection='polar')
        subset = df[df['obj_type'] == obj_type]
        
        # Adjust angles for polar plot (0 at right, going counterclockwise)
        angles = subset['rot_z'].values
        
        # Create histogram
        bins = np.linspace(-np.pi, np.pi, 16)
        ax.hist(angles, bins=bins)
        ax.set_title(obj_type)
        ax.set_theta_zero_location('E')  # 0 degrees at the right
        ax.set_theta_direction(-1)  # clockwise
        ax.set_rticks([])  # No radial ticks
    
    plt.tight_layout()
    plt.savefig('orientation_by_type_polar.png', dpi=300)
    
    # Calculate predominant orientation for each object type
    predominant_orientation = {}
    for obj_type in df['obj_type'].unique():
        subset = df[df['obj_type'] == obj_type]
        angles = subset['rot_z'].values
        
        # Create histogram
        hist, bin_edges = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find the bin with maximum count
        max_bin = np.argmax(hist)
        predominant_angle = bin_centers[max_bin]
        
        predominant_orientation[obj_type] = {
            'angle_rad': predominant_angle,
            'angle_deg': np.degrees(predominant_angle),
            'count': hist[max_bin],
            'percentage': hist[max_bin] / len(subset) * 100
        }
    
    print("\nPredominant orientation by object type:")
    for obj_type, data in predominant_orientation.items():
        print(f"{obj_type}: {data['angle_deg']:.1f}° ({data['percentage']:.1f}% of objects)")
    
    return proximity_df, lane_models, predominant_orientation

# Load the data
label_directory = "/home/shuttle_01/Pictures/Lidar-Tiand/TIAND_LiDAR-object detection/2nd_september/Scene4/label"
calib_directory = "/home/shuttle_01/Pictures/Lidar-Tiand/TIAND_LiDAR-object detection/2nd_september/Scene4/calib/camera"

all_objects, frame_ids = load_label_files(label_directory)
df = pd.json_normalize(all_objects)

# Extract position, scale, and rotation for easier access
df['pos_x'] = df.apply(lambda row: row['psr.position.x'], axis=1)
df['pos_y'] = df.apply(lambda row: row['psr.position.y'], axis=1)
df['pos_z'] = df.apply(lambda row: row['psr.position.z'], axis=1)
df['scale_x'] = df.apply(lambda row: row['psr.scale.x'], axis=1)
df['scale_y'] = df.apply(lambda row: row['psr.scale.y'], axis=1) 
df['scale_z'] = df.apply(lambda row: row['psr.scale.z'], axis=1)
df['rot_z'] = df.apply(lambda row: row['psr.rotation.z'], axis=1)

# Run the advanced analysis
proximity_results, lane_results, orientation_results = advanced_analysis(df)

print("\nAdvanced analysis complete! All visualizations have been saved.")