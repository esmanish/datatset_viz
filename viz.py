import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

# Function to load all JSON files from a directory
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
    
    print(f"Loaded data from {len(frame_ids)} frames with {len(all_objects)} objects total")
    return all_objects, frame_ids

# Load the calibration file
def load_calibration(calib_file):
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    return calib_data

# Main analysis function
def analyze_dataset(label_dir, calib_dir):
    # Load object data
    all_objects, frame_ids = load_label_files(label_dir)
    
    # Convert to DataFrame for easier analysis
    df = pd.json_normalize(all_objects)
    
    # Basic dataset statistics
    print("\n--- Dataset Statistics ---")
    print(f"Total number of objects: {len(df)}")
    
    # Count by object type
    obj_type_counts = df['obj_type'].value_counts()
    print("\nObject type distribution:")
    print(obj_type_counts)
    
    # Extract position information
    df['pos_x'] = df.apply(lambda row: row['psr.position.x'], axis=1)
    df['pos_y'] = df.apply(lambda row: row['psr.position.y'], axis=1)
    df['pos_z'] = df.apply(lambda row: row['psr.position.z'], axis=1)
    
    # Extract scale information
    df['scale_x'] = df.apply(lambda row: row['psr.scale.x'], axis=1)
    df['scale_y'] = df.apply(lambda row: row['psr.scale.y'], axis=1) 
    df['scale_z'] = df.apply(lambda row: row['psr.scale.z'], axis=1)
    
    # Extract rotation information
    df['rot_z'] = df.apply(lambda row: row['psr.rotation.z'], axis=1)
    
    # Calculate average dimensions by object type
    print("\nAverage dimensions by object type:")
    for obj_type in df['obj_type'].unique():
        subset = df[df['obj_type'] == obj_type]
        avg_dims = {
            'length': subset['scale_x'].mean(),
            'width': subset['scale_y'].mean(),
            'height': subset['scale_z'].mean()
        }
        print(f"{obj_type}: {avg_dims['length']:.2f}m x {avg_dims['width']:.2f}m x {avg_dims['height']:.2f}m")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Bar chart of object types
    plt.figure(figsize=(10, 6))
    obj_type_counts.plot(kind='bar', color='skyblue')
    plt.title('Object Type Distribution')
    plt.xlabel('Object Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('object_type_distribution.png', dpi=300)
    
    # 2. Scatter plot of object positions (top-down view)
    plt.figure(figsize=(12, 10))
    for obj_type in df['obj_type'].unique():
        subset = df[df['obj_type'] == obj_type]
        plt.scatter(subset['pos_x'], subset['pos_y'], label=obj_type, alpha=0.7)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Spatial Distribution of Objects (Top-Down View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('spatial_distribution.png', dpi=300)
    
    # 3. Density heatmap
    plt.figure(figsize=(10, 8))
    plt.hist2d(df['pos_x'], df['pos_y'], bins=20, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Density Heatmap of Objects')
    plt.savefig('density_heatmap.png', dpi=300)
    
    # 4. Create a 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for obj_type in df['obj_type'].unique():
        subset = df[df['obj_type'] == obj_type]
        ax.scatter(subset['pos_x'], subset['pos_y'], subset['pos_z'], label=obj_type, alpha=0.7)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Spatial Distribution of Objects')
    plt.legend()
    plt.savefig('3d_spatial_distribution.png', dpi=300)
    
    # 5. Object orientation analysis
    plt.figure(figsize=(10, 6))
    plt.hist(df['rot_z'], bins=20, range=(-np.pi, np.pi), color='orange')
    plt.xlabel('Z-Rotation (radians)')
    plt.ylabel('Count')
    plt.title('Distribution of Object Orientations')
    plt.savefig('orientation_hist.png', dpi=300)
    
    # 6. Bird's eye view with orientation
    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    
    # Plot just a sample of objects to avoid overcrowding
    sample_size = min(100, len(df))
    sampled_df = df.sample(sample_size)
    
    for _, obj in sampled_df.iterrows():
        x, y = obj['pos_x'], obj['pos_y']
        length, width = obj['scale_x'], obj['scale_y']
        angle = obj['rot_z']
        
        # Calculate bounding box corners
        corners = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2],
            [-length/2, -width/2]  # Close the rectangle
        ])
        
        # Rotation matrix
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Rotate corners
        rotated_corners = np.dot(corners, rot_matrix.T)
        
        # Translate corners to object position
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y
        
        # Plot the bounding box
        ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'k-', alpha=0.5)
        
        # Add a direction arrow for orientation
        arrow_length = min(length, width) * 0.8
        arrow_end = np.array([arrow_length, 0])
        arrow_end = np.dot(arrow_end, rot_matrix.T)
        ax.arrow(x, y, arrow_end[0], arrow_end[1], head_width=0.3, head_length=0.5, fc='red', ec='red')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Bird\'s Eye View with Object Orientations')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('bev_with_orientation.png', dpi=300)
    
    print("\nAnalysis complete! Generated visualizations have been saved.")
    
    return df

# Run the analysis
if __name__ == "__main__":
    label_directory = "/home/shuttle_01/Pictures/Lidar-Tiand/TIAND_LiDAR-object detection/2nd_september/Scene4/label"  # Update this path
    calib_directory = "/home/shuttle_01/Pictures/Lidar-Tiand/TIAND_LiDAR-object detection/2nd_september/Scene4/calib/camera"  # Update this path
    
    df = analyze_dataset(label_directory, calib_directory)