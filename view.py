import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

# Paths to KITTI dataset (adjust these to your local paths)
sequence = "00"
dataset_path = "00/image_2"  # Path to image_2 folder
pose_file = "00.txt"  # Path to 00.txt

# Read pose data from 00.txt
def read_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            # Each line has 12 values (3x4 matrix flattened)
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            # Convert to 4x4 homogeneous transformation matrix
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose
            poses.append(pose_4x4)
    return poses

# Get list of image files sorted by name
def get_image_files(dataset_path):
    image_files = sorted(Path(dataset_path).glob('*.png'))
    return image_files

# Extract translation (x, z) from pose matrix
def get_translation_xz(pose):
    translation = pose[:3, 3]
    return translation[0], translation[2]  # Return x, z

# Main visualization function
def visualize_sequence():
    # Load data
    poses = read_poses(pose_file)
    image_files = get_image_files(dataset_path)
    
    # Ensure we have matching number of poses and images
    n_frames = min(len(poses), len(image_files))
    print(f"Processing {n_frames} frames")

    # Set up Matplotlib for 2D plotting
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Camera Trajectory (X-Z Plane)')
    ax.grid(True)

    # Initialize trajectory plot
    translations = np.array([get_translation_xz(pose) for pose in poses])
    x, z = translations.T
    line, = ax.plot([], [], 'b-', label='Trajectory')
    point, = ax.plot([], [], 'ro', label='Current Position')
    ax.legend()

    # Set axis limits with padding (10% of range)
    x_range = max(x) - min(x)
    z_range = max(z) - min(z)
    padding_x = 0.1 * x_range
    padding_z = 0.1 * z_range
    ax.set_xlim(min(x) - padding_x, max(x) + padding_x)
    ax.set_ylim(min(z) - padding_z, max(z) + padding_z)  # Fixed y-axis limits

    # OpenCV window for images
    cv2.namedWindow('KITTI Sequence', cv2.WINDOW_NORMAL)

    # Iterate through frames
    for i in range(n_frames):
        # Read and display image
        img = cv2.imread(str(image_files[i]))
        if img is None:
            print(f"Failed to load image {image_files[i]}")
            continue
        cv2.imshow('KITTI Sequence', img)

        # Update trajectory plot
        curr_translations = translations[:i+1]
        if len(curr_translations) > 0:
            line.set_data(curr_translations[:, 0], curr_translations[:, 1])
            point.set_data([curr_translations[-1, 0]], [curr_translations[-1, 1]])

        # Redraw plot
        fig.canvas.draw()
        fig.canvas.flush_events()

      

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    visualize_sequence()