import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Paths to KITTI dataset (adjust these to your local paths)
sequence = "00"
dataset_path = "00/image_2"  # Path to image_2 folder
pose_file = "00.txt"  # Path to ground truth poses

# Camera intrinsic parameters (from P2 in KITTI calibration)
K = np.array([[718.856, 0.0, 607.1928],
              [0.0, 718.856, 185.2157],
              [0.0, 0.0, 1.0]])

# Read ground truth poses from pose_file
def read_ground_truth_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose
            poses.append(pose_4x4)
    return poses

# Get list of image files sorted by name
def get_image_files(dataset_path):
    image_files = sorted(Path(dataset_path).glob('*.png'))
    return image_files

# Check if features are well distributed across the image
def check_feature_distribution(keypoints, img_shape):
    """Check if features are well distributed across the image"""
    if len(keypoints) < 50:
        return False
    
    h, w = img_shape[:2]
    points = np.array([kp.pt for kp in keypoints])
    
    # Divide image into 3x3 grid and check coverage
    grid_h, grid_w = h // 3, w // 3
    grid_coverage = np.zeros((3, 3), dtype=bool)
    
    for point in points:
        x, y = int(point[0]), int(point[1])
        grid_x = min(x // grid_w, 2)
        grid_y = min(y // grid_h, 2)
        grid_coverage[grid_y, grid_x] = True
    
    # Require at least 6 out of 9 grid cells to have features
    return np.sum(grid_coverage) >= 6

# Extract and match features using Shi-Tomasi and optical flow
def extract_and_match_features(img1, img2):
    if img1 is None or img2 is None:
        print("Error: One or both input images is None")
        return None, None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Improved feature detection parameters
    feature_params = dict(maxCorners=3000,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)
    
    keypoints1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if keypoints1 is None or len(keypoints1) < 50:
        print(f"Not enough corners detected in first image: {len(keypoints1) if keypoints1 is not None else 0}")
        return None, None, None

    # Improved optical flow parameters
    lk_params = dict(winSize=(21,21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    p1 = keypoints1.astype(np.float32)
    p2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p1, None, **lk_params)

    # Filter out bad matches
    good_old = p1[status==1]
    good_new = p2[status==1]
    errors = err[status==1]
    
    # Additional filtering based on tracking error
    error_threshold = 12.0
    good_mask = errors.flatten() < error_threshold
    good_old = good_old[good_mask]
    good_new = good_new[good_mask]
    
    if len(good_new) < 50:
        print(f"Not enough good tracked points: {len(good_new)}")
        return None, None, None

    keypoints1 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_old]
    keypoints2 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_new]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(good_new))]
    print(f"Corners detected: {len(keypoints1)}, Tracked points: {len(keypoints2)}")
    return keypoints1, keypoints2, matches

# Properly compose relative pose with global pose
def update_global_pose(global_R, global_t, R_rel, t_rel):
    """Properly compose relative pose with global pose"""
    # Transform relative translation to global coordinates
    t_global = global_R @ t_rel
    
    # Update global pose
    new_global_t = global_t + t_global
    new_global_R = global_R @ R_rel
    
    return new_global_R, new_global_t

# Improved pose estimation with better validation
def get_pose_improved(keypoints1, keypoints2, matches, K):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    focal_length = K[0,0]
    principal_point = (K[0,2], K[1,2])
    
    # Multiple essential matrix estimation attempts
    best_R, best_t, best_inliers = None, None, 0
    
    for threshold in [0.5, 1.0, 1.5]:  # Try different thresholds
        essential_matrix, mask = cv2.findEssentialMat(
            points1, points2,
            focal=focal_length,
            pp=principal_point,
            method=cv2.RANSAC,
            prob=0.9999,
            threshold=threshold,
            maxIters=5000
        )
        
        if essential_matrix is None:
            continue
            
        # Filter points using RANSAC mask
        if mask is not None:
            inlier_points1 = points1[mask.ravel() == 1]
            inlier_points2 = points2[mask.ravel() == 1]
        else:
            inlier_points1, inlier_points2 = points1, points2
        
        if len(inlier_points1) < 10:
            continue
        
        # Recover pose
        inliers, R, t, mask_pose = cv2.recoverPose(
            essential_matrix, inlier_points1, inlier_points2, K
        )
        
        # Keep best result based on inliers
        if inliers > best_inliers:
            best_R, best_t, best_inliers = R, t, inliers
    
    if best_R is None:
        return np.eye(3), np.zeros((3,1)), 0
    
    # Additional validation: check if translation is reasonable
    t_norm = np.linalg.norm(best_t)
    if t_norm < 0.01 or t_norm > 10.0:  # Unreasonable translation
        return np.eye(3), np.zeros((3,1)), 0
    
    confidence = best_inliers / len(matches) if len(matches) > 0 else 0
    return best_R, best_t, confidence

# Process a single frame with improved pose estimation
def process_frame_improved(prev_frame, curr_frame, global_R, global_t, K):
    """Process a single frame with improved pose estimation"""
    
    # Extract and match features
    kp1, kp2, matches = extract_and_match_features(prev_frame, curr_frame)
    
    if kp1 is None or matches is None:
        return global_R, global_t, 0.0
    
    # Check feature distribution
    if not check_feature_distribution(kp1, prev_frame.shape):
        print("Poor feature distribution, skipping frame")
        return global_R, global_t, 0.0
    
    # Estimate relative pose
    R_rel, t_rel, confidence = get_pose_improved(kp1, kp2, matches, K)
    
    # Additional motion validation
    if confidence < 0.3:
        return global_R, global_t, 0.0
    
    # Check for reasonable motion (avoid jumps)
    translation_magnitude = np.linalg.norm(t_rel)
    if translation_magnitude > 5.0:  # Unreasonably large motion
        print(f"Large motion detected ({translation_magnitude:.2f}), skipping")
        return global_R, global_t, 0.0
    
    # Update global pose with proper composition
    new_global_R, new_global_t = update_global_pose(global_R, global_t, R_rel, t_rel)
    
    return new_global_R, new_global_t, confidence

# Calculate scale correction factor
def calculate_scale_correction(vo_trajectory, gt_poses, window_size=10):
    """Calculate scale correction using recent trajectory segments"""
    if len(vo_trajectory) < 2 or len(gt_poses) < len(vo_trajectory):
        return 1.0
    
    # Use recent trajectory segment for scale estimation
    start_idx = max(0, len(vo_trajectory) - window_size)
    
    # Calculate VO displacement
    vo_start = np.array(vo_trajectory[start_idx])
    vo_end = np.array(vo_trajectory[-1])
    vo_displacement = np.linalg.norm(vo_end - vo_start)
    
    # Calculate GT displacement
    gt_start = gt_poses[start_idx][:3, 3]
    gt_end = gt_poses[len(vo_trajectory)-1][:3, 3]
    gt_displacement = np.linalg.norm(gt_end - gt_start)
    
    if vo_displacement < 0.1:  # Avoid division by very small numbers
        return 1.0
    
    scale = gt_displacement / vo_displacement
    
    # Limit scale changes to reasonable range
    scale = np.clip(scale, 0.1, 10.0)
    
    return scale

# Main visualization function
def visualize_odometry():
    # Load image files and ground truth poses
    image_files = get_image_files(dataset_path)
    if not image_files:
        print("Error: No images found in dataset path")
        return
        
    ground_truth_poses = read_ground_truth_poses(pose_file)
    if not ground_truth_poses:
        print("Error: No ground truth poses found")
        return

    # Initialize global pose for visual odometry
    global_R = np.eye(3)
    global_t = np.zeros((3,1))
    trajectory = [global_t.flatten()]  # List of 3D translation vectors
    trajectory_scaled = [global_t.flatten()]  # Scale-corrected trajectory
    
    # Scale correction parameters
    scale_factor = 1.0
    scale_update_interval = 5  # Update scale every N frames

    # Extract x, y coordinates from ground truth poses (KITTI: X=forward, Y=left, Z=up)
    gt_translations = np.array([pose[:3, 3] for pose in ground_truth_poses])
    gt_x, gt_y = gt_translations[:, 0], gt_translations[:, 1]  # Changed from Z to Y

    # Set up Matplotlib for 2D plotting (x-y plane for top-down view)
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Left)')
    ax.set_title('Visual Odometry vs Ground Truth (Top-Down View)')
    ax.grid(True)
    ax.set_aspect('equal')  # Equal aspect ratio for proper visualization
    
    # Initialize plots
    vo_line, = ax.plot([], [], 'b-', linewidth=2, label='Visual Odometry', alpha=0.7)
    vo_scaled_line, = ax.plot([], [], 'c-', linewidth=2, label='Scale-Corrected VO')
    vo_point, = ax.plot([], [], 'ro', markersize=8, label='VO Current Position')
    gt_line, = ax.plot([], [], 'g-', linewidth=2, label='Ground Truth')
    gt_point, = ax.plot([], [], 'mo', markersize=8, label='GT Current Position')
    ax.legend()

    # OpenCV window for images
    cv2.namedWindow('KITTI Sequence', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('KITTI Sequence', 1024, 384)

    # Read first frame
    prev_frame = cv2.imread(str(image_files[0]))
    if prev_frame is None:
        print(f"Error: Could not read first image {image_files[0]}")
        return

    print(f"Processing {len(image_files)} frames...")
    
    # Process frames
    n_frames = min(len(image_files), len(ground_truth_poses))
    for i in range(1, n_frames):
        # Read current frame
        curr_frame = cv2.imread(str(image_files[i]))
        if curr_frame is None:
            print(f"Failed to load image {image_files[i]}")
            continue

        # Process frame with improved method
        new_global_R, new_global_t, confidence = process_frame_improved(
            prev_frame, curr_frame, global_R, global_t, K
        )
        
        # Update global pose if processing was successful
        if confidence > 0:
            global_R = new_global_R
            global_t = new_global_t
            trajectory.append(global_t.flatten())
            
            # Update scale factor periodically
            if i % scale_update_interval == 0:
                scale_factor = calculate_scale_correction(trajectory, ground_truth_poses)
                print(f"Frame {i}: Scale factor updated to {scale_factor:.3f}, Confidence: {confidence:.3f}")
            
            # Apply scale correction
            scaled_position = trajectory[-1] * scale_factor
            trajectory_scaled.append(scaled_position)
        else:
            print(f"Frame {i}: Processing failed, keeping previous position")
            # Keep previous position
            trajectory.append(trajectory[-1])
            trajectory_scaled.append(trajectory_scaled[-1])

        # Update trajectory plots every 5 frames for better performance
        if i % 5 == 0 or i == n_frames - 1:
            traj_array = np.array(trajectory)
            traj_scaled_array = np.array(trajectory_scaled)
            
            # Original and scaled VO trajectories (X-Y plane)
            vo_x, vo_y = traj_array[:, 0], traj_array[:, 1]  # Changed from Z to Y
            vo_x_scaled, vo_y_scaled = traj_scaled_array[:, 0], traj_scaled_array[:, 1]
            vo_line.set_data(vo_x, vo_y)
            vo_scaled_line.set_data(vo_x_scaled, vo_y_scaled)
            vo_point.set_data([vo_x_scaled[-1]], [vo_y_scaled[-1]])

            # Ground truth trajectory
            gt_x_curr, gt_y_curr = gt_x[:i+1], gt_y[:i+1]
            gt_line.set_data(gt_x_curr, gt_y_curr)
            gt_point.set_data([gt_x_curr[-1]], [gt_y_curr[-1]])

            # Update axis limits with padding
            x_all = np.concatenate([vo_x, vo_x_scaled, gt_x[:i+1]])
            y_all = np.concatenate([vo_y, vo_y_scaled, gt_y[:i+1]])
            x_range = max(x_all) - min(x_all)
            y_range = max(y_all) - min(y_all)
            padding_x = 0.1 * x_range if x_range > 0 else 1.0
            padding_y = 0.1 * y_range if y_range > 0 else 1.0
            ax.set_xlim(min(x_all) - padding_x, max(x_all) + padding_x)
            ax.set_ylim(min(y_all) - padding_y, max(y_all) + padding_y)

            # Redraw plot
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Display current image with detailed info
        info_text = f"Frame: {i}/{n_frames-1} | Scale: {scale_factor:.3f} | Conf: {confidence:.3f}"
        cv2.putText(curr_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('KITTI Sequence', curr_frame)

        # Control frame rate
        time.sleep(0.05)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early exit requested")
            break

        prev_frame = curr_frame.copy()

    # Final statistics
    if len(trajectory) > 1 and len(trajectory_scaled) > 1:
        final_vo_pos = np.array(trajectory[-1])
        final_scaled_pos = np.array(trajectory_scaled[-1])
        final_gt_pos = ground_truth_poses[len(trajectory)-1][:3, 3]
        
        original_error = np.linalg.norm(final_vo_pos - final_gt_pos)
        scaled_error = np.linalg.norm(final_scaled_pos - final_gt_pos)
        
        print(f"\n=== Final Results ===")
        print(f"Processed {len(trajectory)} frames")
        print(f"Original VO final position: [{final_vo_pos[0]:.2f}, {final_vo_pos[1]:.2f}, {final_vo_pos[2]:.2f}]")
        print(f"Scaled VO final position: [{final_scaled_pos[0]:.2f}, {final_scaled_pos[1]:.2f}, {final_scaled_pos[2]:.2f}]")
        print(f"Ground truth final position: [{final_gt_pos[0]:.2f}, {final_gt_pos[1]:.2f}, {final_gt_pos[2]:.2f}]")
        print(f"Original VO error: {original_error:.3f} meters")
        print(f"Scale-corrected VO error: {scaled_error:.3f} meters")
        if original_error > 0:
            improvement = ((original_error - scaled_error) / original_error * 100)
            print(f"Improvement: {improvement:.1f}%")
        
        # Calculate final trajectory angle
        if len(trajectory_scaled) > 10:
            start_pos = np.array(trajectory_scaled[0])
            end_pos = np.array(trajectory_scaled[-1])
            trajectory_vector = end_pos - start_pos
            trajectory_angle = np.arctan2(trajectory_vector[1], trajectory_vector[0]) * 180 / np.pi
            
            gt_start = ground_truth_poses[0][:3, 3]
            gt_end = ground_truth_poses[len(trajectory_scaled)-1][:3, 3]
            gt_vector = gt_end - gt_start
            gt_angle = np.arctan2(gt_vector[1], gt_vector[0]) * 180 / np.pi
            
            print(f"VO trajectory angle: {trajectory_angle:.1f}°")
            print(f"GT trajectory angle: {gt_angle:.1f}°")
            print(f"Angle error: {abs(trajectory_angle - gt_angle):.1f}°")

    print("\nVisualization complete. Close the plot window to exit.")
    
    # Keep plot open
    plt.ioff()
    plt.show()
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        visualize_odometry()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        cv2.destroyAllWindows()
        plt.close('all')
    except Exception as e:
        print(f"Error occurred: {e}")
        cv2.destroyAllWindows()
        plt.close('all')