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

# Extract and match features using Shi-Tomasi and optical flow
def extract_and_match_features(img1, img2):
    if img1 is None or img2 is None:
        print("Error: One or both input images is None")
        return None, None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Improved feature detection parameters
    feature_params = dict(maxCorners=3000,  # Increased for better tracking
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)
    
    keypoints1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if keypoints1 is None or len(keypoints1) < 50:  # Increased minimum threshold
        print(f"Not enough corners detected in first image: {len(keypoints1) if keypoints1 is not None else 0}")
        return None, None, None

    # Improved optical flow parameters
    lk_params = dict(winSize=(21,21),  # Increased window size
                     maxLevel=3,       # Increased pyramid levels
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
    
    if len(good_new) < 50:  # Increased minimum threshold
        print(f"Not enough good tracked points: {len(good_new)}")
        return None, None, None

    keypoints1 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_old]
    keypoints2 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_new]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(good_new))]
    print(f"Corners detected: {len(keypoints1)}, Tracked points: {len(keypoints2)}")
    return keypoints1, keypoints2, matches

# Estimate pose between two frames with improved robustness
def get_pose(keypoints1, keypoints2, matches, K):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    focal_length = K[0,0]
    principal_point = (K[0,2], K[1,2])
    
    # Improved essential matrix estimation
    essential_matrix, mask = cv2.findEssentialMat(
        points1, points2,
        focal=focal_length,
        pp=principal_point,
        method=cv2.RANSAC,
        prob=0.9999,      # Higher confidence
        threshold=0.5,    # Lower threshold for better precision
        maxIters=5000     # More iterations
    )
    
    if essential_matrix is None:
        print("Failed to compute essential matrix")
        return np.eye(3), np.zeros((3,1)), 0
    
    # Filter points using RANSAC mask
    if mask is not None:
        points1 = points1[mask.ravel() == 1]
        points2 = points2[mask.ravel() == 1]
    
    if len(points1) < 10:
        print("Not enough inlier points for pose recovery")
        return np.eye(3), np.zeros((3,1)), 0
    
    _, R, t, mask_pose = cv2.recoverPose(essential_matrix, points1, points2, K)
    
    # Calculate confidence based on number of inliers
    confidence = len(points1) / len(matches) if len(matches) > 0 else 0
    
    return R, t, confidence

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

    # Extract x, z coordinates from ground truth poses
    gt_translations = np.array([pose[:3, 3] for pose in ground_truth_poses])
    gt_x, gt_z = gt_translations[:, 0], gt_translations[:, 2]

    # Set up Matplotlib for 2D plotting (x-z plane)
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Visual Odometry vs Ground Truth (X-Z Plane)')
    ax.grid(True)
    
    # Initialize plots
    vo_line, = ax.plot([], [], 'b-', label='Visual Odometry')
    vo_point, = ax.plot([], [], 'ro', label='VO Current Position')
    gt_line, = ax.plot([], [], 'g-', label='Ground Truth')
    gt_point, = ax.plot([], [], 'mo', label='GT Current Position')
    ax.legend()

    # OpenCV window for images
    cv2.namedWindow('KITTI Sequence', cv2.WINDOW_NORMAL)

    # Read first frame
    prev_frame = cv2.imread(str(image_files[0]))
    if prev_frame is None:
        print(f"Error: Could not read first image {image_files[0]}")
        return

    # Process frames
    n_frames = min(len(image_files), len(ground_truth_poses))
    for i in range(1, n_frames):
        # Read current frame
        curr_frame = cv2.imread(str(image_files[i]))
        if curr_frame is None:
            print(f"Failed to load image {image_files[i]}")
            continue

        # Extract and match features
        kp1, kp2, matches = extract_and_match_features(prev_frame, curr_frame)
        if kp1 is not None and matches is not None:
            # Estimate relative pose
            R, t, confidence = get_pose(kp1, kp2, matches, K)
            
            # Only update pose if confidence is reasonable
            if confidence > 0.3:  # Minimum confidence threshold
                # Update global pose
                global_t = global_t + global_R @ t
                global_R = global_R @ R
                trajectory.append(global_t.flatten())
            else:
                print(f"Frame {i}: Low confidence ({confidence:.3f}), skipping pose update")
                # Keep previous position
                trajectory.append(trajectory[-1])

        # Update trajectory plots
        traj_array = np.array(trajectory)
        
        # Original and scaled VO trajectories
        vo_x, vo_z = traj_array[:, 0], traj_array[:, 2]
        vo_line.set_data(vo_x, vo_z)
        vo_point.set_data([vo_x[-1]], [vo_z[-1]])

        # Ground truth trajectory
        gt_x_curr, gt_z_curr = gt_x[:i+1], gt_z[:i+1]
        gt_line.set_data(gt_x_curr, gt_z_curr)
        gt_point.set_data([gt_x_curr[-1]], [gt_z_curr[-1]])

        # Update axis limits with padding
        x_all = np.concatenate([vo_x, gt_x[:i+1]])
        z_all = np.concatenate([vo_z, gt_z[:i+1]])
        x_range = max(x_all) - min(x_all)
        z_range = max(z_all) - min(z_all)
        padding_x = 0.1 * x_range if x_range > 0 else 1.0
        padding_z = 0.1 * z_range if z_range > 0 else 1.0
        ax.set_xlim(min(x_all) - padding_x, max(x_all) + padding_x)
        ax.set_ylim(min(z_all) - padding_z, max(z_all) + padding_z)

        # Redraw plot
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Display current image with frame info
        info_text = f"Frame: {i}/{n_frames-1}"
        cv2.putText(curr_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('KITTI Sequence', curr_frame)

        # Control frame rate (~10 FPS)
        time.sleep(0.1)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = curr_frame.copy()

    # Final statistics
    if len(trajectory) > 1:
        final_vo_pos = np.array(trajectory[-1])
        final_gt_pos = ground_truth_poses[len(trajectory)-1][:3, 3]
        
        original_error = np.linalg.norm(final_vo_pos - final_gt_pos)
        
        print(f"\nFinal Results:")
        print(f"Original VO error: {original_error:.3f} meters")

    # Keep plot open
    plt.ioff()
    plt.show()
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_odometry()
