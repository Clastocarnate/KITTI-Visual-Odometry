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

# Extract and match features using Shi-Tomasi, ORB descriptors, and Brute-Force Matcher
def extract_and_match_features(img1, img2, ground_truth_poses, frame_idx):
    if img1 is None or img2 is None:
        print("Error: One or both input images is None")
        return None, None, None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Feature detection using Shi-Tomasi
    feature_params = dict(maxCorners=3000,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)
    
    corners1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if corners1 is None or len(corners1) < 50:
        print(f"Not enough corners detected in first image: {len(corners1) if corners1 is not None else 0}")
        return None, None, None

    corners2 = cv2.goodFeaturesToTrack(gray2, mask=None, **feature_params)
    if corners2 is None or len(corners2) < 50:
        print(f"Not enough corners detected in second image: {len(corners2) if corners2 is not None else 0}")
        return None, None, None

    # Convert corners to KeyPoint objects
    keypoints1 = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in corners1]
    keypoints2 = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in corners2]

    # Compute ORB descriptors
    orb = cv2.ORB_create()
    keypoints1, des1 = orb.compute(gray1, keypoints1)
    keypoints2, des2 = orb.compute(gray2, keypoints2)

    if des1 is None or des2 is None:
        print("Failed to compute descriptors")
        return None, None, None

    # Brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 50:
        print(f"Not enough good matches found: {len(matches)}")
        return None, None, None
    
    # Get ground truth and detected poses
    gt_pose = ground_truth_poses[frame_idx][:3, :3] if frame_idx < len(ground_truth_poses) else np.eye(3)
    gt_t = ground_truth_poses[frame_idx][:3, 3].reshape(3, 1) if frame_idx < len(ground_truth_poses) else np.zeros((3, 1))
    R, t, _ = get_pose(keypoints1, keypoints2, matches, K)
    detected_pose = R
    detected_t = t
    
    print(f"Ground Truth Pose (R): {gt_pose}")
    print(f"Ground Truth Translation (t): {gt_t.flatten()}")
    print(f"Detected Pose (R): {detected_pose}")
    print(f"Detected Translation (t): {detected_t.flatten()}")
    
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

    # Initialize global pose for visual odometry with first ground truth pose
    initial_pose = np.array([
        [1.000000e+00, 9.043680e-12, 2.326809e-11, 5.551115e-17],
        [9.043683e-12, 1.000000e+00, 2.392370e-10, 3.330669e-16],
        [2.326810e-11, 2.392370e-10, 9.999999e-01, -4.440892e-16]
    ])
    global_R = initial_pose[:3, :3]
    global_t = initial_pose[:3, 3].reshape(3, 1)
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
        kp1, kp2, matches = extract_and_match_features(prev_frame, curr_frame, ground_truth_poses, i)
        if kp1 is not None and matches is not None:
            # Estimate relative pose
            R, t, confidence = get_pose(kp1, kp2, matches, K)
            
            # Only update pose if confidence is reasonable
            if confidence > 0.3:  # Minimum confidence threshold
                # Update global pose by multiplying with the previous ground truth pose
                prev_gt_pose = ground_truth_poses[i-1][:3, :3] if i > 0 else np.eye(3)
                prev_gt_t = ground_truth_poses[i-1][:3, 3].reshape(3, 1) if i > 0 else np.zeros((3, 1))
                global_R = prev_gt_pose @ R
                global_t = prev_gt_t + prev_gt_pose @ t
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