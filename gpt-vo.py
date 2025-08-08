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

# ----- Helper functions -----

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

def get_image_files(dataset_path):
    return sorted(Path(dataset_path).glob('*.png'))

def extract_and_match_features(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=3000,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)
    
    keypoints1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if keypoints1 is None or len(keypoints1) < 50:
        return None, None, None

    lk_params = dict(winSize=(21,21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    p1 = keypoints1.astype(np.float32)
    p2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p1, None, **lk_params)

    good_old = p1[status==1]
    good_new = p2[status==1]
    errors = err[status==1]

    error_threshold = 12.0
    good_mask = errors.flatten() < error_threshold
    good_old = good_old[good_mask]
    good_new = good_new[good_mask]

    if len(good_new) < 50:
        return None, None, None

    keypoints1 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_old]
    keypoints2 = [cv2.KeyPoint(x[0], x[1], 10) for x in good_new]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(good_new))]
    return keypoints1, keypoints2, matches

def get_pose(keypoints1, keypoints2, matches, K):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    focal_length = K[0,0]
    principal_point = (K[0,2], K[1,2])
    
    essential_matrix, mask = cv2.findEssentialMat(
        points1, points2,
        focal=focal_length,
        pp=principal_point,
        method=cv2.RANSAC,
        prob=0.9999,
        threshold=0.5,
        maxIters=5000
    )
    
    if essential_matrix is None:
        return np.eye(3), np.zeros((3,1)), 0
    
    if mask is not None:
        points1 = points1[mask.ravel() == 1]
        points2 = points2[mask.ravel() == 1]
    
    if len(points1) < 10:
        return np.eye(3), np.zeros((3,1)), 0
    
    _, R, t, mask_pose = cv2.recoverPose(essential_matrix, points1, points2, K)
    
    confidence = len(points1) / len(matches) if len(matches) > 0 else 0
    return R, t, confidence

def calculate_scale_correction(vo_trajectory, gt_poses, window_size=10):
    if len(vo_trajectory) < 2 or len(gt_poses) < len(vo_trajectory):
        return 1.0
    
    start_idx = max(0, len(vo_trajectory) - window_size)
    vo_start = np.array(vo_trajectory[start_idx])
    vo_end = np.array(vo_trajectory[-1])
    vo_displacement = np.linalg.norm(vo_end - vo_start)
    
    gt_start = gt_poses[start_idx][:3, 3]
    gt_end = gt_poses[len(vo_trajectory)-1][:3, 3]
    gt_displacement = np.linalg.norm(gt_end - gt_start)
    
    if vo_displacement < 0.1:
        return 1.0
    
    scale = gt_displacement / vo_displacement
    return np.clip(scale, 0.1, 10.0)

def align_vo_to_gt(vo_points, gt_points):
    vo_centroid = np.mean(vo_points, axis=0)
    gt_centroid = np.mean(gt_points, axis=0)
    vo_centered = vo_points - vo_centroid
    gt_centered = gt_points - gt_centroid

    H = vo_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    aligned = (R @ vo_centered.T).T + gt_centroid
    return aligned, R, gt_centroid - R @ vo_centroid

# ----- Main -----
def visualize_odometry():
    image_files = get_image_files(dataset_path)
    ground_truth_poses = read_ground_truth_poses(pose_file)
    if not image_files or not ground_truth_poses:
        print("Error loading data")
        return

    global_R = np.eye(3)
    global_t = np.zeros((3,1))
    trajectory = [global_t.flatten()]
    scale_factor = 1.0
    scale_update_interval = 5

    gt_translations = np.array([pose[:3, 3] for pose in ground_truth_poses])
    gt_x, gt_z = gt_translations[:, 0], gt_translations[:, 2]

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Visual Odometry vs Ground Truth (Aligned)')
    ax.grid(True)
    
    vo_line, = ax.plot([], [], 'b-', label='Visual Odometry')
    vo_point, = ax.plot([], [], 'ro', label='VO Current Position')
    gt_line, = ax.plot([], [], 'g-', label='Ground Truth')
    gt_point, = ax.plot([], [], 'mo', label='GT Current Position')
    ax.legend()

    cv2.namedWindow('KITTI Sequence', cv2.WINDOW_NORMAL)
    prev_frame = cv2.imread(str(image_files[0]))

    # Alignment placeholders
    alignment_done = False
    R_align = np.eye(3)
    t_align = np.zeros(3)

    n_frames = min(len(image_files), len(ground_truth_poses))
    for i in range(1, n_frames):
        curr_frame = cv2.imread(str(image_files[i]))
        kp1, kp2, matches = extract_and_match_features(prev_frame, curr_frame)

        if kp1 is not None and matches is not None:
            R, t, confidence = get_pose(kp1, kp2, matches, K)
            if confidence > 0.3:
                global_t = global_t + global_R @ t
                global_R = global_R @ R
                trajectory.append(global_t.flatten())

                if i % scale_update_interval == 0:
                    scale_factor = calculate_scale_correction(trajectory, ground_truth_poses)
            else:
                trajectory.append(trajectory[-1])

        # Alignment after a few frames
        if not alignment_done and i > 5:
            vo_segment = np.array(trajectory)
            gt_segment = gt_translations[:len(vo_segment)]
            aligned, R_align, t_align = align_vo_to_gt(vo_segment, gt_segment)
            trajectory = aligned.tolist()
            alignment_done = True

        traj_array = np.array(trajectory)
        vo_x, vo_z = traj_array[:, 0], traj_array[:, 2]
        vo_line.set_data(vo_x, vo_z)
        vo_point.set_data([vo_x[-1]], [vo_z[-1]])

        gt_x_curr, gt_z_curr = gt_x[:i+1], gt_z[:i+1]
        gt_line.set_data(gt_x_curr, gt_z_curr)
        gt_point.set_data([gt_x_curr[-1]], [gt_z_curr[-1]])

        ax.set_xlim(min(vo_x.min(), gt_x_curr.min())-5, max(vo_x.max(), gt_x_curr.max())+5)
        ax.set_ylim(min(vo_z.min(), gt_z_curr.min())-5, max(vo_z.max(), gt_z_curr.max())+5)

        fig.canvas.draw()
        fig.canvas.flush_events()

        info_text = f"Frame: {i}/{n_frames-1}, Scale: {scale_factor:.3f}"
        cv2.putText(curr_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('KITTI Sequence', curr_frame)

        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = curr_frame.copy()

    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_odometry()
