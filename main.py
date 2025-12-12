import cv2
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Dict
import os
import sys
import fundamental_matrix as fm     #this imports our extension 
from evaluate_reconstruction import evaluate_sfm_reconstruction





def choose_dataset(name):
    if name == "Ahmeds_Room":
    
        K_0 = np.array([[3157.17, 0.0, 1708.7 ],
                        [0.0, 3158.83, 1174.85],
                        [0.0,0.0, 1.0]])
        path = "Ahmeds_Room/out_keyframes_00"
        start, end = 3, 19
        return K_0, path, start, end
    elif name == "Playroom_lowres":
        K_0 = np.array([[753.15438141,   0.        , 378.14104591],
                        [  0.        , 753.07835139, 499.71795709],
                        [  0.        ,   0.        ,   1.        ]])
        path = "playroom_lowres/playroom_lowres/playroom_es143_small_000"
        start, end = 2, 15
        return K_0, path, start, end

    elif name == "SEC_2":
        K_0 = np.array([[1734.05, 0.0, 506.82],
                        [0.0, 1744.92, 1091.69],
                        [0.0, 0.0,1.0  ]])
        path = "SEC2/out_keyframes_00"
        start, end = 4, 25
        return K_0, path, start, end
    elif name == "SEC_3":
        K_0 = np.array([[1624.8, 0.0,527.83],
                        [0.0, 1652.42, 1128.6],
                        [0.0, 0.0, 1.0]])
        path = "SEC3/out_keyframes_00"
        start, end = 3, 15
        return K_0, path, start, end
    elif name == "castle":
        K_0 = np.array([[2.76103666e+03, 0.00000000e+00, 1.51928594e+03],
        [0.00000000e+00, 2.76103666e+03, 1.00922694e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        path = "castle/00"
        start, end = 2, 18
        return K_0, path, start, end
    else:
        raise ValueError(f"Unknown dataset name: {name}")

# Camera Intrinsic Matrix (Given)
# K_0 = np.array([[753.15438141,   0.        , 378.14104591],
#                 [  0.        , 753.07835139, 499.71795709],
#                 [  0.        ,   0.        ,   1.        ]])

#This is the camera for SEC_2!!
#Distortion coefficients: [[-0.07606  0.17155  0.03197 -0.00344 -1.20333]]
# K_0 = np.array([[1734.05, 0.0, 506.82],
# [0.0, 1744.92, 1091.69],
# [0.0, 0.0,1.0  ]])

#This is the camera for SEC_3!!
#Distortion coefficients: [[ 0.07459 -0.50705  0.02768 -0.00453  0.68038]]
#Intrinsic camera matrix:
# K_0 = np.array([[1624.8, 0.0,527.83],
# [0.0, 1652.42, 1128.6],
# [0.0, 0.0, 1.0]])


#This is the camera for Ahmed's Video!!
#RMSE of reprojected points: 1.6462128618338445
#Distortion coefficients: [[ 0.15536 -0.32381  0.01389 -0.03284  0.91952]]
#Intrinsic camera matrix:
# K_0 = np.array([[3157.17, 0.0, 1708.7 ],
# [0.0, 3158.83, 1174.85],
# [0.0,0.0, 1.0]])


def load_data(path, start, end):
    """Load the 5 images from the images folder"""

    images = []
    for i in range(start, end):
        full_path = f"{path}{i:02}.jpg"                    #f"playroom_lowres/playroom_lowres/playroom_es143_small_000{i+1:02}.JPG"     
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {full_path}")
        images.append(img)
    return images


def detect_and_match_keypoints(img1, img2):
    """
    Detect keypoints using SIFT and match them robustly using RANSAC
    Returns: matched keypoints in both images
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Use FLANN-based matcher for better performance
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test for robust matching
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract matched keypoint coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, good_matches, kp1, kp2


def estimate_essential_matrix(pts1, pts2, K):
    """
    Estimate Essential Matrix using RANSAC
    Returns: Essential matrix and inlier mask
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask


def extension_estimate_essential_matrix(pts1, pts2, K):
    '''
    This is the implementation of our own RANSAC + 8-point algorithm to estimate the Essential Matrix
    It uses functions from fundamental_matrix.py, so see exact implementation there.
    Returns: Essential matrix and inlier mask
    '''

    F, mask = fm.estimate_fundamental_matrix(pts1, pts2, ransac_thresh=1.0, confidence=0.99)

    E = K.T @ F @ K

    U,D,Vt = np.linalg.svd(E)

    #normalize the singular values
    S = [1, 1, 0]
    E = U @ np.diag(S) @ Vt
    
    return E, mask


def extract_pose(E, pts1, pts2, K):
    """
    Extract camera pose (R, t) from Essential Matrix
    Returns: Rotation matrix R and translation vector t
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask


def triangulate_points(pts1, pts2, P1, P2):
    """
    Triangulate 3D points from two views
    P1, P2: Projection matrices for camera 1 and 2 (3x4)
    Returns: 3D points in homogeneous coordinates
    """
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3, :] / points_4d[3, :]

    return points_3d.T


def filter_points_in_front(points_3d, R, t, pts1, mask, max_distance=50.0):
    """
    Filter points that are in front of both cameras and meet quality criteria

    Args:
        points_3d: 3D points to filter
        R, t: Relative camera pose
        pts1: 2D points from first view
        mask: RANSAC inlier mask
        max_distance: Maximum distance from first camera
    """
    # Check points in front of first camera (identity pose)
    z1 = points_3d[:, 2]

    # Transform points to second camera coordinate system
    points_cam2 = (R @ points_3d.T).T + t.T
    z2 = points_cam2[:, 2]

    # Calculate distance from first camera (at origin)
    distances = np.linalg.norm(points_3d, axis=1)

    # Keep points that satisfy all criteria:
    # 1. In front of both cameras
    # 2. RANSAC inliers
    # 3. Not too far away (likely triangulation errors)
    valid_mask = (
        (z1 > 0) &
        (z2 > 0) &
        (mask.ravel() == 1) &
        (distances < max_distance) &
        (np.isfinite(points_3d).all(axis=1))  # No NaN or Inf
    )

    return points_3d[valid_mask], pts1[valid_mask], valid_mask


def reconstruct_base_scene(img1, img2, K, extension = False):
    """
    Reconstruct base scene from two views
    Returns: 3D points, camera poses, matched 2D points, and keypoints with descriptors
    """
    print("Step 1: Detecting and matching keypoints...")
    pts1, pts2, matches, kp1, kp2 = detect_and_match_keypoints(img1, img2)
    print(f"  Found {len(matches)} good matches")

    print("Step 2: Estimating Essential Matrix...")

    if extension:
        print("  Using EXTENSION implementation of Essential Matrix estimation")
        E, mask = extension_estimate_essential_matrix(pts1, pts2, K)
    else:
        E, mask = estimate_essential_matrix(pts1, pts2, K)

    # Filter inliers
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    print(f"  RANSAC inliers: {np.sum(mask)}/{len(mask)}")

    print("Step 3: Extracting camera pose...")
    R, t, pose_mask = extract_pose(E, pts1_inliers, pts2_inliers, K)

    # Create projection matrices
    # Camera 1: Identity pose [I | 0]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # Camera 2: [R | t]
    P2 = K @ np.hstack([R, t])

    print("Step 4: Triangulating 3D points...")
    points_3d = triangulate_points(pts1_inliers, pts2_inliers, P1, P2)

    # Filter points in front of both cameras
    points_3d_valid, pts1_valid, valid_mask = filter_points_in_front(
        points_3d, R, t, pts1_inliers, mask[mask.ravel() == 1]
    )

    print(f"  Valid 3D points: {len(points_3d_valid)}")

    # Store camera poses
    cameras = [
        {'R': np.eye(3), 't': np.zeros((3, 1)), 'P': P1},
        {'R': R, 't': t, 'P': P2}
    ]

    # Store 2D-3D correspondences for future views
    point_cloud = {
        'points_3d': points_3d_valid,
        'colors': []
    }

    # Also return keypoints and matches info for tracking
    pts2_valid = pts2_inliers[valid_mask]

    return point_cloud, cameras, pts1_valid, pts2_valid, kp1, kp2

def estimate_pose_pnp(pts_2d, pts_3d, K):
    """
    Estimate camera pose using PnP with RANSAC

    Args:
        pts_2d: 2D image points (Nx2)
        pts_3d: Corresponding 3D points (Nx3)
        K: Camera intrinsic matrix

    Returns:
        R: Rotation matrix
        t: Translation vector
        inliers: Inlier mask
    """
    # Use solvePnPRansac for robust pose estimation
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.astype(np.float32),
        pts_2d.astype(np.float32),
        K,
        None,  # No distortion
        reprojectionError=8.0,
        confidence=0.99,
        iterationsCount=1000,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None:
        return None, None, None

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec

    return R, t, inliers


def incremental_sfm(images, K, extension = False):
    """
    Perform incremental Structure from Motion with proper PnP-based registration
    """
    print("="*60)
    print("STRUCTURE FROM MOTION PIPELINE")
    print("="*60)

    # Step 1: Reconstruct base scene from first two images
    print("\n[BASE SCENE RECONSTRUCTION]")
    point_cloud, cameras, pts_2d_view0, pts_2d_view1, _, _ = reconstruct_base_scene(images[0], images[1], K, extension)

    # Track 2D-3D correspondences
    # Structure: List where each element corresponds to a 3D point
    # Each element is a dict: {view_idx: 2d_point}
    point_3d_observations = []
    for i in range(len(pts_2d_view0)):
        point_3d_observations.append({
            0: pts_2d_view0[i],
            1: pts_2d_view1[i]
        })

    points_3d_all = point_cloud['points_3d']

    # Step 2: Incrementally add remaining views using PnP
    print("\n[INCREMENTAL VIEW ADDITION WITH PnP]")
    for view_idx in range(2, len(images)):
        print(f"\n--- Processing View {view_idx+1}/{len(images)} ---")

        # Step 2.1: Match with previous view(s) to find 2D-3D correspondences
        # Try matching with the last view
        ref_view_idx = view_idx - 1

        pts_ref, pts_new, matches, _, _ = detect_and_match_keypoints(
            images[ref_view_idx], images[view_idx]
        )

        print(f"  Matched with view {ref_view_idx+1}: {len(matches)} matches")

        # Find which matches correspond to existing 3D points
        pts_2d_for_pnp = []
        pts_3d_for_pnp = []
        matched_3d_indices = []

        for pt_ref, pt_new in zip(pts_ref, pts_new):
            # Check all 3D points to see if any are observed in ref_view_idx at pt_ref
            for idx_3d, observations in enumerate(point_3d_observations):
                if ref_view_idx in observations:
                    obs_pt = observations[ref_view_idx]
                    # Check if this is the same point (within tolerance)
                    dist = np.linalg.norm(obs_pt - pt_ref)
                    if dist < 1.0:  # Tight threshold for accuracy
                        pts_2d_for_pnp.append(pt_new)
                        pts_3d_for_pnp.append(points_3d_all[idx_3d])
                        matched_3d_indices.append(idx_3d)
                        break  # Found the correspondence, move to next match

        if len(pts_2d_for_pnp) < 6:     #we only need 3 (see lecture), but to be safe we double that
            print(f"  WARNING: Only {len(pts_2d_for_pnp)} 2D-3D correspondences! Skipping view {view_idx+1}")
            continue

        pts_2d_for_pnp = np.array(pts_2d_for_pnp)
        pts_3d_for_pnp = np.array(pts_3d_for_pnp)

        print(f"  Found {len(pts_2d_for_pnp)} unique 2D-3D correspondences")

        # Step 2.2: Estimate camera pose using PnP
        print(f"  Estimating camera pose with PnP...")
        R_new, t_new, inliers = estimate_pose_pnp(pts_2d_for_pnp, pts_3d_for_pnp, K)

        if R_new is None:
            print(f"  WARNING: PnP failed! Skipping view {view_idx+1}")
            continue

        inlier_ratio = 100 * len(inliers) / len(pts_2d_for_pnp)
        print(f"  PnP inliers: {len(inliers)}/{len(pts_2d_for_pnp)} ({inlier_ratio:.1f}%)")

        # Add observations for the matched 3D points
        for i, idx_3d in enumerate(matched_3d_indices):
            if i in inliers.ravel():  # Only add PnP inliers
                point_3d_observations[idx_3d][view_idx] = pts_2d_for_pnp[i]

        # Create projection matrix for new camera
        P_new = K @ np.hstack([R_new, t_new])
        cameras.append({'R': R_new, 't': t_new, 'P': P_new})

        # Step 2.3: Triangulate new 3D points from unmatched features
        # Find features in pts_ref/pts_new that don't correspond to existing 3D points
        new_pts_2d_prev = []
        new_pts_2d_new = []

        for pt_ref, pt_new in zip(pts_ref, pts_new):
            is_existing = False
            for observations in point_3d_observations:
                if ref_view_idx in observations:
                    obs_pt = observations[ref_view_idx]
                    if np.linalg.norm(obs_pt - pt_ref) < 1.0:
                        is_existing = True
                        break

            if not is_existing:
                new_pts_2d_prev.append(pt_ref)
                new_pts_2d_new.append(pt_new)

        if len(new_pts_2d_prev) > 0:
            new_pts_2d_prev = np.array(new_pts_2d_prev)
            new_pts_2d_new = np.array(new_pts_2d_new)

            prev_camera = cameras[ref_view_idx]
            P_prev = prev_camera['P']
            points_3d_new = triangulate_points(new_pts_2d_prev, new_pts_2d_new, P_prev, P_new)

            # Filter valid points
            R_rel = R_new @ prev_camera['R'].T
            t_rel = R_new @ (-prev_camera['R'].T @ prev_camera['t']) + t_new

            dummy_mask = np.ones((len(points_3d_new), 1))
            points_3d_valid, pts_2d_prev_valid, valid_mask = filter_points_in_front(
                points_3d_new, R_rel, t_rel, new_pts_2d_prev, dummy_mask
            )

            print(f"  Triangulated {len(points_3d_valid)} new 3D points")

            # Add new 3D points and their observations
            if len(points_3d_valid) > 0:
                points_3d_all = np.vstack([points_3d_all, points_3d_valid])
                pts_2d_new_valid = new_pts_2d_new[valid_mask]

                for i in range(len(points_3d_valid)):
                    point_3d_observations.append({
                        ref_view_idx: pts_2d_prev_valid[i],
                        view_idx: pts_2d_new_valid[i]
                    })
        else:
            print(f"  No new points to triangulate")

    print(f"\n[FINAL STATISTICS]")
    print(f"  Total cameras: {len(cameras)}")
    print(f"  Total 3D points: {len(points_3d_all)}")

    return points_3d_all, cameras, point_3d_observations


def filter_outliers(point_cloud, cameras, method='statistical', camera_box_scale=3.0):
    """
    Filter outlier points to improve visualization

    Args:
        point_cloud: Nx3 array of 3D points
        cameras: List of camera dictionaries
        method: 'statistical' or 'camera_box'
        camera_box_scale: Scale factor for bounding box around cameras

    Returns:
        filtered_point_cloud: Points with outliers removed
    """
    # Get camera positions
    camera_positions = []
    for cam in cameras:
        C = -cam['R'].T @ cam['t']
        camera_positions.append(C.ravel())
    camera_positions = np.array(camera_positions)

    if method == 'camera_box':
        # Create bounding box around cameras
        cam_min = camera_positions.min(axis=0)
        cam_max = camera_positions.max(axis=0)
        cam_center = (cam_min + cam_max) / 2
        cam_range = cam_max - cam_min

        # Expand box by scale factor
        box_min = cam_center - camera_box_scale * cam_range
        box_max = cam_center + camera_box_scale * cam_range

        # Filter points inside box
        valid_mask = (
            (point_cloud[:, 0] >= box_min[0]) & (point_cloud[:, 0] <= box_max[0]) &
            (point_cloud[:, 1] >= box_min[1]) & (point_cloud[:, 1] <= box_max[1]) &
            (point_cloud[:, 2] >= box_min[2]) & (point_cloud[:, 2] <= box_max[2])
        )

        filtered_points = point_cloud[valid_mask]
        print(f"  Camera box filter: kept {len(filtered_points)}/{len(point_cloud)} points")

    else:  # statistical method
        # Calculate statistics
        mean = np.mean(point_cloud, axis=0)
        std = np.std(point_cloud, axis=0)

        # Calculate distance from camera center
        cam_center = camera_positions.mean(axis=0)
        distances = np.linalg.norm(point_cloud - cam_center, axis=1)

        # Filter using percentile-based method
        distance_threshold = np.percentile(distances, 95)  # Keep 95% of points

        # Also filter points too far in any dimension (outliers)
        max_std_factor = 3.0
        valid_mask = (
            (np.abs(point_cloud - mean) < max_std_factor * std).all(axis=1) &
            (distances < distance_threshold)
        )

        filtered_points = point_cloud[valid_mask]
        print(f"  Statistical filter: kept {len(filtered_points)}/{len(point_cloud)} points")

    return filtered_points


def visualize_reconstruction_plotly(point_cloud, cameras, filter_outliers_flag=True, filter_method='statistical'):
    """
    Create interactive 3D visualization using Plotly

    Args:
        point_cloud: Nx3 array of 3D points
        cameras: List of camera dictionaries
        filter_outliers_flag: Whether to filter outliers
        filter_method: 'statistical' or 'camera_box'
    """
    print("\n[CREATING INTERACTIVE VISUALIZATION]")

    # Print point cloud statistics before filtering
    cam_center = np.mean([(-cam['R'].T @ cam['t']).ravel() for cam in cameras], axis=0)
    distances = np.linalg.norm(point_cloud - cam_center, axis=1)
    print(f"  Point cloud stats before filtering:")
    print(f"    Total points: {len(point_cloud)}")
    print(f"    Distance from cameras - min: {distances.min():.2f}, max: {distances.max():.2f}, median: {np.median(distances):.2f}")

    # Filter outliers for better visualization
    if filter_outliers_flag:
        print(f"  Filtering outliers using '{filter_method}' method...")
        point_cloud_filtered = filter_outliers(point_cloud, cameras, method=filter_method)
    else:
        point_cloud_filtered = point_cloud

    # Create figure
    fig = go.Figure()

    # Add 3D point cloud
    fig.add_trace(go.Scatter3d(
        x=point_cloud_filtered[:, 0],
        y=point_cloud_filtered[:, 1],
        z=point_cloud_filtered[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=point_cloud_filtered[:, 2],  # Color by depth
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title="Depth (Z)")
        ),
        name='3D Points'
    ))

    # Add cameras
    camera_positions = []
    for i, cam in enumerate(cameras):
        # Camera center in world coordinates: C = -R^T * t
        C = -cam['R'].T @ cam['t']
        camera_positions.append(C.ravel())

    camera_positions = np.array(camera_positions)

    # Add camera positions as larger markers
    fig.add_trace(go.Scatter3d(
        x=camera_positions[:, 0],
        y=camera_positions[:, 1],
        z=camera_positions[:, 2],
        mode='markers+text',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond',
        ),
        text=[f'Cam {i+1}' for i in range(len(cameras))],
        textposition='top center',
        name='Cameras'
    ))

    # Add camera coordinate frames
    for i, cam in enumerate(cameras):
        C = -cam['R'].T @ cam['t']
        C = C.ravel()

        # Camera axes (optical axis points in +Z direction in camera coords)
        axis_length = 0.5
        R = cam['R'].T  # World to camera rotation

        # X-axis (right) - Red
        x_axis = C + axis_length * R[:, 0]
        fig.add_trace(go.Scatter3d(
            x=[C[0], x_axis[0]],
            y=[C[1], x_axis[1]],
            z=[C[2], x_axis[2]],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Y-axis (down) - Green
        y_axis = C + axis_length * R[:, 1]
        fig.add_trace(go.Scatter3d(
            x=[C[0], y_axis[0]],
            y=[C[1], y_axis[1]],
            z=[C[2], y_axis[2]],
            mode='lines',
            line=dict(color='green', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Z-axis (forward/optical axis) - Blue
        z_axis = C + axis_length * R[:, 2]
        fig.add_trace(go.Scatter3d(
            x=[C[0], z_axis[0]],
            y=[C[1], z_axis[1]],
            z=[C[2], z_axis[2]],
            mode='lines',
            line=dict(color='blue', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Calculate appropriate camera view based on point cloud extent
    points_range = point_cloud_filtered.max(axis=0) - point_cloud_filtered.min(axis=0)
    max_range = points_range.max()
    points_center = (point_cloud_filtered.max(axis=0) + point_cloud_filtered.min(axis=0)) / 2

    # Set camera view at a distance proportional to the scene size
    view_distance = 1.8

    # Update layout with better aspect ratio
    fig.update_layout(
        title=f'Structure from Motion - 3D Reconstruction ({len(point_cloud_filtered)} points)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',  # Use cube for equal aspect ratio
            xaxis=dict(range=[points_center[0]-max_range/1.5, points_center[0]+max_range/1.5]),
            yaxis=dict(range=[points_center[1]-max_range/1.5, points_center[1]+max_range/1.5]),
            zaxis=dict(range=[points_center[2]-max_range/1.5, points_center[2]+max_range/1.5]),
            camera=dict(
                eye=dict(x=view_distance, y=view_distance, z=view_distance)
            )
        ),
        width=1400,
        height=900,
        showlegend=True
    )

    print("  Visualization created successfully!")
    print(f"  Scene extent: X={points_range[0]:.2f}, Y={points_range[1]:.2f}, Z={points_range[2]:.2f}")
    print("  Opening interactive plot in browser...")

    # Show the plot
    fig.show()

    return fig


def main():
    """Main Structure from Motion Pipeline"""
    name = ""
    extension = False
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "Playroom_lowres"  # Default dataset name

    if len(sys.argv) > 2:
        extension = sys.argv[2]
        if extension.lower() == 'true':
            extension = True
            print("Using EXTENSION implementation for Essential Matrix estimation")
    else:
        extension = False
        print("Using OpenCV implementation for Essential Matrix estimation")

    K_0, path, start, end = choose_dataset(name)
    print(f"Selected dataset: {name}")
    


    # Load images
    print("Loading images...")
    images = load_data(path, start, end)
    print(f"Loaded {len(images)} images\n")

    # Run incremental SfM
    point_cloud, cameras, point_observations = incremental_sfm(images, K_0, extension)

    # Visualize results
    fig = visualize_reconstruction_plotly(point_cloud, cameras)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Reconstructed {len(point_cloud)} 3D points from {len(cameras)} cameras")
    print("Check your browser for the interactive 3D visualization")

    # Evaluate reconstruction quality
    print("\n" + "="*60)
    print("EVALUATING RECONSTRUCTION QUALITY")
    print("="*60)
    stats, fig_static, fig_interactive = evaluate_sfm_reconstruction(
        point_cloud, cameras, point_observations, K_0,
        show_plots=True, save_plots=True, output_prefix=f'eval_{name}'
    )


if __name__ == "__main__":
    main()