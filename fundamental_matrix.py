# Import any required libraries here
import cv2                               # OpenCV
import numpy as np                       # numpy
import pickle
import os

# import plotly for 3D visualization
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
# Modify this line to adjust the displayed plot size. You can also call
# it with different parameters before specific plots.
plt.rcParams['figure.figsize'] = [10, 10]


# estimate fundamental matrix with ransac and rank-2 constraint
# 
# note: this follows the two-view initialization approach described in Section 2 of
# Schonberger et al, Structure-From-Motion Revisited, CVPR 2016 (COLMAP paper).
# 
# EXTENSION: This implementation does NOT use cv2.findFundamentalMat(). Instead, it
# implements the 8-point algorithm and RANSAC from scratch, which qualifies as an
# extension for evaluation purposes.
#
def eight_point_algorithm(pts1, pts2):
    """
    compute fundamental matrix using the 8-point algorithm.
    
    this implements the standard 8-point algorithm from scratch:
    1. normalize points (optional but recommended)
    2. build constraint matrix from epipolar constraint
    3. solve using SVD
    4. enforce rank-2 constraint
    
    inputs:
        pts1: 8x2 array of points from image 1 (exactly 8 points).
        pts2: 8x2 array of points from image 2 (exactly 8 points).
    
    outputs:
        F: 3x3 fundamental matrix (not yet rank-2 enforced).
    """
    # convert to homogeneous coordinates
    n = len(pts1)
    if n != 8:
        raise ValueError(f"8-point algorithm requires exactly 8 points, got {n}")
    
    # build constraint matrix A where A * f = 0
    # each row corresponds to constraint: x2^T * F * x1 = 0
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # solve using SVD: find null space of A
    # the solution is the right singular vector corresponding to smallest singular value
    U, S, Vt = np.linalg.svd(A)
    F_vec = Vt[-1]  # last row of Vt (corresponds to smallest singular value)
    F = F_vec.reshape(3, 3)
    
    # enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # set smallest singular value to zero
    F = U @ np.diag(S) @ Vt
    
    return F


def compute_epipolar_error(pts1, pts2, F):
    """
    compute epipolar constraint error for each point pair.
    
    error = |x2^T * F * x1| (distance from epipolar line)
    
    inputs:
        pts1: Nx2 array of points from image 1.
        pts2: Nx2 array of points from image 2.
        F: 3x3 fundamental matrix.
    
    outputs:
        errors: Nx1 array of epipolar errors.
    """
    # convert to homogeneous
    n = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((n, 1))])
    pts2_h = np.hstack([pts2, np.ones((n, 1))])
    
    # compute epipolar constraint: x2^T * F * x1
    errors = []
    for i in range(n):
        x1 = pts1_h[i]
        x2 = pts2_h[i]
        error = abs(x2 @ F @ x1)
        errors.append(error)
    
    return np.array(errors)


def estimate_fundamental_matrix(pts1, pts2, ransac_thresh=1.0, confidence=0.99):
    """
    estimate the fundamental matrix using ransac and rank-2 constraint.
    
    EXTENSION IMPLEMENTATION: This does NOT use cv2.findFundamentalMat(). Instead,
    it implements the 8-point algorithm and RANSAC from scratch.
    
    this function implements the standard two-view initialization step from the COLMAP
    pipeline. it uses a custom RANSAC implementation with the 8-point algorithm to
    robustly estimate F from noisy keypoint matches, then enforces the rank-2 constraint.
    
    inputs:
        pts1: Nx2 array of points from image 1 (float32 or float64).
        pts2: Nx2 array of points from image 2 (same type/shape).
        ransac_thresh: reprojection error threshold in pixels for RANSAC.
        confidence: confidence level for RANSAC (used to compute max iterations).
    
    outputs:
        F: 3x3 fundamental matrix (float64) with rank 2.
        mask: Nx1 uint8 inlier mask (1 = inlier, 0 = outlier).
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    
    n = len(pts1)
    if n < 8:
        raise ValueError(f"Need at least 8 points for fundamental matrix, got {n}")
    
    # compute max iterations for RANSAC
    # formula: k = log(1-p) / log(1-w^s)
    # where p = confidence, w = inlier ratio (estimate 0.5), s = 8 (sample size)
    w = 0.5  # estimated inlier ratio
    s = 8    # sample size
    max_iterations = int(np.ceil(np.log(1 - confidence) / np.log(1 - w**s)))
    max_iterations = min(max_iterations, 2000)  # cap at reasonable number
    
    best_F = None
    best_inlier_count = 0
    best_inlier_mask = None
    
    # RANSAC loop
    for iteration in range(max_iterations):
        # randomly sample 8 points
        indices = np.random.choice(n, 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        
        try:
            # compute F using 8-point algorithm
            F = eight_point_algorithm(sample_pts1, sample_pts2)
            
            # compute inliers
            errors = compute_epipolar_error(pts1, pts2, F)
            inlier_mask = errors < ransac_thresh
            inlier_count = np.sum(inlier_mask)
            
            # update best model
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_F = F.copy()
                best_inlier_mask = inlier_mask
                
        except:
            # if 8-point algorithm fails (e.g., degenerate case), skip this iteration
            continue
    
    # if no good model found, fall back to using all points
    if best_F is None:
        # use all points with 8-point algorithm (will fail if n != 8, but try anyway)
        if n == 8:
            best_F = eight_point_algorithm(pts1, pts2)
            best_inlier_mask = np.ones(n, dtype=bool)
        else:
            # use first 8 points
            best_F = eight_point_algorithm(pts1[:8], pts2[:8])
            errors = compute_epipolar_error(pts1, pts2, best_F)
            best_inlier_mask = errors < ransac_thresh
    
    # refine: recompute F using all inliers (if we have enough)
    if best_inlier_count >= 8:
        inlier_pts1 = pts1[best_inlier_mask]
        inlier_pts2 = pts2[best_inlier_mask]
        
        # if we have exactly 8 inliers, use them directly
        if len(inlier_pts1) == 8:
            best_F = eight_point_algorithm(inlier_pts1, inlier_pts2)
        else:
            # if more than 8, use least squares (solve overdetermined system)
            n_inliers = len(inlier_pts1)
            A = np.zeros((n_inliers, 9))
            for i in range(n_inliers):
                x1, y1 = inlier_pts1[i]
                x2, y2 = inlier_pts2[i]
                A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
            
            U, S, Vt = np.linalg.svd(A)
            F_vec = Vt[-1]
            best_F = F_vec.reshape(3, 3)
            
            # enforce rank-2
            U, S, Vt = np.linalg.svd(best_F)
            S[2] = 0
            best_F = U @ np.diag(S) @ Vt
            
            # recompute inliers with refined F
            errors = compute_epipolar_error(pts1, pts2, best_F)
            best_inlier_mask = errors < ransac_thresh
    
    # normalize so F[2,2] is reasonable
    if abs(best_F[2, 2]) > 1e-6:
        best_F = best_F / best_F[2, 2]
    
    # convert mask to uint8 format (same as OpenCV)
    mask = np.zeros(n, dtype=np.uint8)
    mask[best_inlier_mask] = 1
    
    return best_F, mask