# ES 143: Structure from Motion Pipeline

This is the GitHub for the final project of Class ES 143
Team members: Ahmed Hasan Baky, Benjamin Knoebel del Olmo, Neha Sajja and Harry Price

## Overview

This project reconstructs 3D scene geometry from a sequence of at least 5 images captured from different viewpoints using a calibrated smartphone camera.

## Usage

### Basic Usage

There are four options to choose from:
1. Playroom (given Example Data)
2. SEC_2  (Two videos recorded in the SEC, the numbering serves as a reminder of SEC_1, the fallen brother whos calibration failed)
3. SEC_3
4. Ahmeds_Room (curtesy of Ahmed)
5. Castle (our external test Dataset from openMVG, https://github.com/openMVG/SfM_quality_evaluation/tree/master)


## Running the Pipeline

### Default (Playroom dataset)
```bash
python main.py
```

### Specific Dataset
```bash
python main.py Playroom_lowres    
python main.py Ahmeds_Room        
python main.py SEC_2              
python main.py SEC_3
python main.py castle              
```

### With Custom Implementation
```bash
python main.py Playroom_lowres true   # Uses extension Fundamental Matrix Estimation
```
The true will make the pipeline use our own fundamental matrix estimation, which is sadly quite error-prone.  

The script will:
1. Load images from the specified dataset folder
2. Run the SfM pipeline with PnP-based registration
3. Automatically filter outliers for better visualization
4. Print progress and statistics
5. Open an interactive 3D visualization in your browser
6. Compute reprojection errors and generate evaluation report
7. Save evaluation plots and statistics



## Pipeline Steps

### 1. Keypoint Detection and Matching
- **Algorithm**: SIFT (Scale-Invariant Feature Transform)
- **Matching**: FLANN-based matcher 

### 2. Base Scene Reconstruction (First Two Views)
- **Essential Matrix Estimation**: Uses RANSAC for robust estimation
- **Pose Extraction**: Decomposes essential matrix to get rotation (R) and translation (t)
- **Triangulation**: Computes 3D points from matched keypoints using `cv2.triangulatePoints()`
- **Filtering**: Removes points behind either camera (cheirality check)

### 3. Incremental View Addition (Views 3-5) - PnP-Based Registration
- **2D-3D Correspondence Finding**: Matches new view with previous views and identifies which 2D points correspond to existing 3D points
- **PnP Pose Estimation**: Uses `cv2.solvePnPRansac()` to estimate camera pose from 2D-3D correspondences
- **Point Tracking**: Maintains observations of each 3D point across multiple views
- **New Point Triangulation**: After camera registration, triangulates new 3D points from unmatched features

### 4. Interactive 3D Visualization
- **Point Cloud**: All triangulated 3D points colored by depth
- **Camera Positions**: Red diamond markers showing camera locations
- **Camera Frames**: RGB axes showing camera orientations
  - Red: X-axis (right)
  - Green: Y-axis (down)
  - Blue: Z-axis (optical axis/forward)
- **Interactivity**: Rotate, zoom, pan using Plotly controls
- **Automatic Scaling**: Scene bounds automatically adjusted to point cloud extent
- **Outlier Filtering**: Removes distant outliers that would compress the main scene
- **Aspect Ratio**: Uses cube aspect mode for equal scaling on all axes


## Reconstruction Quality Evaluation

The pipeline includes automatic evaluation of reconstruction quality using reprojection errors. For each 3D point **X** and camera **P**, we compute:

```
e = ||x_observed - P*X||₂
```

The evaluation generates:
- **Console output**: Detailed statistics table with mean, median, std, and per-view errors
- **Static plots** (`eval_{dataset}_static.png`): 6-panel figure with error distributions, box plots, and statistics
- **Interactive plots** (`eval_{dataset}_interactive.html`): Interactive Plotly visualizations


## Requirements

- OpenCV (`cv2`)
- NumPy
- Plotly
- Matplotlib (for evaluation plots)

