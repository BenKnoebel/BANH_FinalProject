"""
Evaluation script for Structure from Motion reconstruction quality.
Computes reprojection errors and generates statistics and visualizations.

Usage:
    from evaluate_reconstruction import evaluate_sfm_reconstruction
    stats, fig = evaluate_sfm_reconstruction(points_3d, cameras, point_observations, K)
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def project_point(X, P):
    """
    Project a 3D point into image using projection matrix P

    Args:
        X: 3D point (3,) or (N, 3)
        P: Projection matrix (3, 4)

    Returns:
        x: 2D image coordinates
    """
    if X.ndim == 1:
        X_h = np.append(X, 1)  # Convert to homogeneous
        x_h = P @ X_h
        x = x_h[:2] / x_h[2]  # Convert from homogeneous to 2D
        return x
    else:
        # Multiple points
        X_h = np.hstack([X, np.ones((len(X), 1))])  # Convert to homogeneous
        x_h = (P @ X_h.T).T
        x = x_h[:, :2] / x_h[:, 2:3]  # Convert from homogeneous to 2D
        return x


def compute_reprojection_error(points_3d, cameras, point_observations):
    """
    Compute reprojection errors for all 3D points in all views where they are observed.

    Args:
        points_3d: Nx3 array of 3D points
        cameras: List of camera dictionaries with 'P' (projection matrix)
        point_observations: List of dicts, where point_observations[i] = {view_idx: 2d_point}

    Returns:
        all_errors: List of all reprojection errors
        per_view_errors: Dict mapping view_idx to list of errors for that view
        per_point_errors: List of mean errors for each 3D point
        statistics: Dictionary of statistics
    """
    all_errors = []
    per_view_errors = {i: [] for i in range(len(cameras))}
    per_point_errors = []

    # For each 3D point
    for point_idx, (X, observations) in enumerate(zip(points_3d, point_observations)):
        point_errors = []

        # For each view where this point is observed
        for view_idx, x_observed in observations.items():
            # Get projection matrix for this view
            P = cameras[view_idx]['P']

            # Project 3D point into image
            x_pred = project_point(X, P)

            # Compute reprojection error (Euclidean distance)
            error = np.linalg.norm(x_observed - x_pred)

            all_errors.append(error)
            per_view_errors[view_idx].append(error)
            point_errors.append(error)

        # Store mean error for this point
        if point_errors:
            per_point_errors.append(np.mean(point_errors))

    # Compute statistics
    all_errors = np.array(all_errors)
    per_point_errors = np.array(per_point_errors)

    statistics = {
        'mean_error': np.mean(all_errors),
        'std_error': np.std(all_errors),
        'median_error': np.median(all_errors),
        'min_error': np.min(all_errors),
        'max_error': np.max(all_errors),
        'percentile_95': np.percentile(all_errors, 95),
        'total_observations': len(all_errors),
        'num_3d_points': len(points_3d),
        'num_cameras': len(cameras),
    }

    # Per-view statistics
    per_view_stats = {}
    for view_idx, errors in per_view_errors.items():
        if errors:
            errors_array = np.array(errors)
            per_view_stats[view_idx] = {
                'mean': np.mean(errors_array),
                'std': np.std(errors_array),
                'median': np.median(errors_array),
                'num_observations': len(errors_array)
            }

    statistics['per_view_stats'] = per_view_stats

    return all_errors, per_view_errors, per_point_errors, statistics


def visualize_reprojection_errors(all_errors, per_view_errors, per_point_errors, statistics):
    """
    Create comprehensive visualizations of reprojection errors

    Args:
        all_errors: Array of all reprojection errors
        per_view_errors: Dict mapping view_idx to errors
        per_point_errors: Array of mean errors per 3D point
        statistics: Dictionary of statistics

    Returns:
        fig: Matplotlib figure with multiple subplots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Structure from Motion - Reprojection Error Analysis', fontsize=16, fontweight='bold')

    # 1. Histogram of all reprojection errors
    ax = axes[0, 0]
    ax.hist(all_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(statistics['mean_error'], color='red', linestyle='--', linewidth=2, label=f"Mean: {statistics['mean_error']:.3f}")
    ax.axvline(statistics['median_error'], color='green', linestyle='--', linewidth=2, label=f"Median: {statistics['median_error']:.3f}")
    ax.set_xlabel('Reprojection Error (pixels)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of All Reprojection Errors', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative distribution
    ax = axes[0, 1]
    sorted_errors = np.sort(all_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, linewidth=2, color='steelblue')
    ax.axvline(statistics['percentile_95'], color='red', linestyle='--', linewidth=2,
               label=f"95th percentile: {statistics['percentile_95']:.3f}")
    ax.set_xlabel('Reprojection Error (pixels)', fontsize=11)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax.set_title('Cumulative Distribution of Errors', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Box plot per view
    ax = axes[0, 2]
    view_indices = sorted(per_view_errors.keys())
    errors_per_view = [per_view_errors[i] for i in view_indices]
    bp = ax.boxplot(errors_per_view, labels=[f'View {i+1}' for i in view_indices],
                    patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Camera View', fontsize=11)
    ax.set_ylabel('Reprojection Error (pixels)', fontsize=11)
    ax.set_title('Reprojection Errors per Camera View', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Mean error per view (bar chart)
    ax = axes[1, 0]
    means = [statistics['per_view_stats'][i]['mean'] for i in view_indices]
    stds = [statistics['per_view_stats'][i]['std'] for i in view_indices]
    x_pos = np.arange(len(view_indices))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                  color='steelblue', edgecolor='black')
    ax.set_xlabel('Camera View', fontsize=11)
    ax.set_ylabel('Mean Reprojection Error (pixels)', fontsize=11)
    ax.set_title('Mean Error per View (with std)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'View {i+1}' for i in view_indices])
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Scatter plot: Point errors vs number of observations
    ax = axes[1, 1]
    num_observations = [len(obs) for obs in per_view_errors.values() if obs]
    if len(per_point_errors) > 0:
        # Create scatter with transparency
        ax.scatter(range(len(per_point_errors)), per_point_errors,
                  alpha=0.5, s=20, c='steelblue')
        ax.axhline(np.mean(per_point_errors), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(per_point_errors):.3f}')
        ax.set_xlabel('3D Point Index', fontsize=11)
        ax.set_ylabel('Mean Reprojection Error (pixels)', fontsize=11)
        ax.set_title('Mean Error per 3D Point', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6. Statistics summary (text)
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""
    REPROJECTION ERROR STATISTICS
    ═══════════════════════════════════

    Overall Statistics:
    ───────────────────────────────────
    Mean Error:           {statistics['mean_error']:.4f} px
    Std Deviation:        {statistics['std_error']:.4f} px
    Median Error:         {statistics['median_error']:.4f} px
    Min Error:            {statistics['min_error']:.4f} px
    Max Error:            {statistics['max_error']:.4f} px
    95th Percentile:      {statistics['percentile_95']:.4f} px

    Dataset Statistics:
    ───────────────────────────────────
    Total Observations:   {statistics['total_observations']}
    Number of 3D Points:  {statistics['num_3d_points']}
    Number of Cameras:    {statistics['num_cameras']}
    Avg Obs per Point:    {statistics['total_observations']/statistics['num_3d_points']:.2f}
    """
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig


def create_interactive_error_visualization(all_errors, per_view_errors, statistics):
    """
    Create interactive Plotly visualization of reprojection errors

    Returns:
        fig: Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Error Distribution', 'Cumulative Distribution',
                       'Errors per View', 'Error Statistics per View'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "bar"}]]
    )

    # 1. Histogram
    fig.add_trace(
        go.Histogram(x=all_errors, nbinsx=50, name='Errors',
                    marker_color='steelblue', opacity=0.7),
        row=1, col=1
    )

    # 2. Cumulative distribution
    sorted_errors = np.sort(all_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    fig.add_trace(
        go.Scatter(x=sorted_errors, y=cumulative, mode='lines',
                  name='Cumulative', line=dict(color='steelblue', width=2)),
        row=1, col=2
    )

    # 3. Box plot per view
    view_indices = sorted(per_view_errors.keys())
    for view_idx in view_indices:
        fig.add_trace(
            go.Box(y=per_view_errors[view_idx], name=f'View {view_idx+1}',
                  marker_color='lightblue'),
            row=2, col=1
        )

    # 4. Mean error per view
    means = [statistics['per_view_stats'][i]['mean'] for i in view_indices]
    stds = [statistics['per_view_stats'][i]['std'] for i in view_indices]
    fig.add_trace(
        go.Bar(x=[f'View {i+1}' for i in view_indices], y=means,
              error_y=dict(type='data', array=stds),
              marker_color='steelblue', name='Mean Error'),
        row=2, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Error (pixels)", row=1, col=1)
    fig.update_xaxes(title_text="Error (pixels)", row=1, col=2)
    fig.update_xaxes(title_text="Camera View", row=2, col=1)
    fig.update_xaxes(title_text="Camera View", row=2, col=2)

    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=1, col=2)
    fig.update_yaxes(title_text="Error (pixels)", row=2, col=1)
    fig.update_yaxes(title_text="Mean Error (pixels)", row=2, col=2)

    fig.update_layout(
        title_text="Reprojection Error Analysis - Interactive",
        showlegend=False,
        height=800,
        width=1400
    )

    return fig


def print_statistics_table(statistics):
    """
    Print a formatted table of statistics

    Args:
        statistics: Dictionary of statistics
    """
    print("\n" + "="*70)
    print(" " * 15 + "REPROJECTION ERROR STATISTICS")
    print("="*70)

    print("\nOverall Statistics:")
    print("-" * 70)
    print(f"  Mean Reprojection Error:        {statistics['mean_error']:8.4f} pixels")
    print(f"  Standard Deviation:             {statistics['std_error']:8.4f} pixels")
    print(f"  Median Reprojection Error:      {statistics['median_error']:8.4f} pixels")
    print(f"  Min Reprojection Error:         {statistics['min_error']:8.4f} pixels")
    print(f"  Max Reprojection Error:         {statistics['max_error']:8.4f} pixels")
    print(f"  95th Percentile:                {statistics['percentile_95']:8.4f} pixels")

    print("\nDataset Statistics:")
    print("-" * 70)
    print(f"  Total Observations:             {statistics['total_observations']:8d}")
    print(f"  Number of 3D Points:            {statistics['num_3d_points']:8d}")
    print(f"  Number of Cameras:              {statistics['num_cameras']:8d}")
    print(f"  Average Observations per Point: {statistics['total_observations']/statistics['num_3d_points']:8.2f}")

    print("\nPer-View Statistics:")
    print("-" * 70)
    print(f"{'View':<8} {'Mean (px)':<12} {'Std (px)':<12} {'Median (px)':<12} {'Num Obs':<10}")
    print("-" * 70)

    for view_idx in sorted(statistics['per_view_stats'].keys()):
        stats = statistics['per_view_stats'][view_idx]
        print(f"{view_idx+1:<8} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
              f"{stats['median']:<12.4f} {stats['num_observations']:<10d}")

    print("="*70 + "\n")


def evaluate_sfm_reconstruction(points_3d, cameras, point_observations, K=None,
                                show_plots=True, save_plots=False, output_prefix='sfm_eval'):
    """
    Main evaluation function for Structure from Motion reconstruction.

    Args:
        points_3d: Nx3 array of reconstructed 3D points
        cameras: List of camera dictionaries with 'R', 't', and 'P' keys
        point_observations: List of dicts, where point_observations[i] = {view_idx: 2d_point}
        K: Camera intrinsic matrix (optional, for reference)
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
        output_prefix: Prefix for saved plot files

    Returns:
        statistics: Dictionary containing all computed statistics
        fig_static: Matplotlib figure with static plots
        fig_interactive: Plotly figure with interactive plots
    """
    print("\n" + "="*70)
    print(" " * 20 + "EVALUATING RECONSTRUCTION")
    print("="*70)

    # Compute reprojection errors
    print("\nComputing reprojection errors...")
    all_errors, per_view_errors, per_point_errors, statistics = compute_reprojection_error(
        points_3d, cameras, point_observations
    )

    # Print statistics table
    print_statistics_table(statistics)

    # Create visualizations
    print("Generating visualizations...")
    fig_static = visualize_reprojection_errors(all_errors, per_view_errors,
                                               per_point_errors, statistics)

    fig_interactive = create_interactive_error_visualization(all_errors, per_view_errors,
                                                             statistics)

    # Save plots if requested
    if save_plots:
        print(f"\nSaving plots with prefix '{output_prefix}'...")
        fig_static.savefig(f'{output_prefix}_static.png', dpi=300, bbox_inches='tight')
        fig_interactive.write_html(f'{output_prefix}_interactive.html')
        print(f"  Saved: {output_prefix}_static.png")
        print(f"  Saved: {output_prefix}_interactive.html")

    # Show plots if requested
    if show_plots:
        print("\nDisplaying plots...")
        plt.show()
        fig_interactive.show()

    print("\nEvaluation complete!")
    print("="*70 + "\n")

    return statistics, fig_static, fig_interactive


# Example usage (for testing)
if __name__ == "__main__":
    # This is just for testing - normally this would be called from main.py
    print("This module should be imported and used with actual SfM reconstruction data.")
    print("\nExample usage:")
    print("  from evaluate_reconstruction import evaluate_sfm_reconstruction")
    print("  stats, fig_static, fig_interactive = evaluate_sfm_reconstruction(")
    print("      points_3d, cameras, point_observations, K)")
