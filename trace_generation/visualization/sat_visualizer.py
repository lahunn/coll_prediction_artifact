"""
2D Separating Axis Theorem (SAT) Visualization for AABB-OBB Collision

This script visualizes the core principle of the Separating Axis Theorem by
showing the projections of an AABB (yellow) and an OBB (blue) onto all four
potential separating axes in a single figure.

This visualization is for a non-colliding case, demonstrating that if at least
one axis can be found where the projections do not overlap, the shapes are
considered separated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os


def get_rectangle_corners(center, width, height, angle_deg):
    """Calculate the corners of a rotated rectangle."""
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[c, -s], [s, c]])

    half_width = width / 2
    half_height = height / 2

    corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ]
    )

    return center + corners @ rotation_matrix.T


def project_onto_axis(corners, axis):
    """Project the corners of a shape onto a given axis."""
    projections = corners @ axis
    return np.min(projections), np.max(projections)


def visualize_projection_on_axis(ax, aabb_corners, obb_corners, axis, axis_name):
    """
    Helper function to visualize the projection of AABB and OBB on a single axis within a subplot.
    """
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Project shapes onto the axis
    aabb_min, aabb_max = project_onto_axis(aabb_corners, axis)
    obb_min, obb_max = project_onto_axis(obb_corners, axis)

    # Check for overlap on this axis
    is_separated = (aabb_max < obb_min) or (obb_max < aabb_min)

    # --- Plotting ---
    # Draw shapes
    ax.add_patch(
        Polygon(aabb_corners, closed=True, color="yellow", alpha=0.6, label="AABB")
    )
    ax.add_patch(
        Polygon(obb_corners, closed=True, color="blue", alpha=0.6, label="OBB")
    )

    # Draw the axis line
    line_start = -10 * axis
    line_end = 10 * axis
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        "k--",
        alpha=0.5,
    )

    # Draw projection lines
    for corner in aabb_corners:
        proj_point = (corner @ axis) * axis
        ax.plot([corner[0], proj_point[0]], [corner[1], proj_point[1]], "y:", alpha=0.7)

    for corner in obb_corners:
        proj_point = (corner @ axis) * axis
        ax.plot([corner[0], proj_point[0]], [corner[1], proj_point[1]], "b:", alpha=0.7)

    # Draw the projected intervals on the axis line
    aabb_proj_start = aabb_min * axis
    aabb_proj_end = aabb_max * axis
    ax.plot(
        [aabb_proj_start[0], aabb_proj_end[0]],
        [aabb_proj_start[1], aabb_proj_end[1]],
        color="orange",
        linewidth=4,
        label="AABB Projection",
    )

    obb_proj_start = obb_min * axis
    obb_proj_end = obb_max * axis
    # Shift the OBB projection slightly for visibility
    shift = axis[::-1] * np.array([-1, 1]) * 0.2
    ax.plot(
        [obb_proj_start[0] + shift[0], obb_proj_end[0] + shift[0]],
        [obb_proj_start[1] + shift[1], obb_proj_end[1] + shift[1]],
        color="dodgerblue",
        linewidth=4,
        label="OBB Projection",
    )

    # --- Final Touches ---
    ax.set_aspect("equal", "box")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    status = "Separated" if is_separated else "Overlapping"
    ax.set_title(f"Projection on {axis_name}\n(Result: {status})", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.2)


if __name__ == "__main__":
    # --- Define a non-overlapping scenario ---
    AABB_CENTER = np.array([-2.5, 0])
    AABB_DIMS = (3, 3)

    OBB_CENTER = np.array([1.5, 2.0])
    OBB_DIMS = (2, 4)
    OBB_ANGLE = 30

    # --- Get corners of the shapes ---
    aabb_corners = get_rectangle_corners(AABB_CENTER, AABB_DIMS[0], AABB_DIMS[1], 0)
    obb_corners = get_rectangle_corners(OBB_CENTER, OBB_DIMS[0], OBB_DIMS[1], OBB_ANGLE)

    # --- Define the four axes to test ---
    axes = []
    axis_names = []

    # 1. AABB's X-axis
    axes.append(np.array([1, 0]))
    axis_names.append("AABB Axis 1 (X)")

    # 2. AABB's Y-axis
    axes.append(np.array([0, 1]))
    axis_names.append("AABB Axis 2 (Y)")

    # 3. OBB's first axis
    obb_angle_rad = np.deg2rad(OBB_ANGLE)
    axes.append(np.array([np.cos(obb_angle_rad), np.sin(obb_angle_rad)]))
    axis_names.append("OBB Axis 1")

    # 4. OBB's second axis (perpendicular to the first)
    axes.append(np.array([-np.sin(obb_angle_rad), np.cos(obb_angle_rad)]))
    axis_names.append("OBB Axis 2")

    # --- Create the 2x2 plot ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("SAT Visualization for a Non-Colliding Case", fontsize=16)

    # Visualize projection on each axis
    for i, ax_subplot in enumerate(axs.flat):
        visualize_projection_on_axis(
            ax_subplot, aabb_corners, obb_corners, axes[i], axis_names[i]
        )

    # Create a single legend for the entire figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(
        rect=(0, 0.05, 1, 0.95)
    )  # Adjust layout to make space for suptitle and legend

    # Save the figure
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "figures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, "sat_4_axes_visualization.png")
    plt.savefig(filename)
    print(f"Generated plot: {filename}")
    plt.close()
