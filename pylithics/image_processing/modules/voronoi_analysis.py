import os
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.polygon import LinearRing
from shapely.ops import unary_union, voronoi_diagram


def calculate_voronoi_points(metrics, inverted_image, padding_factor=0.02):
    """
    Calculate Voronoi diagram polygons and convex hull from dorsal surface centroids.
    Also computes metrics for the Voronoi cells (number of cells, area of each cell,
    and number of shared edges) as well as the convex hull metrics (width, height, area).

    This function filters the input metrics to include only those with a dorsal surface
    classification, extracts centroids from both the dorsal parent and its child contours,
    creates a shapely MultiPoint, and then computes a Voronoi diagram clipped to the dorsal
    contour. It also calculates the convex hull of the centroids and a padded bounding box
    based on the dorsal contour bounds.

    Args:
        metrics (list): List of contour metrics containing centroids, contours, and surface types.
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        padding_factor (float): Percentage of padding to add to the dorsal contour bounding box.
                                  Defaults to 0.02 (i.e., 2%).

    Returns:
        dict: A dictionary containing:
            - 'voronoi_diagram': Shapely GeometryCollection of the Voronoi cells.
            - 'voronoi_cells': List of dictionaries for each Voronoi cell with keys:
                  'polygon' (shapely Polygon), 'area' (float), 'shared_edges' (int),
                  'metric_index' (int) - index of the corresponding metric in the input list.
            - 'voronoi_metrics': Dictionary with overall Voronoi metrics (e.g., 'num_cells').
            - 'convex_hull': Shapely Geometry representing the convex hull of the centroids.
            - 'convex_hull_metrics': Dictionary with keys 'width', 'height', and 'area'.
            - 'points': Shapely MultiPoint object of the dorsal centroids.
            - 'bounding_box': Dictionary with padded bounding box values:
                  'x_min', 'x_max', 'y_min', 'y_max'.
            - 'dorsal_contour': Shapely Polygon for the dorsal contour.
    """
    # Find all metrics related to the dorsal surface (parent and scars)
    dorsal_metrics = []
    dorsal_metric_indices = []

    # First, find the parent dorsal surface
    dorsal_parent = None
    for i, m in enumerate(metrics):
        if m.get("surface_type") == "Dorsal" and m["parent"] == m["scar"]:
            dorsal_parent = m["parent"]
            dorsal_metrics.append(m)
            dorsal_metric_indices.append(i)
            break

    if dorsal_parent is None:
        logging.warning("No Dorsal surface parent found.")
        return None

    # Then, find all scars on the dorsal surface
    for i, m in enumerate(metrics):
        # Skip the parent (already added)
        if m["parent"] == dorsal_parent and m["parent"] != m["scar"]:
            dorsal_metrics.append(m)
            dorsal_metric_indices.append(i)

    if not dorsal_metrics:
        logging.warning("No Dorsal surface metrics available for Voronoi diagram.")
        return None

    # Collect centroids and process the dorsal contour
    centroids = []
    centroid_to_metric_idx = {}  # Maps centroid coordinates to original metric index
    dorsal_contour = None

    for i, dorsal_metric in enumerate(dorsal_metrics):
        # Add centroid and track which metric it came from
        cx = dorsal_metric["centroid_x"]
        cy = dorsal_metric["centroid_y"]
        centroids.append(Point(cx, cy))

        # Store original index in the metrics list
        original_idx = dorsal_metric_indices[i]
        centroid_to_metric_idx[(cx, cy)] = original_idx

        # Extract the dorsal surface contour (only once)
        if dorsal_contour is None and "contour" in dorsal_metric and dorsal_metric["parent"] == dorsal_metric["scar"]:
            raw_contour = dorsal_metric["contour"]
            if isinstance(raw_contour, list) and isinstance(raw_contour[0], list):
                flat_contour = [(point[0][0], point[0][1]) for point in raw_contour]
                dorsal_contour = Polygon(flat_contour)
            else:
                raise ValueError("Contour format is not as expected. Please check the metrics data structure.")

    if dorsal_contour is None:
        logging.warning("No dorsal contour data available for the Dorsal surface.")
        return None

    # Create a Shapely MultiPoint from the centroids
    points = MultiPoint(centroids)

    # Generate Voronoi polygons clipped to the dorsal contour
    vor = voronoi_diagram(points, envelope=dorsal_contour)

    # Calculate the convex hull of the centroids
    convex_hull = points.convex_hull

    # Calculate padded bounding box of the dorsal contour
    x_min, y_min, x_max, y_max = dorsal_contour.bounds
    x_padding = (x_max - x_min) * padding_factor
    y_padding = (y_max - y_min) * padding_factor
    bounding_box = {
        'x_min': x_min - x_padding,
        'x_max': x_max + x_padding,
        'y_min': y_min - y_padding,
        'y_max': y_max + y_padding
    }

    # Process each Voronoi region: clip to the dorsal contour and collect valid polygons
    voronoi_cells = []

    # Convert points to list for indexed access
    point_list = list(points.geoms)

    for i, region in enumerate(vor.geoms):
        if i >= len(point_list):  # Safety check
            continue

        point = point_list[i]
        point_key = (point.x, point.y)

        clipped_region = region.intersection(dorsal_contour)

        if not clipped_region.is_empty:
            if clipped_region.geom_type == 'Polygon':
                cell_polygon = clipped_region
            elif clipped_region.geom_type == 'MultiPolygon':
                cell_polygon = max(clipped_region.geoms, key=lambda p: p.area)
            else:
                continue  # Skip geometries that are not polygons

            # Store the polygon, its area, and original metric index
            metric_idx = centroid_to_metric_idx.get(point_key, -1)
            voronoi_cells.append({
                'polygon': cell_polygon,
                'area': cell_polygon.area,
                'metric_index': metric_idx  # Track which metric this cell belongs to
            })

    num_cells = len(voronoi_cells)

    # Calculate shared edges between Voronoi cells
    shared_edges_counts = [0] * num_cells
    tolerance = 1e-6  # small tolerance for geometric comparisons
    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            inter = voronoi_cells[i]['polygon'].exterior.intersection(voronoi_cells[j]['polygon'].exterior)
            if not inter.is_empty:
                if inter.geom_type == 'LineString':
                    if inter.length > tolerance:
                        shared_edges_counts[i] += 1
                        shared_edges_counts[j] += 1
                elif inter.geom_type == 'MultiLineString':
                    total_length = sum(line.length for line in inter.geoms)
                    if total_length > tolerance:
                        shared_edges_counts[i] += 1
                        shared_edges_counts[j] += 1

    # Assemble the metrics for each Voronoi cell
    voronoi_cell_metrics = []
    for idx, cell in enumerate(voronoi_cells):
        cell_dict = {
            'polygon': cell['polygon'],
            'area': cell['area'],
            'shared_edges': shared_edges_counts[idx],
            'metric_index': cell['metric_index']  # Include the metric index
        }
        voronoi_cell_metrics.append(cell_dict)

    # Calculate convex hull metrics based on its bounding box and area
    ch_minx, ch_miny, ch_maxx, ch_maxy = convex_hull.bounds
    ch_width = ch_maxx - ch_minx
    ch_height = ch_maxy - ch_miny
    convex_hull_metrics = {
        'width': ch_width,
        'height': ch_height,
        'area': convex_hull.area
    }

    # # Voronoi cell debugging. Re-work for logging
    # print("=====================================")
    # # Print the Voronoi debug info
    # print("\n=== VORONOI DIAGRAM DEBUG INFO ===")
    # print(f"Total number of cells: {num_cells}")

    # # Print the point coordinates used to generate the Voronoi diagram
    # print("\nCentroids used to generate Voronoi cells:")
    # point_coords = [(p.x, p.y) for p in points.geoms]
    # for i, coord in enumerate(point_coords):
    #     print(f"  Point {i+1}: ({coord[0]:.2f}, {coord[1]:.2f})")

    # # Print the cell areas with corresponding point coordinates
    # print("\nVoronoi cell areas with their corresponding centroids:")
    # for i, cell in enumerate(voronoi_cell_metrics):
    #     if i < len(point_coords):
    #         print(f"  Cell {i+1}: Area = {cell['area']:.2f}, Centroid = ({point_coords[i][0]:.2f}, {point_coords[i][1]:.2f})")
    #     else:
    #         print(f"  Cell {i+1}: Area = {cell['area']:.2f}, Centroid = Unknown")

    # # Also print the mapping information for verification
    # print("\nMapping between cells and metrics:")
    # for i, cell in enumerate(voronoi_cell_metrics):
    #     metric_idx = cell['metric_index']
    #     if metric_idx != -1 and metric_idx < len(metrics):
    #         metric = metrics[metric_idx]
    #         print(f"  Cell {i+1}: Mapped to metric index {metric_idx}, Feature = {metric.get('scar', 'Unknown')}")
    #     else:
    #         print(f"  Cell {i+1}: No valid metric mapping")

    # print("=====================================")

    result = {
        'voronoi_diagram': vor,
        'voronoi_cells': voronoi_cell_metrics,
        'voronoi_metrics': {
            'num_cells': num_cells
        },
        'convex_hull': convex_hull,
        'convex_hull_metrics': convex_hull_metrics,
        'points': points,
        'bounding_box': bounding_box,
        'dorsal_contour': dorsal_contour
    }
    return result


def visualize_voronoi_diagram(voronoi_data, inverted_image, output_path):
    """
    Visualize the Voronoi diagram and convex hull on the dorsal surface. This function replicates
    the visualization features from the original generate_voronoi_diagram() function:
      - Displays the original (inverted back) image as a background.
      - Plots the Voronoi cells (clipped to the dorsal contour) as colored patches using a color-blind-friendly colormap.
      - Overlays the convex hull of the centroids.
      - Highlights and annotates all centroids (both dorsal surface and child contours) with labels.
      - Dynamically adjusts the plot bounds using the padded bounding box.
      - Does not display cell metrics on the image.

    Args:
        voronoi_data (dict): Dictionary produced by calculate_voronoi_points() containing:
            - 'voronoi_cells': List of dicts (each with 'polygon', 'area', 'shared_edges').
            - 'convex_hull': Shapely Geometry for the convex hull.
            - 'points': Shapely MultiPoint of the centroids.
            - 'bounding_box': Dict with keys 'x_min', 'x_max', 'y_min', 'y_max'.
        inverted_image (numpy.ndarray): Inverted binary thresholded image.
        output_path (str): Path to save the generated Voronoi diagram visualization.

    Returns:
        None
    """
    # Invert the image back to its original form (black illustration on white background)
    original_image = cv2.bitwise_not(inverted_image)
    background_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(background_image)

    # Prepare a list to hold matplotlib polygon patches for the Voronoi cells
    patches = []
    for cell_dict in voronoi_data['voronoi_cells']:
        cell_polygon = cell_dict['polygon']
        if cell_polygon.geom_type == 'Polygon':
            patch = MplPolygon(np.array(cell_polygon.exterior.coords), closed=True)
            patches.append(patch)
        elif cell_polygon.geom_type == 'MultiPolygon':
            largest = max(cell_polygon.geoms, key=lambda p: p.area)
            patch = MplPolygon(np.array(largest.exterior.coords), closed=True)
            patches.append(patch)
        else:
            logging.warning("Skipping Voronoi cell with unsupported geometry type: %s", cell_polygon.geom_type)

    # Use a color-blind-friendly colormap (e.g., 'tab10') and normalization
    colormap = get_cmap('tab10')
    norm = Normalize(vmin=0, vmax=len(patches))
    colors = [colormap(norm(i)) for i in range(len(patches))]

    patch_collection = PatchCollection(
        patches,
        alpha=0.6,
        facecolor=colors,
        edgecolor="white",
        linewidths=2
    )
    ax.add_collection(patch_collection)

    # Overlay the convex hull
    convex_hull = voronoi_data['convex_hull']
    if not convex_hull.is_empty:
        if convex_hull.geom_type == 'Polygon':
            hull_coords = np.array(convex_hull.exterior.coords)
            ax.plot(hull_coords[:, 0], hull_coords[:, 1], color="black", linewidth=2, label="Convex Hull")
        elif convex_hull.geom_type == 'LineString':
            hull_coords = np.array(convex_hull.coords)
            ax.plot(hull_coords[:, 0], hull_coords[:, 1], color="black", linewidth=2, label="Convex Hull")
        elif convex_hull.geom_type == 'Point':
            ax.plot(convex_hull.x, convex_hull.y, 'ko', label="Convex Hull")
        else:
            logging.warning("Convex hull has unsupported geometry type: %s", convex_hull.geom_type)

    # Highlight all centroids from the MultiPoint object and annotate them.
    points = voronoi_data['points']
    centroid_xs = [p.x for p in points.geoms]
    centroid_ys = [p.y for p in points.geoms]
    ax.plot(centroid_xs, centroid_ys, 'ro', markersize=5, label='Dorsal Surface Centroids')
    for i, p in enumerate(points.geoms):
        label = "Surface Center" if i == 0 else f"C{i}"
        ax.text(p.x + 10, p.y + 2, label, color="black", fontsize=12)

    # Adjust plot limits using the padded bounding box
    bbox = voronoi_data['bounding_box']
    ax.set_xlim(bbox['x_min'], bbox['x_max'])
    ax.set_ylim(bbox['y_max'], bbox['y_min'])  # Invert y-axis to match image coordinates

    # Set title, labels, and legend
    ax.set_title("Voronoi Diagram with Convex Hull")
    ax.set_xlabel("Horizontal Distance")
    ax.set_ylabel("Vertical Distance")
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    logging.info("Saved Voronoi diagram visualization to: %s", output_path)