"""
PyLithics: Voronoi Analysis
============================

Generates Voronoi diagrams for spatial distribution analysis of
dorsal surface features. Provides tessellation patterns, spatial
density metrics, and visualization outputs.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import voronoi_diagram


def calculate_voronoi_points(
    metrics: List[Dict],
    inverted_image: np.ndarray,
    padding_factor: float = 0.02
) -> Optional[Dict]:
    """
    Calculate Voronoi diagram and convex hull from dorsal centroids.

    Parameters
    ----------
    metrics : list
        Contour metrics with centroids and surface types.
    inverted_image : np.ndarray
        Inverted binary thresholded image.
    padding_factor : float
        Padding around dorsal contour bounds (default 2%).

    Returns
    -------
    dict or None
        Voronoi data including cells, convex hull, and bounding
        box. None if no dorsal surface found.
    """
    dorsal_metrics, dorsal_indices = _find_dorsal_metrics(metrics)
    if not dorsal_metrics:
        return None

    centroids, idx_map, dorsal_contour = _extract_centroids(
        dorsal_metrics, dorsal_indices
    )
    if dorsal_contour is None:
        logging.warning(
            "No dorsal contour data available "
            "for the Dorsal surface."
        )
        return None

    points = MultiPoint(centroids)
    vor = voronoi_diagram(points, envelope=dorsal_contour)

    cells = _build_voronoi_cells(
        vor, points, dorsal_contour, idx_map
    )
    _count_shared_edges(cells)

    bbox = _padded_bounding_box(dorsal_contour, padding_factor)
    hull = points.convex_hull
    hull_metrics = _convex_hull_metrics(hull)

    return {
        'voronoi_diagram': vor,
        'voronoi_cells': cells,
        'voronoi_metrics': {'num_cells': len(cells)},
        'convex_hull': hull,
        'convex_hull_metrics': hull_metrics,
        'points': points,
        'bounding_box': bbox,
        'dorsal_contour': dorsal_contour,
    }


def _find_dorsal_metrics(
    metrics: List[Dict]
) -> Tuple[List[Dict], List[int]]:
    """Find dorsal parent and child metrics."""
    dorsal_metrics = []
    dorsal_indices = []
    dorsal_parent = None

    for i, m in enumerate(metrics):
        if (m.get("surface_type") == "Dorsal"
                and m["parent"] == m["scar"]):
            dorsal_parent = m["parent"]
            dorsal_metrics.append(m)
            dorsal_indices.append(i)
            break

    if dorsal_parent is None:
        logging.warning("No Dorsal surface parent found.")
        return [], []

    for i, m in enumerate(metrics):
        if (m["parent"] == dorsal_parent
                and m["parent"] != m["scar"]):
            dorsal_metrics.append(m)
            dorsal_indices.append(i)

    if not dorsal_metrics:
        logging.warning("No Dorsal metrics for Voronoi.")
        return [], []

    return dorsal_metrics, dorsal_indices


def _extract_centroids(
    dorsal_metrics: List[Dict],
    dorsal_indices: List[int]
) -> Tuple[List[Point], Dict[Tuple, int], Optional[Polygon]]:
    """
    Extract centroids and dorsal contour polygon.

    Returns
    -------
    tuple
        (centroids, centroid_to_metric_index, dorsal_contour)
    """
    centroids = []
    idx_map = {}
    dorsal_contour = None

    for i, m in enumerate(dorsal_metrics):
        cx, cy = m["centroid_x"], m["centroid_y"]
        centroids.append(Point(cx, cy))
        idx_map[(cx, cy)] = dorsal_indices[i]

        if (dorsal_contour is None
                and "contour" in m
                and m["parent"] == m["scar"]):
            dorsal_contour = _contour_to_polygon(m["contour"])

    return centroids, idx_map, dorsal_contour


def _contour_to_polygon(raw_contour: list) -> Polygon:
    """Convert a raw contour list to a Shapely Polygon."""
    if (isinstance(raw_contour, list)
            and isinstance(raw_contour[0], list)):
        flat = [
            (pt[0][0], pt[0][1]) for pt in raw_contour
        ]
        return Polygon(flat)
    raise ValueError(
        "Unexpected contour format. Check metrics data."
    )


def _build_voronoi_cells(
    vor, points: MultiPoint,
    dorsal_contour: Polygon,
    idx_map: Dict[Tuple, int]
) -> List[Dict]:
    """Clip Voronoi regions to dorsal contour."""
    cells = []
    point_list = list(points.geoms)

    for i, region in enumerate(vor.geoms):
        if i >= len(point_list):
            continue

        pt = point_list[i]
        clipped = region.intersection(dorsal_contour)

        if clipped.is_empty:
            continue

        if clipped.geom_type == 'Polygon':
            poly = clipped
        elif clipped.geom_type == 'MultiPolygon':
            poly = max(clipped.geoms, key=lambda p: p.area)
        else:
            continue

        cells.append({
            'polygon': poly,
            'area': poly.area,
            'shared_edges': 0,
            'metric_index': idx_map.get(
                (pt.x, pt.y), -1
            ),
        })

    return cells


def _count_shared_edges(cells: List[Dict]) -> None:
    """Count shared edges between adjacent Voronoi cells."""
    tolerance = 1e-6
    n = len(cells)

    for i in range(n):
        for j in range(i + 1, n):
            inter = (
                cells[i]['polygon'].exterior.intersection(
                    cells[j]['polygon'].exterior
                )
            )
            if inter.is_empty:
                continue

            length = 0.0
            if inter.geom_type == 'LineString':
                length = inter.length
            elif inter.geom_type == 'MultiLineString':
                length = sum(
                    line.length for line in inter.geoms
                )

            if length > tolerance:
                cells[i]['shared_edges'] += 1
                cells[j]['shared_edges'] += 1


def _padded_bounding_box(
    contour: Polygon, padding_factor: float
) -> Dict[str, float]:
    """Calculate padded bounding box from contour bounds."""
    x_min, y_min, x_max, y_max = contour.bounds
    x_pad = (x_max - x_min) * padding_factor
    y_pad = (y_max - y_min) * padding_factor
    return {
        'x_min': x_min - x_pad,
        'x_max': x_max + x_pad,
        'y_min': y_min - y_pad,
        'y_max': y_max + y_pad,
    }


def _convex_hull_metrics(hull) -> Dict[str, float]:
    """Calculate convex hull width, height, and area."""
    minx, miny, maxx, maxy = hull.bounds
    return {
        'width': maxx - minx,
        'height': maxy - miny,
        'area': hull.area,
    }


# --------------- Visualization ---------------

def visualize_voronoi_diagram(
    voronoi_data: Dict,
    inverted_image: np.ndarray,
    output_path: str,
    conversion_factor: Optional[float] = None
) -> None:
    """
    Visualize Voronoi diagram and convex hull on dorsal surface.

    Parameters
    ----------
    voronoi_data : dict
        Output from calculate_voronoi_points().
    inverted_image : np.ndarray
        Inverted binary thresholded image.
    output_path : str
        Path to save the visualization.
    conversion_factor : float, optional
        Pixels per mm. If provided, axes show millimeters.
    """
    original = cv2.bitwise_not(inverted_image)
    background = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(background)

    _draw_voronoi_patches(ax, voronoi_data)
    _draw_convex_hull(ax, voronoi_data)
    _draw_centroids(ax, voronoi_data)
    _set_plot_bounds(ax, voronoi_data, conversion_factor)

    ax.set_title("Voronoi Diagram with Convex Hull")
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    logging.info(
        "Saved Voronoi diagram visualization to: %s",
        output_path
    )


def _draw_voronoi_patches(ax, voronoi_data: Dict) -> None:
    """Draw colored Voronoi cell patches."""
    patch_list = []
    for cell in voronoi_data['voronoi_cells']:
        poly = cell['polygon']
        if poly.geom_type == 'Polygon':
            patch = MplPolygon(
                np.array(poly.exterior.coords), closed=True
            )
            patch_list.append(patch)
        elif poly.geom_type == 'MultiPolygon':
            largest = max(poly.geoms, key=lambda p: p.area)
            patch = MplPolygon(
                np.array(largest.exterior.coords), closed=True
            )
            patch_list.append(patch)

    colormap = get_cmap('tab10')
    norm = Normalize(vmin=0, vmax=len(patch_list))
    colors = [colormap(norm(i)) for i in range(len(patch_list))]

    collection = PatchCollection(
        patch_list, alpha=0.6,
        facecolor=colors, edgecolor="white", linewidths=2
    )
    ax.add_collection(collection)


def _draw_convex_hull(ax, voronoi_data: Dict) -> None:
    """Overlay the convex hull on the plot."""
    hull = voronoi_data['convex_hull']
    if hull.is_empty:
        return

    if hull.geom_type == 'Polygon':
        coords = np.array(hull.exterior.coords)
        ax.plot(
            coords[:, 0], coords[:, 1],
            color="black", linewidth=2, label="Convex Hull"
        )
    elif hull.geom_type == 'LineString':
        coords = np.array(hull.coords)
        ax.plot(
            coords[:, 0], coords[:, 1],
            color="black", linewidth=2, label="Convex Hull"
        )
    elif hull.geom_type == 'Point':
        ax.plot(hull.x, hull.y, 'ko', label="Convex Hull")


def _draw_centroids(ax, voronoi_data: Dict) -> None:
    """Plot and label dorsal surface centroids."""
    points = voronoi_data['points']
    xs = [p.x for p in points.geoms]
    ys = [p.y for p in points.geoms]
    ax.plot(
        xs, ys, 'ro', markersize=5,
        label='Dorsal Surface Centroids'
    )

    for i, p in enumerate(points.geoms):
        label = "Surface Center" if i == 0 else f"C{i}"
        ax.text(
            p.x + 10, p.y + 2, label,
            color="black", fontsize=12
        )


def _set_plot_bounds(
    ax, voronoi_data: Dict,
    conversion_factor: Optional[float]
) -> None:
    """Set plot limits and axis labels."""
    bbox = voronoi_data['bounding_box']
    ax.set_xlim(bbox['x_min'], bbox['x_max'])
    ax.set_ylim(bbox['y_max'], bbox['y_min'])

    if conversion_factor and conversion_factor > 0:
        fmt = FuncFormatter(
            lambda val, pos: f"{val / conversion_factor:.1f}"
        )
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("Horizontal Distance (mm)")
        ax.set_ylabel("Vertical Distance (mm)")
    else:
        ax.set_xlabel("Horizontal Distance (pixels)")
        ax.set_ylabel("Vertical Distance (pixels)")
