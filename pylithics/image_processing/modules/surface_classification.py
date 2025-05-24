import logging


def classify_parent_contours(metrics, tolerance=0.1):
    """
    Classify parent contours into surfaces: Dorsal, Ventral, Platform, Lateral.
    Robustly handles cases with fewer than all four surface types.

    Args:
        metrics (list): List of dictionaries containing contour metrics.
        tolerance (float): Dimensional tolerance for surface comparison.

    Returns:
        list: Updated metrics with surface classifications.
    """
    # Extract parent contours only
    parents = [m for m in metrics if m["parent"] == m["scar"]]

    if not parents:
        logging.warning("No parent contours found for classification.")
        return metrics

    # Initialize classification
    for parent in parents:
        parent["surface_type"] = None

    surfaces_identified = []

    # Identify Dorsal Surface (always present if any parents exist)
    try:
        dorsal = max(parents, key=lambda p: p["area"])
        dorsal["surface_type"] = "Dorsal"
        surfaces_identified.append("Dorsal")
    except ValueError:
        logging.error("Unable to identify the dorsal surface due to missing or invalid parent metrics.")
        return metrics

    # If only one parent contour, we're done - it's the dorsal surface
    if len(parents) == 1:
        logging.info("Only one parent contour found, classified as Dorsal surface.")
        return metrics

    # Identify Ventral Surface
    ventral = None
    for parent in parents:
        if parent["surface_type"] is None:
            if (
                abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                and abs(parent["width"] - dorsal["width"]) <= tolerance * dorsal["width"]
                and abs(parent["area"] - dorsal["area"]) <= tolerance * dorsal["area"]
            ):
                parent["surface_type"] = "Ventral"
                ventral = parent
                surfaces_identified.append("Ventral")
                break

    # Identify Platform Surface
    platform = None
    platform_candidates = [
        p for p in parents if p["surface_type"] is None and p["height"] < dorsal["height"] and p["width"] < dorsal["width"]
    ]
    if platform_candidates:
        platform = min(platform_candidates, key=lambda p: p["area"])
        platform["surface_type"] = "Platform"
        surfaces_identified.append("Platform")

    # Identify Lateral Surface - only if platform exists
    if platform is not None:
        for parent in parents:
            if parent["surface_type"] is None:
                if (
                    abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                    and abs(parent["height"] - platform["height"]) > tolerance * platform["height"]
                    and parent["width"] != dorsal["width"]
                ):
                    parent["surface_type"] = "Lateral"
                    surfaces_identified.append("Lateral")
                    break
    # Alternative logic when platform doesn't exist but we need to classify lateral
    elif ventral is not None:
        for parent in parents:
            if parent["surface_type"] is None:
                if (
                    abs(parent["height"] - dorsal["height"]) <= tolerance * dorsal["height"]
                    and abs(parent["width"] - dorsal["width"]) > tolerance * dorsal["width"]
                ):
                    parent["surface_type"] = "Lateral"
                    surfaces_identified.append("Lateral")
                    break

    # Assign default surface type if still None
    for parent in parents:
        if parent["surface_type"] is None:
            parent["surface_type"] = "Unclassified"

    logging.info("Classified parent contours into surfaces: %s.", ", ".join(surfaces_identified))
    return metrics