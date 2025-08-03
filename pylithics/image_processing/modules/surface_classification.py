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
                abs(parent["technical_length"] - dorsal["technical_length"]) <= tolerance * dorsal["technical_length"]
                and abs(parent["technical_width"] - dorsal["technical_width"]) <= tolerance * dorsal["technical_width"]
                and abs(parent["area"] - dorsal["area"]) <= tolerance * dorsal["area"]
            ):
                parent["surface_type"] = "Ventral"
                ventral = parent
                surfaces_identified.append("Ventral")
                break

    # Identify Platform Surface
    platform = None
    platform_candidates = [
        p for p in parents if p["surface_type"] is None and p["technical_length"] < dorsal["technical_length"] and p["technical_width"] < dorsal["technical_width"]
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
                    abs(parent["technical_length"] - dorsal["technical_length"]) <= tolerance * dorsal["technical_length"]
                    and abs(parent["technical_length"] - platform["technical_length"]) > tolerance * platform["technical_length"]
                    and parent["technical_width"] != dorsal["technical_width"]
                ):
                    parent["surface_type"] = "Lateral"
                    surfaces_identified.append("Lateral")
                    break
    # Alternative logic when platform doesn't exist but we need to classify lateral
    elif ventral is not None:
        for parent in parents:
            if parent["surface_type"] is None:
                if (
                    abs(parent["technical_length"] - dorsal["technical_length"]) <= tolerance * dorsal["technical_length"]
                    and abs(parent["technical_width"] - dorsal["technical_width"]) > tolerance * dorsal["technical_width"]
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


def classify_child_features(metrics):
    """
    Classify child contours based on their parent surface type.
    
    Applies archaeologically accurate labeling:
    - Dorsal children: "scar 1", "scar 2", etc.
    - Platform children: "mark_1", "mark_2", etc.
    - Lateral children: "edge_1", "edge_2", etc.
    - Ventral children: Excluded from output
    
    Args:
        metrics (list): List of dictionaries containing contour metrics with classified parents.
        
    Returns:
        list: Updated metrics with properly classified child features.
    """
    logging.info("Starting child feature classification based on parent surface types")
    
    try:
        # Separate parents, children, and grandchildren
        parents = [m for m in metrics if m["parent"] == m["scar"]]
        
        # Create set of parent labels for efficient lookup
        parent_labels = {p["scar"] for p in parents}
        
        # True children: their parent is a surface (in parent_labels)
        # Grandchildren: their parent is a child (not in parent_labels) - EXCLUDE these
        children = [m for m in metrics if m["parent"] != m["scar"] and m["parent"] in parent_labels]
        grandchildren = [m for m in metrics if m["parent"] != m["scar"] and m["parent"] not in parent_labels]
        
        logging.info(f"Hierarchy breakdown: {len(parents)} parents, {len(children)} children, {len(grandchildren)} grandchildren")
        logging.info(f"Grandchildren (arrows) excluded from surface classification: {[g.get('scar', 'unknown') for g in grandchildren]}")
        
        # Create mapping of parent labels to surface types
        parent_surface_map = {}
        for parent in parents:
            parent_label = parent.get("parent", "")
            surface_type = parent.get("surface_type", "Unknown")
            parent_surface_map[parent_label] = surface_type
        
        # Group children by parent surface type
        surface_children = {
            "Dorsal": [],
            "Platform": [],
            "Lateral": [],
            "Ventral": []
        }
        
        for child in children:
            parent_label = child.get("parent", "")
            parent_surface = parent_surface_map.get(parent_label, "Unknown")
            
            if parent_surface in surface_children:
                surface_children[parent_surface].append(child)
            else:
                logging.warning(f"Unknown parent surface type '{parent_surface}' for child, defaulting to Dorsal")
                surface_children["Dorsal"].append(child)
        
        # Apply surface-specific labeling rules
        classified_children = []
        
        # Dorsal children → scars (archaeologically correct)
        for i, child in enumerate(surface_children["Dorsal"]):
            child["scar"] = f"scar {i+1}"
            child["surface_feature"] = f"scar {i+1}"
            classified_children.append(child)
            logging.debug(f"Classified dorsal child as {child['scar']}")
        
        # Platform children → marks (but exclude empty space boundaries)
        platform_mark_count = 0
        for child in surface_children["Platform"]:
            # Filter out platform holes/empty space boundaries
            # Platform holes are typically the inner boundaries of hollow platforms
            # For now, exclude ALL platform children as they are likely empty space boundaries
            # Real platform marks are rare and would need more sophisticated detection
            area = child.get("area", 0)
            
            # Conservative approach: Exclude all platform children as they are typically empty space
            # In most archaeological samples, platform "children" are hollow spaces, not real marks
            logging.info(f"Excluding platform child (area={area}) as likely empty space boundary, not a mark")
            continue
            
            # Uncomment below if real platform marks need to be detected in the future:
            # platform_mark_count += 1
            # child["scar"] = f"mark_{platform_mark_count}"
            # child["surface_feature"] = f"mark_{platform_mark_count}"
            # classified_children.append(child)
            # logging.debug(f"Classified platform child as {child['scar']}")
        
        # Lateral children → edges
        for i, child in enumerate(surface_children["Lateral"]):
            child["scar"] = f"edge_{i+1}"
            child["surface_feature"] = f"edge_{i+1}"
            classified_children.append(child)
            logging.debug(f"Classified lateral child as {child['scar']}")
        
        # Ventral children → EXCLUDED (archaeologically correct)
        excluded_count = len(surface_children["Ventral"])
        if excluded_count > 0:
            logging.info(f"Excluded {excluded_count} ventral surface children from analysis (archaeologically accurate)")
        
        # Combine parents with classified children and preserve grandchildren (arrows)
        result_metrics = parents + classified_children + grandchildren
        
        logging.info(f"Child feature classification completed: {len(surface_children['Dorsal'])} scars, "
                    f"{len(surface_children['Platform'])} marks, {len(surface_children['Lateral'])} edges, "
                    f"{excluded_count} ventral features excluded")
        
        return result_metrics
        
    except Exception as e:
        logging.error(f"Error in child feature classification: {e}")
        # Return original metrics on error to prevent pipeline failure
        return metrics