"""Tests for per-lithic JSON export."""

import json
import math
import os
import tempfile

import pytest

from pylithics.image_processing.modules.json_export import (
    SCHEMA_VERSION,
    save_measurements_to_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parent_metric(**overrides):
    base = {
        "image_id": "test_image.png",
        "parent": "parent 1",
        "scar": "parent 1",
        "surface_type": "Dorsal",
        "surface_feature": "Dorsal",
        "centroid_x": 100.0,
        "centroid_y": 100.0,
        "technical_width": 60.0,
        "technical_length": 80.0,
        "max_width": 65.0,
        "max_length": 85.0,
        "area": 4800.0,
        "aspect_ratio": 1.33,
        "perimeter": 280.0,
    }
    base.update(overrides)
    return base


def _scar_metric(name="scar 1", **overrides):
    base = {
        "image_id": "test_image.png",
        "parent": "parent 1",
        "scar": name,
        "surface_type": "Dorsal",
        "surface_feature": name,
        "centroid_x": 50.0,
        "centroid_y": 50.0,
        "max_width": 20.0,
        "max_length": 25.0,
        "area": 400.0,
        "aspect_ratio": 1.25,
        "is_cortex": False,
        "has_arrow": False,
    }
    base.update(overrides)
    return base


def _calibration(method="scale_bar", pixels_per_mm=10.0, scale_confidence=0.95):
    return {
        "calibration_method": method,
        "pixels_per_mm": pixels_per_mm,
        "scale_confidence": scale_confidence,
    }


def _save_and_load(metrics, calibration_metadata=None):
    """Write JSON to a temp file and return the parsed document."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "out.json")
        save_measurements_to_json(metrics, path, calibration_metadata)
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# save_measurements_to_json: schema and structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveMeasurementsToJson:
    """Schema-level checks against the spec at .claude/specs/JsonOutput.md."""

    def test_writes_file_at_requested_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "nested", "lithic.json")
            save_measurements_to_json(
                [_parent_metric()], path, _calibration(),
            )
            assert os.path.exists(path)
            with open(path) as f:
                json.load(f)  # Must parse cleanly

    def test_top_level_keys_match_spec(self):
        doc = _save_and_load([_parent_metric()], _calibration())
        assert set(doc.keys()) == {
            "schema_version", "image_id", "calibration", "surfaces",
        }

    def test_schema_version_is_one(self):
        doc = _save_and_load([_parent_metric()], _calibration())
        assert doc["schema_version"] == SCHEMA_VERSION == 1

    def test_image_id_pulled_from_metric(self):
        doc = _save_and_load(
            [_parent_metric(image_id="awbari.png")], _calibration(),
        )
        assert doc["image_id"] == "awbari.png"

    def test_calibration_block_holds_per_lithic_metadata(self):
        doc = _save_and_load(
            [_parent_metric()],
            _calibration(method="scale_bar", pixels_per_mm=25.2,
                         scale_confidence=1.0),
        )
        assert doc["calibration"] == {
            "method": "scale_bar",
            "pixels_per_mm": 25.2,
            "scale_confidence": 1.0,
        }

    def test_calibration_block_present_when_metadata_missing(self):
        doc = _save_and_load([_parent_metric()], calibration_metadata=None)
        assert doc["calibration"] == {
            "method": None,
            "pixels_per_mm": None,
            "scale_confidence": None,
        }


# ---------------------------------------------------------------------------
# Surface and feature nesting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurfaceAndFeatureNesting:
    """Surfaces hold their own metrics and a `features` array of children."""

    def test_dorsal_surface_includes_voronoi_and_symmetry_blocks(self):
        parent = _parent_metric(
            voronoi_num_cells=5, voronoi_cell_area=900.0,
            convex_hull_width=40.0, convex_hull_height=50.0,
            convex_hull_area=2000.0,
            top_area=1200.0, bottom_area=1300.0,
            left_area=1250.0, right_area=1250.0,
            vertical_symmetry=0.96, horizontal_symmetry=1.0,
        )
        doc = _save_and_load([parent], _calibration())

        surface = doc["surfaces"][0]
        assert surface["voronoi"]["num_cells"] == 5
        assert surface["voronoi"]["convex_hull_area"] == 2000.0
        assert surface["symmetry"]["vertical_symmetry"] == 0.96
        assert surface["symmetry"]["horizontal_symmetry"] == 1.0

    def test_non_dorsal_surfaces_have_null_voronoi_and_symmetry(self):
        ventral = _parent_metric(
            parent="parent 2", scar="parent 2",
            surface_type="Ventral", surface_feature="Ventral",
        )
        doc = _save_and_load([ventral], _calibration())

        surface = doc["surfaces"][0]
        assert surface["surface_type"] == "Ventral"
        assert surface["voronoi"] is None
        assert surface["symmetry"] is None

    def test_lateral_convexity_only_on_lateral_surface(self):
        dorsal = _parent_metric()
        lateral = _parent_metric(
            parent="parent 2", scar="parent 2",
            surface_type="Lateral", surface_feature="Lateral",
            lateral_convexity=0.84,
        )
        doc = _save_and_load([dorsal, lateral], _calibration())

        surfaces_by_type = {s["surface_type"]: s for s in doc["surfaces"]}
        assert surfaces_by_type["Dorsal"]["lateral_convexity"] is None
        assert surfaces_by_type["Lateral"]["lateral_convexity"] == 0.84

    def test_scar_count_only_on_dorsal_surface(self):
        dorsal = _parent_metric()
        ventral = _parent_metric(
            parent="parent 2", scar="parent 2",
            surface_type="Ventral", surface_feature="Ventral",
        )
        doc = _save_and_load(
            [dorsal, ventral, _scar_metric("scar 1"), _scar_metric("scar 2")],
            _calibration(),
        )
        surfaces_by_type = {s["surface_type"]: s for s in doc["surfaces"]}
        assert surfaces_by_type["Dorsal"]["scar_count"] == 2
        assert surfaces_by_type["Ventral"]["scar_count"] is None

    def test_children_appear_inside_parent_features_array(self):
        metrics = [
            _parent_metric(),
            _scar_metric("scar 1"),
            _scar_metric("scar 2"),
        ]
        doc = _save_and_load(metrics, _calibration())

        dorsal = doc["surfaces"][0]
        assert [f["surface_feature"] for f in dorsal["features"]] == [
            "scar 1", "scar 2",
        ]

    def test_cortex_children_sit_in_dorsal_features_with_flag(self):
        metrics = [
            _parent_metric(),
            _scar_metric("scar 1"),
            _scar_metric(
                "cortex 1",
                is_cortex=True,
                cortex_area=300.0,
                cortex_percentage=6.25,
            ),
        ]
        doc = _save_and_load(metrics, _calibration())

        features = doc["surfaces"][0]["features"]
        labels = {f["surface_feature"]: f for f in features}
        assert labels["scar 1"]["is_cortex"] is False
        assert labels["cortex 1"]["is_cortex"] is True
        assert labels["cortex 1"]["cortex_area"] == 300.0
        assert labels["cortex 1"]["cortex_percentage"] == 6.25

    def test_lateral_edges_nested_under_lateral_surface(self):
        dorsal = _parent_metric()
        lateral = _parent_metric(
            parent="parent 2", scar="parent 2",
            surface_type="Lateral", surface_feature="Lateral",
        )
        edge = _scar_metric(
            "edge 1",
            parent="parent 2", surface_type="Lateral",
            surface_feature="edge 1",
        )
        doc = _save_and_load([dorsal, lateral, edge], _calibration())

        surfaces_by_type = {s["surface_type"]: s for s in doc["surfaces"]}
        assert len(surfaces_by_type["Dorsal"]["features"]) == 0
        assert [f["surface_feature"]
                for f in surfaces_by_type["Lateral"]["features"]] == ["edge 1"]


# ---------------------------------------------------------------------------
# null / NaN handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNullAndNaNHandling:
    """Absent values become JSON `null`; non-finite floats become `null`."""

    def test_absent_metric_field_is_emitted_as_null(self):
        # Drop perimeter from the metric — JSON should still expose the key
        parent = _parent_metric()
        del parent["perimeter"]
        doc = _save_and_load([parent], _calibration())

        surface = doc["surfaces"][0]
        assert "perimeter" in surface
        assert surface["perimeter"] is None

    def test_nan_input_is_serialized_as_null(self):
        parent = _parent_metric(centroid_x=float("nan"))
        doc = _save_and_load([parent], _calibration())
        assert doc["surfaces"][0]["centroid_x"] is None

    def test_infinity_input_is_serialized_as_null(self):
        parent = _parent_metric(area=math.inf)
        doc = _save_and_load([parent], _calibration())
        assert doc["surfaces"][0]["total_area"] is None

    def test_booleans_are_native_json_booleans(self):
        parent = _parent_metric()
        scar = _scar_metric("scar 1", has_arrow=True, arrow_angle=42.0)
        doc = _save_and_load([parent, scar], _calibration())

        feature = doc["surfaces"][0]["features"][0]
        # JSON booleans round-trip to Python bool, never strings
        assert feature["has_arrow"] is True
        assert feature["is_cortex"] is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:

    def test_empty_metrics_writes_document_with_empty_surfaces(self):
        doc = _save_and_load([], _calibration())
        assert doc["surfaces"] == []
        assert doc["image_id"] is None
        assert doc["calibration"]["method"] == "scale_bar"

    def test_multiple_surfaces_round_trip_independently(self):
        dorsal = _parent_metric()
        platform = _parent_metric(
            parent="parent 2", scar="parent 2",
            surface_type="Platform", surface_feature="Platform",
        )
        doc = _save_and_load([dorsal, platform], _calibration())
        surface_types = [s["surface_type"] for s in doc["surfaces"]]
        assert surface_types == ["Dorsal", "Platform"]

    def test_area_is_renamed_to_total_area_in_json(self):
        """The CSV column is `total_area`; the metric dict key is `area`."""
        doc = _save_and_load([_parent_metric(area=4800.0)], _calibration())
        assert doc["surfaces"][0]["total_area"] == 4800.0
        assert "area" not in doc["surfaces"][0]


# ---------------------------------------------------------------------------
# Integration: end-to-end with the pipeline orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_export_json_flag_writes_per_lithic_files_alongside_csv():
    """A real batch run with --export_json must produce both CSV and JSON."""
    import yaml

    import cv2
    import numpy as np
    from PIL import Image

    from pylithics.app import PyLithicsApplication
    from pylithics.image_processing.config import clear_config_cache

    clear_config_cache()

    def _save_image_with_dpi(image, path, dpi=300):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path, dpi=(dpi, dpi))

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        images_dir = os.path.join(data_dir, "images")
        os.makedirs(images_dir)

        image = np.full((300, 400, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 200), (80, 70, 60), thickness=-1)
        cv2.circle(image, (110, 100), 20, (50, 45, 40), thickness=-1)
        _save_image_with_dpi(image, os.path.join(images_dir, "artifact.png"))

        meta_path = os.path.join(data_dir, "metadata.csv")
        with open(meta_path, "w") as f:
            f.write("image_id,scale_id,scale\nartifact.png,scale_1,15.0\n")

        config_path = os.path.join(temp_dir, "config.yaml")
        config = {
            "data_export": {"csv": True, "json_per_lithic": True},
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        app = PyLithicsApplication(config_file=config_path)
        result = app.run_batch_analysis(data_dir, meta_path)

        processed_dir = os.path.join(data_dir, "processed")
        json_path = os.path.join(processed_dir, "json", "artifact.json")
        csv_path = os.path.join(processed_dir, "processed_metrics.csv")

        assert result["processed_successfully"] == 1
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        with open(json_path) as f:
            doc = json.load(f)
        assert doc["image_id"] == "artifact.png"
        assert doc["schema_version"] == SCHEMA_VERSION
        assert isinstance(doc["surfaces"], list)


@pytest.mark.integration
def test_no_json_directory_when_flag_not_set():
    """Default runs produce CSV only — no `json/` subdirectory."""
    import yaml

    import cv2
    import numpy as np
    from PIL import Image

    from pylithics.app import PyLithicsApplication
    from pylithics.image_processing.config import clear_config_cache

    clear_config_cache()

    def _save_image_with_dpi(image, path, dpi=300):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path, dpi=(dpi, dpi))

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        images_dir = os.path.join(data_dir, "images")
        os.makedirs(images_dir)

        image = np.full((300, 400, 3), 240, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 200), (80, 70, 60), thickness=-1)
        _save_image_with_dpi(image, os.path.join(images_dir, "artifact.png"))

        meta_path = os.path.join(data_dir, "metadata.csv")
        with open(meta_path, "w") as f:
            f.write("image_id,scale_id,scale\nartifact.png,scale_1,15.0\n")

        # Default config — json_per_lithic is False
        app = PyLithicsApplication()
        app.run_batch_analysis(data_dir, meta_path)

        processed_dir = os.path.join(data_dir, "processed")
        assert os.path.exists(
            os.path.join(processed_dir, "processed_metrics.csv")
        )
        assert not os.path.exists(os.path.join(processed_dir, "json"))
