"""Tests for metadata reading utilities."""

import csv
from unittest.mock import patch

import pytest

from pylithics.image_processing.utils import read_metadata


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.mark.unit
class TestReadMetadata:
    """read_metadata parses metadata CSVs into a list of dicts."""

    def test_parses_valid_csv(self, sample_metadata_file):
        metadata = read_metadata(sample_metadata_file)

        assert len(metadata) == 3
        assert metadata[0] == {
            "image_id": "test_image_1.png",
            "scale_id": "scale_1",
            "scale": "10.0",
        }

    def test_preserves_additional_columns(self, tmp_path):
        path = tmp_path / "custom.csv"
        _write_csv(
            path,
            [{
                "image_id": "blade_01.png",
                "scale_id": "scale_A",
                "scale": "12.5",
                "artifact_type": "blade",
                "site": "Sector-A",
            }],
            fieldnames=["image_id", "scale_id", "scale", "artifact_type", "site"],
        )

        metadata = read_metadata(str(path))

        assert len(metadata) == 1
        entry = metadata[0]
        assert entry["artifact_type"] == "blade"
        assert entry["site"] == "Sector-A"
        assert entry["scale"] == "12.5"

    def test_handles_unicode_content(self, tmp_path):
        path = tmp_path / "unicode.csv"
        _write_csv(
            path,
            [{
                "image_id": "français.png",
                "scale_id": "échelle_1",
                "scale": "10.0",
                "notes": "Artéfact trouvé à 深圳",
            }],
            fieldnames=["image_id", "scale_id", "scale", "notes"],
        )

        metadata = read_metadata(str(path))

        assert metadata[0]["image_id"] == "français.png"
        assert metadata[0]["notes"] == "Artéfact trouvé à 深圳"

    def test_returns_empty_for_empty_file(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.touch()
        assert read_metadata(str(empty)) == []

    @pytest.mark.parametrize(
        "side_effect",
        [FileNotFoundError, PermissionError("denied")],
    )
    def test_returns_empty_and_logs_on_io_error(self, side_effect):
        with patch("builtins.open", side_effect=side_effect), \
             patch("pylithics.image_processing.utils.logging") as mock_log:
            assert read_metadata("/unused/path.csv") == []
        mock_log.error.assert_called()
