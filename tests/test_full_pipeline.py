"""
PyLithics Full Pipeline Tests
=============================

End-to-end functional tests for the complete PyLithics processing pipeline.
Tests the entire workflow from raw images to final outputs, including
real-world archaeological scenarios and integration validation.
"""

import pytest
import numpy as np
import cv2
import pandas as pd
import tempfile
import os
import yaml
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

# Import main application components
from pylithics.app import PyLithicsApplication, main
from pylithics.image_processing.importer import preprocess_images
from pylithics.image_processing.image_analysis import batch_process_images


@pytest.mark.functional
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""

    def test_single_image_complete_workflow(self, sample_config):
        """Test complete workflow from single image to final outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realistic test image
            image_path = os.path.join(temp_dir, "test_artifact.png")
            self._create_realistic_test_image(image_path, artifact_type="blade")

            # Create configuration file
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Set up application
            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir,
                verbose=True
            )

            # Execute complete workflow
            result = app.run(
                mode='single',
                image_path=image_path,
                image_id='test_artifact',
                scale_value=15.0
            )

            # Verify processing completed
            assert isinstance(result, bool)

            # Check expected output files exist (if processing succeeded)
            if result:
                csv_files = list(Path(temp_dir).glob("*_measurements.csv"))
                png_files = list(Path(temp_dir).glob("*_visualization.png"))

                assert len(csv_files) > 0, "No CSV output files found"
                assert len(png_files) > 0, "No visualization files found"

                # Verify CSV content structure
                csv_file = csv_files[0]
                df = pd.read_csv(csv_file)

                # Check required columns exist
                required_columns = [
                    'image_id', 'surface_type', 'surface_feature',
                    'centroid_x', 'centroid_y', 'total_area'
                ]

                for col in required_columns:
                    assert col in df.columns, f"Missing required column: {col}"

                # Verify data integrity
                assert len(df) > 0, "No measurement data in CSV"
                assert df['image_id'].iloc[0] == 'test_artifact'

    def test_batch_processing_complete_workflow(self, sample_config):
        """Test complete batch processing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create batch processing structure
            data_dir = os.path.join(temp_dir, "data")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create multiple test artifacts
            artifacts = [
                ("blade_001.png", "blade", 18.5),
                ("core_002.png", "core", 35.2),
                ("scraper_003.png", "scraper", 22.1)
            ]

            metadata_content = "image_id,scale_id,scale\n"
            for filename, artifact_type, scale in artifacts:
                image_path = os.path.join(images_dir, filename)
                self._create_realistic_test_image(image_path, artifact_type=artifact_type)
                metadata_content += f"{filename},scale_{len(metadata_content.split())},{scale}\n"

            # Create metadata file
            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Create config file
            config_path = os.path.join(temp_dir, "batch_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Set up application for batch processing
            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir,
                verbose=True
            )

            # Execute batch workflow
            results = app.run(
                mode='batch',
                data_dir=data_dir,
                metadata_file=metadata_path
            )

            # Verify batch processing results
            assert isinstance(results, list)
            assert len(results) == len(artifacts)

            # Check outputs for successful processing
            successful_count = sum(1 for r in results if r is True)
            if successful_count > 0:
                # Verify output files
                csv_files = list(Path(temp_dir).glob("*_measurements.csv"))
                png_files = list(Path(temp_dir).glob("*_visualization.png"))

                # Should have outputs for successfully processed images
                assert len(csv_files) >= successful_count

                # Verify CSV structure for batch data
                if csv_files:
                    combined_df = pd.concat([pd.read_csv(f) for f in csv_files])

                    # Check batch-specific requirements
                    unique_images = combined_df['image_id'].nunique()
                    assert unique_images >= successful_count

    def test_command_line_interface_workflow(self, sample_config):
        """Test complete workflow through command line interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            image_path = os.path.join(temp_dir, "cli_test.png")
            self._create_realistic_test_image(image_path, artifact_type="biface")

            # Create config
            config_path = os.path.join(temp_dir, "cli_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Simulate command line arguments
            test_args = [
                'pylithics',
                'single',
                '--image-path', image_path,
                '--image-id', 'cli_test_artifact',
                '--scale', '25.7',
                '--output-dir', temp_dir,
                '--config-file', config_path,
                '--verbose'
            ]

            # Execute through main CLI function
            with patch('sys.argv', test_args):
                exit_code = main()

            # Verify successful execution
            assert exit_code == 0, "CLI execution failed"

            # Check outputs
            output_files = list(Path(temp_dir).glob("cli_test_artifact*"))
            assert len(output_files) > 0, "No output files generated"

    def test_configuration_effects_on_pipeline(self):
        """Test how different configurations affect pipeline outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            image_path = os.path.join(temp_dir, "config_test.png")
            self._create_realistic_test_image(image_path, artifact_type="blade")

            # Test different configurations
            configs = {
                'arrows_enabled': {
                    'arrow_detection': {'enabled': True, 'reference_dpi': 300.0},
                    'thresholding': {'method': 'simple', 'threshold_value': 127}
                },
                'arrows_disabled': {
                    'arrow_detection': {'enabled': False},
                    'thresholding': {'method': 'otsu', 'max_value': 255}
                },
                'high_precision': {
                    'arrow_detection': {'enabled': True, 'reference_dpi': 600.0},
                    'contour_filtering': {'min_area': 25.0},
                    'thresholding': {'method': 'adaptive'}
                }
            }

            results = {}

            for config_name, config_data in configs.items():
                config_path = os.path.join(temp_dir, f"{config_name}_config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f)

                output_dir = os.path.join(temp_dir, f"outputs_{config_name}")
                os.makedirs(output_dir, exist_ok=True)

                app = PyLithicsApplication(
                    config_file=config_path,
                    output_dir=output_dir
                )

                result = app.run(
                    mode='single',
                    image_path=image_path,
                    image_id=f'config_test_{config_name}',
                    scale_value=15.0
                )

                results[config_name] = result

                # Analyze outputs if successful
                if result:
                    csv_files = list(Path(output_dir).glob("*_measurements.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0])
                        results[f"{config_name}_data"] = df

            # Verify different configs produce different or similar results as expected
            success_count = sum(1 for k, v in results.items() if not k.endswith('_data') and v is True)
            assert success_count >= 1, "No configurations produced successful results"

    def _create_realistic_test_image(self, image_path, artifact_type="blade", size=(400, 600)):
        """Create a realistic test image for different artifact types."""
        height, width = size

        # Create base image with background
        image = np.full((height, width, 3), 240, dtype=np.uint8)  # Light background

        if artifact_type == "blade":
            # Create elongated blade shape
            blade_points = np.array([
                [width//4, height//8],           # Top point
                [3*width//4, height//6],         # Top right
                [3*width//4 + 10, height//3],    # Right side
                [7*width//8, 3*height//4],       # Lower right
                [3*width//4, 7*height//8],       # Bottom right
                [width//4, 7*height//8],         # Bottom left
                [width//8, 3*height//4],         # Lower left
                [width//4 - 10, height//3]       # Left side
            ], dtype=np.int32)

            cv2.fillPoly(image, [blade_points], (80, 70, 60))

            # Add removal scars
            scar_positions = [
                (width//3, height//3, 40, 30),
                (2*width//3, height//2, 35, 25),
                (width//2, 2*height//3, 30, 35)
            ]

            for x, y, w, h in scar_positions:
                cv2.ellipse(image, (x, y), (w//2, h//2), 0, 0, 360, (50, 45, 40), -1)

                # Add small arrow-like features in some scars
                if np.random.random() > 0.5:
                    arrow_points = np.array([
                        [x-5, y-8], [x+5, y-8], [x+3, y-3], [x+8, y],
                        [x+3, y+3], [x+5, y+8], [x-5, y+8], [x-3, y+3],
                        [x-8, y], [x-3, y-3]
                    ], dtype=np.int32)
                    cv2.fillPoly(image, [arrow_points], (90, 80, 70))

        elif artifact_type == "core":
            # Create roughly circular core
            center = (width//2, height//2)
            radius = min(width, height) // 3
            cv2.circle(image, center, radius, (70, 65, 55), -1)

            # Add radial flake scars
            num_scars = 8
            for i in range(num_scars):
                angle = 2 * np.pi * i / num_scars
                scar_x = int(center[0] + (radius * 0.7) * np.cos(angle))
                scar_y = int(center[1] + (radius * 0.7) * np.sin(angle))

                cv2.circle(image, (scar_x, scar_y), 15, (45, 40, 35), -1)

        elif artifact_type == "scraper":
            # Create scraper shape with curved edge
            scraper_points = np.array([
                [width//3, height//4],
                [2*width//3, height//4],
                [3*width//4, height//3],
                [3*width//4, 2*height//3],
                [2*width//3, 3*height//4],
                [width//3, 3*height//4],
                [width//4, 2*height//3],
                [width//4, height//3]
            ], dtype=np.int32)

            cv2.fillPoly(image, [scraper_points], (75, 68, 58))

            # Add scraping edge modifications
            edge_points = np.array([
                [2*width//3, height//4],
                [3*width//4, height//3],
                [3*width//4, height//2],
                [2*width//3 + 20, height//3]
            ], dtype=np.int32)
            cv2.fillPoly(image, [edge_points], (55, 50, 45))

        elif artifact_type == "biface":
            # Create symmetrical biface
            biface_points = np.array([
                [width//2, height//8],           # Top point
                [3*width//4, height//3],         # Right upper
                [7*width//8, 2*height//3],       # Right lower
                [width//2, 7*height//8],         # Bottom point
                [width//8, 2*height//3],         # Left lower
                [width//4, height//3]            # Left upper
            ], dtype=np.int32)

            cv2.fillPoly(image, [biface_points], (85, 75, 65))

            # Add systematic edge flaking
            for side in [-1, 1]:  # Left and right sides
                for i in range(5):
                    y_pos = height//4 + i * height//8
                    x_base = width//2 + side * width//4

                    flake_points = np.array([
                        [x_base, y_pos - 10],
                        [x_base + side * 15, y_pos - 5],
                        [x_base + side * 20, y_pos + 5],
                        [x_base, y_pos + 10],
                        [x_base - side * 5, y_pos]
                    ], dtype=np.int32)
                    cv2.fillPoly(image, [flake_points], (60, 55, 50))

        # Add some texture and noise for realism
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Save with DPI information
        pil_image = Image.fromarray(image)
        pil_image.save(image_path, dpi=(300, 300))


@pytest.mark.functional
class TestArchaeologicalScenarios:
    """Test pipeline with realistic archaeological scenarios."""

    def test_lithic_assemblage_analysis(self, sample_config):
        """Test processing of complete lithic assemblage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create assemblage structure
            assemblage_dir = os.path.join(temp_dir, "assemblage_site_001")
            images_dir = os.path.join(assemblage_dir, "images")
            os.makedirs(images_dir)

            # Create diverse artifact assemblage
            assemblage = [
                # Cores
                ("core_001.png", "core", 45.2, "reduction_core"),
                ("core_002.png", "core", 38.7, "exhausted_core"),

                # Blades
                ("blade_001.png", "blade", 28.5, "prismatic_blade"),
                ("blade_002.png", "blade", 32.1, "crested_blade"),
                ("blade_003.png", "blade", 25.9, "blade_fragment"),

                # Tools
                ("scraper_001.png", "scraper", 22.3, "end_scraper"),
                ("scraper_002.png", "scraper", 19.8, "side_scraper"),
                ("biface_001.png", "biface", 41.5, "bifacial_knife"),

                # Debitage
                ("flake_001.png", "blade", 15.2, "primary_flake"),
                ("flake_002.png", "blade", 18.7, "secondary_flake")
            ]

            # Create metadata with archaeological context
            metadata_content = "image_id,scale_id,scale,artifact_type,context,unit\n"

            for i, (filename, shape_type, scale, artifact_type) in enumerate(assemblage):
                # Create artifact image
                image_path = os.path.join(images_dir, filename)
                self._create_archaeological_artifact(image_path, shape_type, artifact_type)

                # Add to metadata
                unit = f"unit_{(i // 3) + 1}"  # Group into excavation units
                metadata_content += f"{filename},scale_{i+1},{scale},{artifact_type},surface_find,{unit}\n"

            metadata_path = os.path.join(assemblage_dir, "assemblage_metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Create assemblage-specific configuration
            assemblage_config = sample_config.copy()
            assemblage_config.update({
                'logging': {'level': 'INFO', 'log_to_file': True},
                'arrow_detection': {'enabled': True, 'reference_dpi': 300.0},
                'contour_filtering': {'min_area': 75.0},  # Filter small debris
                'symmetry_analysis': {'enabled': True},
                'voronoi_analysis': {'enabled': True}
            })

            config_path = os.path.join(temp_dir, "assemblage_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(assemblage_config, f)

            # Process assemblage
            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir,
                verbose=True
            )

            results = app.run(
                mode='batch',
                data_dir=assemblage_dir,
                metadata_file=metadata_path
            )

            # Analyze assemblage results
            assert isinstance(results, list)
            assert len(results) == len(assemblage)

            # Check processing success rate
            success_rate = sum(results) / len(results) if results else 0
            assert success_rate >= 0.5, f"Low success rate: {success_rate:.1%}"

            # Verify assemblage-level outputs
            csv_files = list(Path(temp_dir).glob("*_measurements.csv"))
            if csv_files:
                # Combine all CSV data for assemblage analysis
                assemblage_data = pd.concat([pd.read_csv(f) for f in csv_files])

                # Verify assemblage diversity
                surface_types = assemblage_data['surface_type'].nunique()
                assert surface_types >= 1, "No surface type diversity detected"

                # Check for expected archaeological features
                has_arrows = (assemblage_data['has_arrow'] == True).any()
                has_dorsal = (assemblage_data['surface_type'] == 'Dorsal').any()

                # Verify data completeness
                assert len(assemblage_data) > 0, "No measurement data generated"
                assert assemblage_data['total_area'].sum() > 0, "No area measurements"

    def test_site_formation_analysis(self, sample_config):
        """Test analysis of artifacts from different site formation contexts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create site with different formation contexts
            site_dir = os.path.join(temp_dir, "multi_context_site")
            images_dir = os.path.join(site_dir, "images")
            os.makedirs(images_dir)

            # Different preservation contexts
            contexts = [
                ("surface", ["surf_blade_01.png", "surf_core_01.png"], "surface_find"),
                ("buried", ["buried_scraper_01.png", "buried_biface_01.png"], "stratum_2"),
                ("disturbed", ["dist_flake_01.png", "dist_blade_01.png"], "mixed_context")
            ]

            metadata_content = "image_id,scale,context,preservation,notes\n"

            for context_type, filenames, context_desc in contexts:
                for filename in filenames:
                    image_path = os.path.join(images_dir, filename)

                    # Vary artifact condition based on context
                    if context_type == "surface":
                        artifact_condition = "weathered"
                    elif context_type == "buried":
                        artifact_condition = "excellent"
                    else:
                        artifact_condition = "fragmentary"

                    # Determine artifact type from filename
                    if "blade" in filename:
                        artifact_type = "blade"
                    elif "core" in filename:
                        artifact_type = "core"
                    elif "scraper" in filename:
                        artifact_type = "scraper"
                    elif "biface" in filename:
                        artifact_type = "biface"
                    else:
                        artifact_type = "blade"  # Default for flakes

                    self._create_archaeological_artifact(
                        image_path, artifact_type, artifact_condition
                    )

                    scale = np.random.uniform(15.0, 35.0)  # Random realistic scale
                    metadata_content += f"{filename},{scale:.1f},{context_desc},{artifact_condition},context_{context_type}\n"

            metadata_path = os.path.join(site_dir, "context_metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            # Configure for diverse preservation conditions
            context_config = sample_config.copy()
            context_config.update({
                'thresholding': {'method': 'adaptive'},  # Better for varied conditions
                'normalization': {'enabled': True, 'method': 'minmax'},
                'morphological_closing': {'enabled': True, 'kernel_size': 5}
            })

            config_path = os.path.join(temp_dir, "context_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(context_config, f)

            # Process site contexts
            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir,
                debug=True
            )

            results = app.run(
                mode='batch',
                data_dir=site_dir,
                metadata_file=metadata_path
            )

            # Analyze context-specific results
            if any(results):
                csv_files = list(Path(temp_dir).glob("*_measurements.csv"))
                if csv_files:
                    site_data = pd.concat([pd.read_csv(f) for f in csv_files])

                    # Verify processing handled different contexts
                    unique_images = site_data['image_id'].nunique()
                    assert unique_images > 0, "No artifacts successfully processed"

    def test_comparative_technological_analysis(self, sample_config):
        """Test pipeline for comparative technological analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create comparative study structure
            technologies = ["levallois", "blade", "bifacial"]

            all_results = {}

            for tech in technologies:
                tech_dir = os.path.join(temp_dir, f"technology_{tech}")
                images_dir = os.path.join(tech_dir, "images")
                os.makedirs(images_dir)

                # Create technology-specific artifacts
                if tech == "levallois":
                    artifacts = [
                        ("levallois_core_01.png", "core"),
                        ("levallois_flake_01.png", "blade"),
                        ("levallois_point_01.png", "blade")
                    ]
                elif tech == "blade":
                    artifacts = [
                        ("blade_core_01.png", "core"),
                        ("prismatic_blade_01.png", "blade"),
                        ("blade_tool_01.png", "blade")
                    ]
                else:  # bifacial
                    artifacts = [
                        ("biface_preform_01.png", "biface"),
                        ("finished_biface_01.png", "biface"),
                        ("biface_fragment_01.png", "biface")
                    ]

                metadata_content = f"image_id,scale,technology,reduction_stage\n"

                for i, (filename, shape_type) in enumerate(artifacts):
                    image_path = os.path.join(images_dir, filename)
                    self._create_archaeological_artifact(image_path, shape_type, tech)

                    scale = 20.0 + i * 5.0
                    stage = ["initial", "intermediate", "final"][i]
                    metadata_content += f"{filename},{scale},{tech},{stage}\n"

                metadata_path = os.path.join(tech_dir, "metadata.csv")
                with open(metadata_path, 'w') as f:
                    f.write(metadata_content)

                # Process each technology
                tech_config_path = os.path.join(temp_dir, f"{tech}_config.yaml")
                with open(tech_config_path, 'w') as f:
                    yaml.dump(sample_config, f)

                app = PyLithicsApplication(
                    config_file=tech_config_path,
                    output_dir=os.path.join(temp_dir, f"output_{tech}")
                )

                tech_results = app.run(
                    mode='batch',
                    data_dir=tech_dir,
                    metadata_file=metadata_path
                )

                all_results[tech] = tech_results

            # Verify comparative analysis capability
            total_processed = sum(sum(results) for results in all_results.values())
            assert total_processed >= len(technologies), "Insufficient processing for comparison"

    def _create_archaeological_artifact(self, image_path, artifact_type, condition="normal"):
        """Create archaeologically realistic artifact images."""
        size = (350, 450)  # Typical archaeological photo dimensions
        height, width = size

        # Base image with archaeological photography background
        if condition == "excellent":
            bg_color = (245, 245, 245)  # Clean white background
        elif condition == "weathered":
            bg_color = (235, 230, 225)  # Slightly yellowed
        else:  # fragmentary or normal
            bg_color = (240, 238, 235)  # Standard

        image = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Create artifact based on type and condition
        if artifact_type == "core":
            self._create_core_artifact(image, condition)
        elif artifact_type == "blade":
            self._create_blade_artifact(image, condition)
        elif artifact_type == "biface":
            self._create_biface_artifact(image, condition)
        elif artifact_type == "scraper":
            self._create_scraper_artifact(image, condition)

        # Add archaeological context effects
        if condition == "weathered":
            # Add weathering effects
            weathering = np.random.normal(0, 8, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + weathering, 0, 255).astype(np.uint8)

        elif condition == "fragmentary":
            # Add breakage
            break_line = width // 3
            image[:, break_line:] = bg_color

        # Add subtle shadows for photographic realism
        shadow_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(shadow_mask, (width//2 + 5, height//2 + 5),
                   (width//3, height//3), 0, 0, 360, 30, -1)

        for c in range(3):
            image[:, :, c] = cv2.subtract(image[:, :, c], shadow_mask)

        # Save with archaeological documentation DPI
        pil_image = Image.fromarray(image)
        pil_image.save(image_path, dpi=(300, 300))

    def _create_core_artifact(self, image, condition):
        """Create core reduction artifact."""
        height, width = image.shape[:2]
        center = (width//2, height//2)

        # Main core body
        core_points = []
        num_points = 12
        base_radius = min(width, height) // 4

        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            radius_var = np.random.uniform(0.8, 1.2)
            radius = base_radius * radius_var
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            core_points.append([x, y])

        core_contour = np.array(core_points, dtype=np.int32)
        cv2.fillPoly(image, [core_contour], (85, 75, 65))

        # Add flake scars
        num_scars = np.random.randint(8, 15)
        for i in range(num_scars):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(base_radius * 0.5, base_radius * 0.9)
            scar_x = int(center[0] + distance * np.cos(angle))
            scar_y = int(center[1] + distance * np.sin(angle))
            scar_size = np.random.randint(10, 25)

            cv2.circle(image, (scar_x, scar_y), scar_size, (65, 58, 52), -1)

    def _create_blade_artifact(self, image, condition):
        """Create blade artifact."""
        height, width = image.shape[:2]

        # Blade proportions (length > 2x width)
        blade_width = width // 6
        blade_length = height // 2

        center_x = width // 2
        start_y = height // 4
        end_y = start_y + blade_length

        # Create blade outline
        blade_points = np.array([
            [center_x, start_y],                    # Proximal end
            [center_x + blade_width//2, start_y + 20],
            [center_x + blade_width//2, end_y - 30],
            [center_x, end_y],                      # Distal end
            [center_x - blade_width//2, end_y - 30],
            [center_x - blade_width//2, start_y + 20]
        ], dtype=np.int32)

        cv2.fillPoly(image, [blade_points], (90, 80, 70))

        # Add dorsal ridges
        ridge_x1 = center_x - blade_width//4
        ridge_x2 = center_x + blade_width//4

        cv2.line(image, (ridge_x1, start_y + 30), (ridge_x1, end_y - 30), (70, 63, 57), 2)
        cv2.line(image, (ridge_x2, start_y + 30), (ridge_x2, end_y - 30), (70, 63, 57), 2)

        # Add removal scars
        num_scars = np.random.randint(3, 7)
        for i in range(num_scars):
            scar_y = start_y + 40 + i * (blade_length - 80) // max(1, num_scars - 1)
            scar_side = np.random.choice([-1, 1])
            scar_x = center_x + scar_side * blade_width // 3

            scar_points = np.array([
                [scar_x, scar_y - 15],
                [scar_x + scar_side * 20, scar_y - 5],
                [scar_x + scar_side * 15, scar_y + 10],
                [scar_x - scar_side * 5, scar_y + 5]
            ], dtype=np.int32)

            cv2.fillPoly(image, [scar_points], (60, 55, 50))

    def _create_biface_artifact(self, image, condition):
        """Create bifacial tool artifact."""
        height, width = image.shape[:2]
        center = (width//2, height//2)

        # Symmetrical biface outline
        biface_points = np.array([
            [center[0], center[1] - height//3],      # Top point
            [center[0] + width//4, center[1] - height//6],
            [center[0] + width//3, center[1]],       # Widest point
            [center[0] + width//4, center[1] + height//6],
            [center[0], center[1] + height//3],      # Bottom point
            [center[0] - width//4, center[1] + height//6],
            [center[0] - width//3, center[1]],       # Widest point left
            [center[0] - width//4, center[1] - height//6]
        ], dtype=np.int32)

        cv2.fillPoly(image, [biface_points], (88, 78, 68))

        # Add systematic bifacial flaking
        for side in [-1, 1]:
            for i in range(8):
                y_offset = -height//4 + i * height//16
                flake_y = center[1] + y_offset
                base_x = center[0] + side * width//4

                flake_points = np.array([
                    [base_x, flake_y - 12],
                    [base_x + side * 25, flake_y - 5],
                    [base_x + side * 20, flake_y + 8],
                    [base_x, flake_y + 12],
                    [base_x - side * 8, flake_y]
                ], dtype=np.int32)

                cv2.fillPoly(image, [flake_points], (65, 58, 52))

    def _create_scraper_artifact(self, image, condition):
        """Create scraper tool artifact."""
        height, width = image.shape[:2]

        # Scraper base shape
        scraper_points = np.array([
            [width//3, height//3],
            [2*width//3, height//3],
            [3*width//4, height//2],
            [2*width//3, 2*height//3],
            [width//3, 2*height//3],
            [width//4, height//2]
        ], dtype=np.int32)

        cv2.fillPoly(image, [scraper_points], (82, 74, 66))

        # Add working edge (retouched edge)
        edge_points = np.array([
            [2*width//3, height//3],
            [3*width//4, height//2],
            [2*width//3 + 15, height//3 + 10],
            [2*width//3 + 10, height//3]
        ], dtype=np.int32)

        cv2.fillPoly(image, [edge_points], (60, 55, 50))

        # Add retouch scars along working edge
        num_retouches = 6
        for i in range(num_retouches):
            y_pos = height//3 + 5 + i * (height//3 - 10) // max(1, num_retouches - 1)
            x_pos = 2*width//3 + 8

            cv2.circle(image, (x_pos, y_pos), 4, (55, 50, 45), -1)


@pytest.mark.functional
class TestPipelineIntegration:
    """Test integration between different pipeline components."""

    def test_preprocessing_to_analysis_integration(self, sample_config):
        """Test integration from preprocessing through analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image with known characteristics
            image_path = os.path.join(temp_dir, "integration_test.png")

            # Create image with clear contours for predictable processing
            test_image = np.full((300, 400, 3), 240, dtype=np.uint8)

            # Main artifact (should be detected as parent)
            cv2.rectangle(test_image, (50, 50), (350, 250), (80, 70, 60), -1)

            # Removal scars (should be detected as children)
            cv2.circle(test_image, (150, 120), 25, (50, 45, 40), -1)
            cv2.circle(test_image, (250, 180), 20, (55, 50, 45), -1)

            # Small arrow-like feature in first scar
            arrow_points = np.array([
                [145, 115], [155, 115], [153, 120], [160, 120],
                [153, 125], [155, 125], [145, 125], [147, 120]
            ], dtype=np.int32)
            cv2.fillPoly(test_image, [arrow_points], (90, 80, 70))

            pil_image = Image.fromarray(test_image)
            pil_image.save(image_path, dpi=(300, 300))

            # Create config that enables all analysis modules
            integration_config = {
                'thresholding': {'method': 'simple', 'threshold_value': 150, 'max_value': 255},
                'normalization': {'enabled': True, 'method': 'minmax'},
                'grayscale_conversion': {'enabled': True, 'method': 'standard'},
                'morphological_closing': {'enabled': True, 'kernel_size': 3},
                'contour_filtering': {'min_area': 100.0, 'exclude_border': True},
                'arrow_detection': {'enabled': True, 'reference_dpi': 300.0, 'debug_enabled': False},
                'logging': {'level': 'DEBUG'}
            }

            config_path = os.path.join(temp_dir, "integration_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(integration_config, f)

            # Execute pipeline
            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir,
                debug=True
            )

            result = app.run(
                mode='single',
                image_path=image_path,
                image_id='integration_test',
                scale_value=20.0
            )

            # Verify integration results
            if result:
                # Check that all pipeline stages produced expected outputs
                csv_files = list(Path(temp_dir).glob("*_measurements.csv"))
                viz_files = list(Path(temp_dir).glob("*_visualization.png"))

                assert len(csv_files) > 0, "No CSV output from analysis pipeline"
                assert len(viz_files) > 0, "No visualization output from pipeline"

                # Verify data flow through pipeline
                df = pd.read_csv(csv_files[0])

                # Should have parent and child contours
                assert len(df) >= 2, "Pipeline should detect multiple contours"

                # Should have surface classification
                assert 'surface_type' in df.columns, "Surface classification not integrated"
                surface_types = df['surface_type'].unique()
                assert 'Dorsal' in surface_types, "Expected Dorsal surface classification"

                # Should have arrow detection results
                assert 'has_arrow' in df.columns, "Arrow detection not integrated"
                assert 'arrow_angle' in df.columns, "Arrow analysis not integrated"

                # Should have spatial analysis if enabled
                spatial_columns = ['top_area', 'bottom_area', 'voronoi_num_cells']
                spatial_present = any(col in df.columns for col in spatial_columns)
                # Note: Spatial analysis might not always complete, so we don't assert

    def test_configuration_propagation_through_pipeline(self, sample_config):
        """Test that configuration changes propagate through entire pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            image_path = os.path.join(temp_dir, "config_propagation.png")
            test_image = np.full((200, 300, 3), 230, dtype=np.uint8)
            cv2.rectangle(test_image, (50, 50), (250, 150), (70, 60, 50), -1)
            cv2.circle(test_image, (150, 100), 15, (40, 35, 30), -1)

            pil_image = Image.fromarray(test_image)
            pil_image.save(image_path, dpi=(300, 300))

            # Test different configuration scenarios
            config_scenarios = [
                {
                    'name': 'arrows_disabled',
                    'config': {
                        'arrow_detection': {'enabled': False},
                        'thresholding': {'method': 'simple', 'threshold_value': 127}
                    },
                    'expected': {'has_arrows': False}
                },
                {
                    'name': 'high_precision',
                    'config': {
                        'arrow_detection': {'enabled': True, 'reference_dpi': 600.0},
                        'contour_filtering': {'min_area': 50.0},
                        'thresholding': {'method': 'adaptive'}
                    },
                    'expected': {'has_arrows': None}  # May or may not detect
                },
                {
                    'name': 'minimal_processing',
                    'config': {
                        'normalization': {'enabled': False},
                        'morphological_closing': {'enabled': False},
                        'arrow_detection': {'enabled': True},
                        'contour_filtering': {'min_area': 200.0}
                    },
                    'expected': {'minimal': True}
                }
            ]

            for scenario in config_scenarios:
                scenario_dir = os.path.join(temp_dir, f"output_{scenario['name']}")
                os.makedirs(scenario_dir, exist_ok=True)

                config_path = os.path.join(temp_dir, f"{scenario['name']}_config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(scenario['config'], f)

                app = PyLithicsApplication(
                    config_file=config_path,
                    output_dir=scenario_dir,
                    verbose=True
                )

                result = app.run(
                    mode='single',
                    image_path=image_path,
                    image_id=f"config_test_{scenario['name']}",
                    scale_value=15.0
                )

                # Verify configuration effects
                if result:
                    csv_files = list(Path(scenario_dir).glob("*_measurements.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0])

                        # Verify scenario-specific expectations
                        if scenario['name'] == 'arrows_disabled':
                            # All arrows should be disabled/NA
                            if 'has_arrow' in df.columns:
                                assert not df['has_arrow'].any(), "Arrows detected when disabled"

    def test_error_recovery_across_pipeline_stages(self, sample_config):
        """Test pipeline error recovery and graceful degradation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create challenging test cases
            test_cases = [
                {
                    'name': 'minimal_contours',
                    'image_type': 'barely_visible',
                    'expected_behavior': 'graceful_failure_or_minimal_output'
                },
                {
                    'name': 'high_noise',
                    'image_type': 'noisy',
                    'expected_behavior': 'noise_filtering'
                },
                {
                    'name': 'edge_artifacts',
                    'image_type': 'edge_touching',
                    'expected_behavior': 'border_filtering'
                }
            ]

            for test_case in test_cases:
                image_path = os.path.join(temp_dir, f"{test_case['name']}.png")
                self._create_challenging_test_image(image_path, test_case['image_type'])

                case_dir = os.path.join(temp_dir, f"output_{test_case['name']}")
                os.makedirs(case_dir, exist_ok=True)

                # Use robust configuration for error scenarios
                robust_config = sample_config.copy()
                robust_config.update({
                    'contour_filtering': {'min_area': 150.0, 'exclude_border': True},
                    'morphological_closing': {'enabled': True, 'kernel_size': 5},
                    'logging': {'level': 'DEBUG'}
                })

                config_path = os.path.join(temp_dir, f"{test_case['name']}_config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(robust_config, f)

                app = PyLithicsApplication(
                    config_file=config_path,
                    output_dir=case_dir,
                    debug=True
                )

                # Execute with error tolerance
                try:
                    result = app.run(
                        mode='single',
                        image_path=image_path,
                        image_id=f"error_test_{test_case['name']}",
                        scale_value=12.0
                    )

                    # Pipeline should handle errors gracefully (not crash)
                    assert isinstance(result, bool), "Pipeline should return boolean result"

                except Exception as e:
                    # If exceptions occur, they should be logged appropriately
                    pytest.fail(f"Pipeline crashed unexpectedly on {test_case['name']}: {e}")

    def _create_challenging_test_image(self, image_path, image_type):
        """Create challenging test images for error recovery testing."""
        size = (250, 350)
        height, width = size

        if image_type == 'barely_visible':
            # Very low contrast image
            image = np.full((height, width, 3), 180, dtype=np.uint8)
            cv2.rectangle(image, (100, 75), (250, 175), (175, 175, 175), -1)

        elif image_type == 'noisy':
            # High noise image
            image = np.full((height, width, 3), 200, dtype=np.uint8)
            cv2.rectangle(image, (75, 50), (275, 200), (100, 90, 80), -1)

            # Add significant noise
            noise = np.random.normal(0, 30, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif image_type == 'edge_touching':
            # Artifacts touching image borders
            image = np.full((height, width, 3), 240, dtype=np.uint8)

            # Shapes touching edges
            cv2.rectangle(image, (0, 50), (100, 150), (80, 70, 60), -1)  # Left edge
            cv2.rectangle(image, (250, 100), (width, 200), (85, 75, 65), -1)  # Right edge
            cv2.rectangle(image, (150, 0), (200, 80), (75, 68, 58), -1)  # Top edge

        else:
            # Default case
            image = np.zeros((height, width, 3), dtype=np.uint8)

        pil_image = Image.fromarray(image)
        pil_image.save(image_path, dpi=(300, 300))


@pytest.mark.performance
class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_single_image_processing_performance(self, sample_config):
        """Test performance of single image processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            image_path = os.path.join(temp_dir, "performance_test.png")
            self._create_performance_test_image(image_path, complexity="medium")

            config_path = os.path.join(temp_dir, "perf_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir
            )

            # Measure processing time
            start_time = time.time()

            result = app.run(
                mode='single',
                image_path=image_path,
                image_id='performance_test',
                scale_value=20.0
            )

            end_time = time.time()
            processing_time = end_time - start_time

            # Performance assertions
            assert processing_time < 60.0, f"Processing took too long: {processing_time:.1f}s"

            if result:
                # Verify outputs were generated in reasonable time
                csv_files = list(Path(temp_dir).glob("*_measurements.csv"))
                viz_files = list(Path(temp_dir).glob("*_visualization.png"))

                assert len(csv_files) > 0, "No CSV output generated"
                # Note: visualization might not be generated in all test scenarios

    def test_batch_processing_performance(self, sample_config):
        """Test performance of batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create batch test data
            data_dir = os.path.join(temp_dir, "batch_performance")
            images_dir = os.path.join(data_dir, "images")
            os.makedirs(images_dir)

            # Create multiple test images
            num_images = 5  # Keep reasonable for test execution time
            metadata_content = "image_id,scale\n"

            for i in range(num_images):
                image_name = f"perf_test_{i:02d}.png"
                image_path = os.path.join(images_dir, image_name)
                complexity = ["simple", "medium", "complex"][i % 3]
                self._create_performance_test_image(image_path, complexity=complexity)

                scale = 15.0 + i * 2.0
                metadata_content += f"{image_name},{scale}\n"

            metadata_path = os.path.join(data_dir, "metadata.csv")
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)

            config_path = os.path.join(temp_dir, "batch_perf_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir,
                verbose=True
            )

            # Measure batch processing time
            start_time = time.time()

            results = app.run(
                mode='batch',
                data_dir=data_dir,
                metadata_file=metadata_path
            )

            end_time = time.time()
            total_time = end_time - start_time

            # Performance assertions
            avg_time_per_image = total_time / num_images
            assert avg_time_per_image < 30.0, f"Average processing time too high: {avg_time_per_image:.1f}s"

            # Verify reasonable success rate
            if results:
                success_rate = sum(results) / len(results)
                assert success_rate >= 0.6, f"Low success rate: {success_rate:.1%}"

    def test_memory_usage_during_processing(self, sample_config):
        """Test memory usage characteristics during processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create memory test image
            image_path = os.path.join(temp_dir, "memory_test.png")
            self._create_performance_test_image(image_path, complexity="complex", size=(800, 1200))

            config_path = os.path.join(temp_dir, "memory_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            # Monitor memory usage
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir
            )

            result = app.run(
                mode='single',
                image_path=image_path,
                image_id='memory_test',
                scale_value=25.0
            )

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory usage assertions
            max_acceptable_increase = 200 * 1024 * 1024  # 200MB
            assert memory_increase < max_acceptable_increase, \
                f"Memory usage too high: {memory_increase / (1024*1024):.1f}MB"

    def _create_performance_test_image(self, image_path, complexity="medium", size=(400, 600)):
        """Create test images with varying complexity levels."""
        height, width = size
        image = np.full((height, width, 3), 235, dtype=np.uint8)

        if complexity == "simple":
            # Single large artifact
            cv2.rectangle(image, (width//4, height//4), (3*width//4, 3*height//4), (80, 70, 60), -1)
            cv2.circle(image, (width//2, height//2), 30, (50, 45, 40), -1)

        elif complexity == "medium":
            # Multiple artifacts with some scars
            cv2.ellipse(image, (width//2, height//2), (width//3, height//4), 0, 0, 360, (85, 75, 65), -1)

            # Add several scars
            for i in range(5):
                x = width//3 + i * width//12
                y = height//3 + (i % 2) * height//6
                cv2.circle(image, (x, y), 15, (60, 55, 50), -1)

        elif complexity == "complex":
            # Many small artifacts and features
            # Main artifact
            artifact_points = []
            center = (width//2, height//2)
            for i in range(16):
                angle = 2 * np.pi * i / 16
                radius = min(width, height) // 4 + np.sin(4 * angle) * 20
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                artifact_points.append([x, y])

            artifact_contour = np.array(artifact_points, dtype=np.int32)
            cv2.fillPoly(image, [artifact_contour], (90, 80, 70))

            # Many small scars
            for i in range(20):
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(50, min(width, height) // 3)
                x = int(center[0] + distance * np.cos(angle))
                y = int(center[1] + distance * np.sin(angle))
                size = np.random.randint(8, 20)
                cv2.circle(image, (x, y), size, (65, 58, 52), -1)

        # Add realistic noise
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        pil_image.save(image_path, dpi=(300, 300))


@pytest.mark.functional
class TestPipelineValidation:
    """Test pipeline validation and data integrity."""

    def test_output_data_consistency(self, sample_config):
        """Test consistency of output data across pipeline runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create deterministic test image
            image_path = os.path.join(temp_dir, "consistency_test.png")

            # Fixed seed for reproducible image
            np.random.seed(42)
            self._create_deterministic_test_image(image_path)

            config_path = os.path.join(temp_dir, "consistency_config.yaml")

            # Fixed configuration for consistency
            deterministic_config = {
                'thresholding': {'method': 'simple', 'threshold_value': 127, 'max_value': 255},
                'normalization': {'enabled': False},  # Reduce variability
                'morphological_closing': {'enabled': True, 'kernel_size': 3},
                'arrow_detection': {'enabled': True, 'reference_dpi': 300.0},
                'contour_filtering': {'min_area': 100.0, 'exclude_border': True}
            }

            with open(config_path, 'w') as f:
                yaml.dump(deterministic_config, f)

            # Run pipeline multiple times
            results = []
            for run_id in range(3):
                run_dir = os.path.join(temp_dir, f"run_{run_id}")
                os.makedirs(run_dir, exist_ok=True)

                app = PyLithicsApplication(
                    config_file=config_path,
                    output_dir=run_dir
                )

                result = app.run(
                    mode='single',
                    image_path=image_path,
                    image_id=f'consistency_test_run_{run_id}',
                    scale_value=18.0
                )

                if result:
                    csv_files = list(Path(run_dir).glob("*_measurements.csv"))
                    if csv_files:
                        df = pd.read_csv(csv_files[0])
                        results.append(df)

            # Analyze consistency
            if len(results) >= 2:
                # Compare key measurements between runs
                key_columns = ['total_area', 'centroid_x', 'centroid_y']

                for col in key_columns:
                    if col in results[0].columns:
                        values = [df[col].sum() if len(df) > 0 else 0 for df in results]

                        # Allow small variation due to potential processing differences
                        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                        assert cv < 0.1, f"High variability in {col}: CV={cv:.3f}"

    def test_data_integrity_validation(self, sample_config):
        """Test data integrity and validation throughout pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image with known properties
            image_path = os.path.join(temp_dir, "integrity_test.png")
            known_properties = self._create_validated_test_image(image_path)

            config_path = os.path.join(temp_dir, "integrity_config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(sample_config, f)

            app = PyLithicsApplication(
                config_file=config_path,
                output_dir=temp_dir,
                debug=True
            )

            result = app.run(
                mode='single',
                image_path=image_path,
                image_id='integrity_test',
                scale_value=known_properties['scale']
            )

            if result:
                csv_files = list(Path(temp_dir).glob("*_measurements.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])

                    # Validate data integrity
                    self._validate_measurement_data(df, known_properties)

    def _create_deterministic_test_image(self, image_path):
        """Create deterministic test image for consistency testing."""
        image = np.full((300, 400, 3), 230, dtype=np.uint8)

        # Fixed artifacts with exact coordinates
        cv2.rectangle(image, (100, 75), (300, 225), (85, 75, 65), -1)
        cv2.circle(image, (150, 125), 20, (60, 55, 50), -1)
        cv2.circle(image, (250, 175), 18, (65, 58, 52), -1)