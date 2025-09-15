# Troubleshooting

## Common Issues and Solutions

This guide helps you diagnose and solve the most frequent problems when using PyLithics.

## Installation Issues

### Python Version Errors

**Problem**: "Python 3.7+ required" or compatibility errors

**Solutions**:
```bash
# Check your Python version
python --version

# Install correct Python version
# macOS (using Homebrew)
brew install python@3.9

# Ubuntu/Debian
sudo apt-get install python3.9

# Windows - download from python.org
```

### Package Installation Failures

**Problem**: `pip install .` fails with dependency errors

**Solutions**:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install with verbose output to see details
pip install . -v

# Install dependencies manually if needed
pip install opencv-python-headless>=4.8.0
pip install numpy>=1.24.0
pip install pandas>=1.5.0
```

### Virtual Environment Issues

**Problem**: Virtual environment not activating or packages not found

**Solutions**:
```bash
# Verify virtual environment is active
which python  # Should show path in your venv

# Recreate virtual environment if corrupted
deactivate
rm -rf palaeo/
python3 -m venv palaeo
source palaeo/bin/activate
pip install .
```

## Image Processing Issues

### No Contours Found

**Problem**: "No contours detected" or empty results

**Diagnosis**:
```bash
# Enable debug mode to see what's happening
pylithics --data_dir ./data --meta_file ./meta.csv \\\n         --log_level DEBUG --show_thresholded_images
```

**Solutions**:
```bash
# Try different thresholding methods
pylithics --data_dir ./data --meta_file ./meta.csv \\\n         --threshold_method otsu

# For poor contrast images
pylithics --data_dir ./data --meta_file ./meta.csv \\\n         --threshold_method adaptive

# Adjust threshold value manually
pylithics --data_dir ./data --meta_file ./meta.csv \\\n         --threshold_method simple --threshold_value 100
```

### Poor Contour Detection

**Problem**: Incomplete or inaccurate contour boundaries

**Image Quality Checks**:
- Ensure high contrast (black lines on white background)
- Verify resolution is at least 300 DPI
- Check for scanning artifacts or noise
- Confirm contours are closed/complete

**Configuration Solutions**:
```yaml
# In config.yaml, try these adjustments
thresholding:\n  method: adaptive\n  adaptive_block_size: 15\n  adaptive_constant: 3\n\npreprocessing:\n  denoise: true\n  morphological_closing: true\n  closing_kernel_size: 3
```

### Scale Calculation Errors

**Problem**: Unrealistic measurements (too large/small)

**Check metadata**:
```csv
# Verify your metadata.csv format\nimage_id,scale_id,scale\nartifact_001.png,scale_001.png,10  # Scale in millimeters
```

**Common mistakes**:
- Scale value in centimeters instead of millimeters
- Missing or incorrect scale images
- Wrong image-scale associations

## Configuration Issues

### Config File Not Loading

**Problem**: Configuration changes not taking effect

**Solutions**:
```bash
# Verify config file path
pylithics --data_dir ./data --meta_file ./meta.csv \\\n         --config_file ./config.yaml --log_level DEBUG

# Check YAML syntax
python -c \"import yaml; yaml.safe_load(open('config.yaml'))\"

# Use absolute paths if needed
pylithics --data_dir $(pwd)/data --meta_file $(pwd)/meta.csv \\\n         --config_file $(pwd)/config.yaml
```

### CLI Overrides Not Working

**Problem**: Command-line arguments ignored

**Solutions**:
```bash
# Verify argument syntax\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --threshold_method=otsu  # Use = or space\n\n# Check argument spelling\npylithics --help  # See all available options

# Use debug logging to verify settings\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --log_level DEBUG
```

## Feature-Specific Issues

### Arrow Detection Problems

**Problem**: Arrows not detected or false positives

**Enable debug mode**:
```bash\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --arrow_debug --log_level DEBUG
```

**Check debug images**:
- Look in `processed/arrow_debug/`
- Review candidate detection images
- Verify DPI settings are correct

**Adjust configuration**:
```yaml\narrow_detection:\n  enabled: true\n  reference_dpi: 300.0  # Match your image DPI\n  min_area_scale_factor: 0.5  # Lower for smaller arrows\n  max_area_scale_factor: 15.0  # Higher for larger arrows\n  min_aspect_ratio: 1.2  # Lower for rounder arrows
```

### Performance Problems

**Problem**: Very slow processing

**Quick solutions**:
```bash
# Disable time-consuming features\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --disable_arrow_detection \\\n         --disable_voronoi \\\n         --disable_symmetry

# Process smaller batches\n# Split your dataset into smaller directories

# Use simpler thresholding\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --threshold_method simple
```

**Memory issues**:
```bash
# Monitor memory usage\ntop -p $(pgrep -f pylithics)\n\n# Process one image at a time if needed\n# Create single-image metadata files for large images
```

### Voronoi Analysis Issues

**Problem**: No Voronoi diagrams generated

**Check requirements**:
- Surfaces must have ≥3 scars for Voronoi analysis
- Voronoi analysis must be enabled in config
- Verify scar detection is working

**Configuration**:
```yaml\nvoronoi_analysis:\n  enabled: true\n  min_scars: 3  # Lower if needed\n  min_surface_area: 50  # Lower for small artifacts
```

## Data Issues

### Missing Output Files

**Problem**: Expected output files not created

**Check permissions**:
```bash
# Verify write permissions\nls -la processed/\nchmod 755 processed/  # If needed
```

**Check processing log**:
```bash
# Review log for errors\ntail -50 processed/pylithics.log\n\n# Look for specific error messages\ngrep ERROR processed/pylithics.log
```

### Unrealistic Measurements

**Problem**: Measurements don't match expectations

**Validation steps**:
1. Check scale information in metadata
2. Verify units (should be millimeters)
3. Review labeled images for accuracy
4. Compare with known measurements

**Debug measurements**:
```python
import pandas as pd\n\n# Load and examine data\ndf = pd.read_csv('processed/measurements.csv')\n\n# Check measurement ranges\nprint(\"Length range:\", df['technical_length'].min(), \"-\", df['technical_length'].max())\nprint(\"Area range:\", df['area'].min(), \"-\", df['area'].max())\n\n# Look for outliers\noutliers = df[df['technical_length'] > 200]  # Adjust threshold\nprint(\"Large artifacts:\", outliers[['image_id', 'technical_length']])
```

## Platform-Specific Issues

### macOS Issues

**OpenCV installation problems**:
```bash
# Try installing with conda instead of pip\nconda install opencv\n\n# Or use homebrew version\nbrew install opencv\npip install opencv-python-headless
```

**Permission errors**:
```bash
# Fix permission issues\nsudo chown -R $(whoami) /usr/local/lib/python*/site-packages/
```

### Windows Issues

**PowerShell execution policy**:
```powershell
# Allow script execution\nSet-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path length limitations**:
- Use shorter directory paths
- Move project closer to root drive
- Enable long path support in Windows

### Linux Issues

**Missing system dependencies**:
```bash
# Ubuntu/Debian\nsudo apt-get install python3-dev libopencv-dev\n\n# CentOS/RHEL\nsudo yum install python3-devel opencv-devel\n\n# Or use conda environment\nconda install opencv numpy pandas
```

## Error Message Guide

### \"FileNotFoundError\"

**Cause**: Missing image or scale files
**Solution**: Check file paths and metadata.csv entries

### \"ValueError: could not convert string to float\"

**Cause**: Invalid scale values in metadata
**Solution**: Ensure scale column contains only numbers

### \"MemoryError\"

**Cause**: Insufficient RAM for large images
**Solution**: Reduce image size or process in batches

### \"ImportError: No module named 'cv2'\"

**Cause**: OpenCV not installed
**Solution**: `pip install opencv-python-headless`

### \"yaml.scanner.ScannerError\"

**Cause**: Invalid YAML syntax in config file
**Solution**: Check indentation and special characters

## Performance Optimization

### For Large Datasets

```bash
# Minimal processing for speed\npylithics --data_dir ./large_dataset --meta_file ./meta.csv \\\n         --disable_arrow_detection \\\n         --disable_voronoi \\\n         --disable_symmetry \\\n         --threshold_method simple \\\n         --no_images
```

### Memory Management

```bash
# Process in smaller batches\nfor batch in batch_*; do\n    pylithics --data_dir ./$batch --meta_file ./$batch/meta.csv\n    # Optional: clear temp files\n    rm -rf ./$batch/processed/debug/\ndone
```

### Disabling Features for Speed

| Feature | Speed Impact | CLI Flag | Config Setting |
|---------|--------------|----------|-----------------|
| Arrow Detection | High | `--disable_arrow_detection` | `arrow_detection.enabled: false` |
| Voronoi Analysis | Medium | `--disable_voronoi` | `voronoi_analysis.enabled: false` |
| Symmetry Analysis | Low | `--disable_symmetry` | `symmetry_analysis.enabled: false` |
| Image Output | Medium | `--no_images` | N/A |

## Getting Help

### Information to Provide

When reporting issues, include:

1. **PyLithics version**: `pylithics --version`
2. **Python version**: `python --version`
3. **Operating system**: OS and version
4. **Command used**: Full command line
5. **Error message**: Complete error text
6. **Log file**: Contents of pylithics.log
7. **Sample data**: Minimal example that reproduces issue

### Where to Get Help

- **GitHub Issues**: [Report bugs and get help](https://github.com/alan-turing-institute/Palaeoanalytics/issues)
- **Documentation**: Check other sections of this guide
- **Email**: Contact the [development team](../about.md)

### Before Reporting

1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Try with sample data provided with PyLithics
4. Test with minimal configuration

## Diagnostic Commands

### System Check

```bash
# Check Python and pip\npython --version\npip --version\n\n# Check PyLithics installation\npylithics --help\n\n# Verify dependencies\npython -c \"import cv2, numpy, pandas; print('Dependencies OK')\"\n\n# Test with sample data\npylithics --data_dir ./pylithics/data --meta_file ./pylithics/data/meta_data.csv
```

### Debug Mode

```bash\n# Full debug output\npylithics --data_dir ./data --meta_file ./meta.csv \\\n         --log_level DEBUG \\\n         --show_thresholded_images \\\n         --arrow_debug
```

This will provide maximum information for diagnosing issues.