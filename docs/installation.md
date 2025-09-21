# Installation Guide

This guide will walk you through installing PyLithics on your system. PyLithics requires Python 3.7 or greater and works on macOS, Windows, and Linux.

## System Requirements

- **Python**: Version 3.7 or higher
- **Operating System**:
  - macOS 10.14 or later
  - Windows 10 or later
  - Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: 500MB for installation plus space for your data

## Step 1: Verify Prerequisites

Before installing PyLithics, ensure you have Python and Git installed on your system.

### Check Python Installation

=== "macOS & Linux"

    ```bash
    # Check Python version (should be 3.7+)
    python3 --version

    # If not installed, install Python
    # macOS (using Homebrew - install from https://brew.sh/)
    brew install python@3.7

    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install python3 python3-pip python3-venv

    # CentOS/RHEL
    sudo yum install python3 python3-pip
    ```

=== "Windows"

    ```powershell
    # Check Python version (should be 3.7+)
    python --version

    # If not installed, download from https://python.org
    # Make sure to check "Add Python to PATH" during installation
    ```

### Check Git Installation

=== "macOS & Linux"

    ```bash
    # Check Git version
    git --version

    # If not installed
    # macOS (using Homebrew - install from https://brew.sh/)
    brew install git

    # Ubuntu/Debian
    sudo apt-get install git

    # CentOS/RHEL
    sudo yum install git
    ```

=== "Windows"

    ```powershell
    # Check Git version
    git --version

    # If not installed, download from https://git-scm.com/
    ```

!!! warning "Python Version"
    PyLithics requires Python 3.7 or higher. If you have an older version, please upgrade before proceeding.

## Step 2: Set Up a Virtual Environment

We strongly recommend using a virtual environment to avoid conflicts with other Python packages.

=== "macOS & Linux"

    ```bash
    # Create virtual environment
    python3 -m venv palaeo

    # Activate virtual environment
    source palaeo/bin/activate
    ```

=== "Windows"

    ```powershell
    # Create virtual environment
    python -m venv palaeo

    # Allow script execution (may require administrator privileges)
    Set-ExecutionPolicy Unrestricted -Scope Process

    # Activate virtual environment
    .\palaeo\Scripts\activate
    ```

!!! tip "Virtual Environment Active"
    When your virtual environment is active, you'll see `(palaeo)` at the beginning of your command prompt.

## Step 3: Clone the Repository

Clone the PyLithics repository from GitHub:

```bash
git clone https://github.com/alan-turing-institute/Palaeoanalytics.git
cd Palaeoanalytics
```

### Choosing a Branch

- **Stable Version**: Use the `main` branch for the most stable release
- **Latest Features**: Use the `staging` branch for the newest features (may be less stable)

```bash
# For stable version (recommended)
git checkout main

# For latest features
git checkout staging
```

## Step 4: Install PyLithics

Install PyLithics and all its dependencies:

```bash
pip install .
```

This command will:
- Install PyLithics as a package
- Install all required dependencies listed in `requirements.txt`
- Set up command-line tools (`pylithics` and `pylithics-run`)

## Step 5: Verify Installation

Test that PyLithics is correctly installed:

```bash
# Check if PyLithics is available
pylithics --help

# Or use the alternative command
pylithics-run --help
```

You should see the help text displaying available options and commands.


## Updating PyLithics

To update to the latest version:

```bash
# Navigate to PyLithics directory
cd Palaeoanalytics

# Pull latest changes
git pull origin main

# Reinstall
pip install . --upgrade
```

## Building Documentation Locally

Documentation tools are installed automatically with PyLithics. To build and view the documentation locally:

```bash
# Serve documentation locally at http://127.0.0.1:8000
pylithics --docs
```

!!! tip "Documentation Tools Included"
    MkDocs and related documentation dependencies are automatically installed with PyLithics, so no additional installation is needed.

## Troubleshooting Installation

### Python Version Issues

If you encounter Python version errors:

```bash
# Check your Python version
python --version

# If needed, install Python 3.7+ using your system's package manager
# macOS (using Homebrew)
brew install python@3.9

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.9

# Windows - download from python.org
```

### macOS-Specific Issues

For macOS users with OS versions below 10.14:

- Consider upgrading your OS to 10.14 or later
- If upgrade isn't possible, you may encounter build issues with some dependencies

### Windows PowerShell Execution Policy

If you get execution policy errors on Windows:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned

# Or for current session only
Set-ExecutionPolicy Unrestricted -Scope Process
```

### Missing Dependencies

If you encounter missing dependency errors:
```bash
# Upgrade pip first
pip install --upgrade pip

# Then reinstall with verbose output
pip install . -v
```

### OpenCV Installation Issues

If OpenCV fails to install:
```bash
# Try installing OpenCV separately first
pip install opencv-python-headless>=4.8.0

# Then install PyLithics
pip install .
```


## Uninstalling

To remove PyLithics:

```bash
# Uninstall PyLithics
pip uninstall pylithics

# Deactivate and remove virtual environment
deactivate
rm -rf palaeo/  # On Windows: rmdir /s palaeo
```

## Next Steps

Now that PyLithics is installed, you're ready to:

1. [Prepare your images](user-guide/image-requirements.md)
2. [Set up metadata](user-guide/metadata-setup.md)
3. [Run your first analysis](user-guide/basic-usage.md)

For any installation issues not covered here, please [check our troubleshooting guide](user-guide/troubleshooting.md) or [open an issue on GitHub](https://github.com/alan-turing-institute/Palaeoanalytics/issues).