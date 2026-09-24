# Installation Guide

This guide gives the procedure to install PyLithics. PyLithics operates
on macOS, Windows and Linux. Python 3.8 or later is necessary.

## What is necessary

- **Python**: 3.8 or later
- **Operating system**:
  - macOS 10.14 or later
  - Windows 10 or later
  - Linux (Ubuntu 18.04 or later, CentOS 7 or later, or equivalent)
- **Memory**: 4 GB RAM minimum. 8 GB is better for large data sets.
- **Disk**: 500 MB for the installation, plus space for your data

## Step 1: Make sure that Python and Git are installed

### Python

=== "macOS & Linux"

    ```bash
    # Show the Python version. It must be 3.8 or later.
    python3 --version

    # If Python is not installed:
    # macOS (with Homebrew, from https://brew.sh/)
    brew install python@3.11

    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install python3 python3-pip python3-venv

    # CentOS/RHEL
    sudo yum install python3 python3-pip
    ```

=== "Windows"

    ```powershell
    # Show the Python version. It must be 3.8 or later.
    python --version

    # If Python is not installed, get it from https://python.org
    # During the installation, select "Add Python to PATH".
    ```

### Git

=== "macOS & Linux"

    ```bash
    # Show the Git version.
    git --version

    # If Git is not installed:
    # macOS (with Homebrew, from https://brew.sh/)
    brew install git

    # Ubuntu/Debian
    sudo apt-get install git

    # CentOS/RHEL
    sudo yum install git
    ```

=== "Windows"

    ```powershell
    # Show the Git version.
    git --version

    # If Git is not installed, get it from https://git-scm.com/
    ```

!!! warning "Python version"
    Python 3.8 or later is necessary. If your version is older, install
    a new version before you continue.

## Step 2: Make a virtual environment

A virtual environment prevents conflicts with other Python packages.
Use one.

=== "macOS & Linux"

    ```bash
    # Make the virtual environment
    python3 -m venv palaeo

    # Activate the virtual environment
    source palaeo/bin/activate
    ```

=== "Windows"

    ```powershell
    # Make the virtual environment
    python -m venv palaeo

    # Let PowerShell start scripts (administrator rights can be necessary)
    Set-ExecutionPolicy Unrestricted -Scope Process

    # Activate the virtual environment
    .\palaeo\Scripts\activate
    ```

!!! tip "Active virtual environment"
    When the virtual environment is active, the command prompt starts
    with `(palaeo)`.

## Step 3: Clone the repository

Clone the PyLithics repository from GitHub:

```bash
git clone https://github.com/alan-turing-institute/Palaeoanalytics.git
cd Palaeoanalytics
```

### Select a branch

- **Stable release**: the `main` branch has the most recent tagged
  release (v2.0.0). Use it unless you have a reason to use the
  development version.
- **Development**: the `develop` branch has the changes for the next
  release. It can be less stable.

```bash
# The stable release
git checkout main

# The development version
git checkout develop
```

To use the same version for a repeat of an analysis, check out the tag:

```bash
git checkout v2.0.0
```

## Step 4: Install PyLithics

Install PyLithics and its dependencies:

```bash
pip install .
```

This command:

- Installs the PyLithics package
- Installs the dependencies in `requirements.txt`
- Makes the `pylithics` and `pylithics-pages` commands available

To read the identifiers printed on published plates, install the
optional OCR package too:

```bash
pip install ".[ocr]"
```

## Step 5: Make sure that the installation is correct

Type `pylithics` with no arguments:

```bash
pylithics
```

The command shows the **welcome screen**: the PyLithics logo and a
panel with five command patterns:

1. **Quick start** — analyse the sample data
2. **Sample data and dashboard** — analyse the sample data and open the dashboard
3. **Open a previous analysis** — open the dashboard for a previous analysis
4. **Help and documentation** — `pylithics --help` and `pylithics --docs`
5. **GitHub** — the URL of the repository

Copy the command that you want.

### Optional: analyse the sample data

To make sure that the full pipeline operates, start the second command
from the welcome screen:

```bash
pylithics --data_dir pylithics/data --explore
```

This command analyses the sample images and opens the dashboard in
your browser. If there are no errors and the dashboard opens, the
installation is correct.

### Full help

```bash
# All flags and options
pylithics --help
```

## Update PyLithics

PyLithics tells you when a new release is available. Once a day, when
you start `pylithics` or `pylithics-pages`, it asks GitHub for the
latest release. If the release is newer than your installation, and
your clone is on the `main` branch, you see:

```
PyLithics v2.1.0 is available. You have v2.0.0.
Update PyLithics? [y/N]
```

Answer `y`. PyLithics pulls the `main` branch and installs it again.
Then it gives the address of the release notes:

```
PyLithics is updated to v2.1.0. Release notes: https://github.com/alan-turing-institute/Palaeoanalytics/releases/tag/v2.1.0
```

Answer `n`, or press Enter, to continue without the update. You are
asked again the next day.

The check is silent when there is no network. It does not ask in a
script, where nobody can answer. There it gives the command to update
by hand. A clone on another branch is not told.

To switch the check off, set `update_check.enabled: false` in
`config.yaml`, or set the environment variable
`PYLITHICS_NO_UPDATE_CHECK`.

To update by hand:

```bash
# Go to the PyLithics directory
cd Palaeoanalytics

# Get the latest changes
git pull origin main

# Install again
pip install .
```

## Build the documentation on your computer

The documentation tools are installed with PyLithics. To build the
documentation and show it in your browser:

```bash
# The documentation is at http://127.0.0.1:8000
pylithics --docs
```

!!! tip "Documentation tools"
    MkDocs and its dependencies are installed with PyLithics. No other
    installation is necessary.

## Installation problems

### Python version

If you get a Python version error:

```bash
# Show your Python version
python --version

# If necessary, install Python 3.8 or later with your package manager
# macOS (with Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11

# Windows: get Python from python.org
```

### macOS

On macOS versions before 10.14:

- Update macOS to 10.14 or later.
- If an update is not possible, some dependencies possibly do not build.

### Windows PowerShell execution policy

If you get an execution policy error on Windows:

```powershell
# Start PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned

# Or for the current session only
Set-ExecutionPolicy Unrestricted -Scope Process
```

### Missing dependencies

If you get a missing dependency error:

```bash
# Update pip first
pip install --upgrade pip

# Then install again with full output
pip install . -v
```

### OpenCV

If OpenCV does not install:

```bash
# Install OpenCV first
pip install opencv-python-headless>=4.8.0

# Then install PyLithics
pip install .
```

## Remove PyLithics

```bash
# Remove the PyLithics package
pip uninstall pylithics

# Stop and remove the virtual environment
deactivate
rm -rf palaeo/  # On Windows: rmdir /s palaeo
```

## Next steps

PyLithics is installed. Now:

1. [Prepare your images](user-guide/image-requirements.md)
2. [Prepare the metadata](user-guide/metadata-setup.md)
3. [Do your first analysis](user-guide/basic-usage.md)

If you have an installation problem that is not in this guide, see the
[troubleshooting guide](user-guide/troubleshooting.md) or [open an
issue on GitHub](https://github.com/alan-turing-institute/Palaeoanalytics/issues).
