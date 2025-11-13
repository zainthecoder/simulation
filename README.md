![Orange Colorful Modern Steps to be Successful in Business Infographic (80 x 65 cm) (3)](https://github.com/user-attachments/assets/6b345863-f8d8-40ea-8adf-26fb94a08649)

# PBN Toolkit — Installation and Setup Guide

## Overview

The **Probabilistic Belief Networks (PBN) Toolkit**, developed by Paul Baggenstoss, is a comprehensive software package for constructing discriminative and generative neural networks using PBN methodology and cross-entropy training. This toolkit includes a graphical user interface (`pbntk.py`) and a complete set of supporting libraries for network development and analysis.

This document provides detailed installation and configuration instructions for Linux systems using Python 3.9 and the required custom Theano backend.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Package Installation](#package-installation)
  - [Custom Theano Backend](#custom-theano-backend)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Running the Toolkit](#running-the-toolkit)
- [Quick Installation Reference](#quick-installation-reference)
- [References](#references)
- [Contact](#contact)

---

## System Requirements

- **Operating System:** Linux (recommended)
- **Python Version:** 3.9.x
- **Package Manager:** Miniconda or Anaconda
- **Backend:** https://class-specific.com/pbntk/files/

---

## Installation

### Environment Setup

Create a dedicated Conda environment for the PBN Toolkit:

```bash
conda create -n env python=3.9
conda activate env
```

### Package Installation

Install the required dependencies via Conda:

```bash
conda install -c conda-forge numpy=1.23.5 scipy matplotlib theano future scikit-learn
```
Latest numpy is not compatible with Theano.

### Custom Theano Backend

The PBN Toolkit requires a custom-modified version of Theano (1.0.5-PMB).

**Download the custom Theano package:**

```
http://class-specific.com/theano/theano_1.0.5-pmb.tgz
```

**Create a symbolic link in your Conda environment:**

```bash
ln -s /home/Downloads/theano_1.0.5-pmb/theano_pmb \
      /home/miniconda3/envs/public_project/lib/python3.9/site-packages/theano
```

> **Note:** Adjust the paths according to your local directory structure.

---

## Configuration

### Environment Variables

The following environment variables must be configured before running the toolkit:

```bash
export PBNTK_BACKEND=THEANO
export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float64
export PYTHONPATH=<(full path to current directory, or directory where PBN Toolkit resides)>
```


---

## Project Structure

```
public_project/
│
├── files/                   # Toolkit files
|   ├── pbntk.py             # Main application entry point
│   ├── *.py / *.pyc         # Python modules and compiled files
│   ├── models/              # Network definition files (required)
│   ├── mn398_0.mat          # Dataset file
│   └── ...                  # Additional supporting files
```

### Important Notes

- All model definition files must be located in the `files/models/` directory
- Ensure the `PYTHONPATH` environment variable points to the `files/` directory
- The dataset file `mn398_0.mat` should be present in the `files/` directory

---

## Running the Toolkit

After completing the installation and configuration steps, launch the PBN Toolkit GUI:

```bash
python pbntk.py
```

---

## Quick Installation Reference

Here is the complete installation sequence:

```bash
# Create and activate environment
conda create -n env python=3.9
conda activate env

# Install dependencies
conda install -c conda-forge numpy=1.23.5 scipy matplotlib theano future scikit-learn

# Link custom Theano (adjust paths as needed)
ln -s /home/Downloads/theano_1.0.5-pmb/theano_pmb \
      /home/miniconda3/envs/public_project/lib/python3.9/site-packages/theano

# Set environment variables
export PBNTK_BACKEND=THEANO
export THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float64
export PYTHONPATH=<(full path to current directory, or directory where PBN Toolkit resides)>

# Run toolkit
python pbntk.py
```

---

## References

- **PBN Toolkit Official Website:** [https://class-specific.com/pbntk/](https://class-specific.com/pbntk/)
- **Custom Theano Distribution:** [http://class-specific.com/theano/theano_1.0.5-pmb.tgz](http://class-specific.com/theano/theano_1.0.5-pmb.tgz)
https://class-specific.com/pbntk/readme.txt
- **PBN Toolkit Official ReadME:** [https://class-specific.com/pbntk/readme.txt](https://class-specific.com/pbntk/readme.txt)
---

## Contact

For technical support, questions, or feedback regarding the PBN Toolkit:

**Author:** Paul Baggenstoss  
**Email:** p.m.baggenstoss@ieee.org

---

## License

Please refer to the license information included with the original toolkit distribution or contact the author for licensing details.

---
