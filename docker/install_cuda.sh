#!/bin/bash
set -euo pipefail

# Script to install CUDA 12.8 for GR00T dependencies
# This script is called from the Dockerfile when INSTALL_GROOT is true

echo "Installing CUDA 12.8 for GR00T dependencies"

# Source OS release information
. /etc/os-release

# Detect Ubuntu version and set appropriate CUDA repository
case "$ID" in
  ubuntu)
    case "$VERSION_ID" in
      "20.04") cuda_repo="ubuntu2004";;
      "22.04") cuda_repo="ubuntu2204";;
      "24.04") cuda_repo="ubuntu2404";;
      *) echo "Unsupported Ubuntu $VERSION_ID"; exit 1;;
    esac ;;
  *) echo "Unsupported base OS: $ID"; exit 1 ;;
esac

echo "Detected Ubuntu $VERSION_ID, using repository: $cuda_repo"

# Update package lists and install prerequisites
apt-get update
apt-get install -y --no-install-recommends wget gnupg ca-certificates

# Download and install CUDA keyring
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb

# Download and install CUDA repository pin
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-${cuda_repo}.pin
mv cuda-${cuda_repo}.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Update package lists with new CUDA repository
apt-get update

# Install CUDA toolkit 12.8
apt-get install -y --no-install-recommends cuda-toolkit-12-8

# Clean up package cache
apt-get -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.0+PTX

echo "CUDA environment variables set:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  PATH=$PATH"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

echo "CUDA 12.8 installation completed successfully"
