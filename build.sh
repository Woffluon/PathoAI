#!/bin/bash
# build.sh - Docker build script with metadata automation

set -e

# Extract version information from git
VERSION=$(git describe --tags --always --dirty)
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)

echo "Building PathoAI Docker image..."
echo "Version: ${VERSION}"
echo "Commit: ${VCS_REF}"
echo "Build Date: ${BUILD_DATE}"

# Ensure Git LFS files are pulled
echo "Checking Git LFS files..."
git lfs pull

# Enable BuildKit for better build performance
export DOCKER_BUILDKIT=1

# Build Docker image with metadata
docker build \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg VCS_REF="${VCS_REF}" \
    --build-arg VERSION="${VERSION}" \
    -t pathoai:${VERSION} \
    -t pathoai:latest \
    .

echo "Build complete!"
echo "Image size:"
docker images pathoai:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
