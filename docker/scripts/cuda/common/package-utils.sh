#!/bin/bash
# shared package management utilities for multi-os support
#
# Required environment variables:
# - CUDA_MAJOR: CUDA major version (e.g., 12)
# - CUDA_MINOR: CUDA minor version (e.g., 9)
# - PYTHON_VERSION: Python version (e.g., 3.12)
# - TARGETPLATFORM: Target platform (e.g., linux/amd64, linux/arm64) - used for local RPM installation
# Note: Using local CentOS 9 Stream RPMs instead of RHEL subscription for required packages

# Assumes rhel check in consuming script
# Note: Using local CentOS RPMs instead of RHEL subscription
ensure_registered() {
  echo "Skipping RHEL registration - using local CentOS RPMs"
}

# Assumes rhel check in consuming script
ensure_unregistered() {
  echo "Skipping RHEL unregistration - using local CentOS RPMs"
  rm -rf /var/cache/dnf/* || true
}

# Install local RPMs from /tmp/packages/rpms directory
# Maps TARGETPLATFORM to rpm arch directory
install_local_rpms() {
    local targetplatform="${TARGETPLATFORM:-linux/amd64}"
    local rpm_arch
    
    case "$targetplatform" in
        linux/amd64)
            rpm_arch="amd64"
            ;;
        linux/arm64)
            rpm_arch="arm64"
            ;;
        *)
            echo "ERROR: Unsupported TARGETPLATFORM: $targetplatform" >&2
            exit 1
            ;;
    esac
    
    local rpm_dir="/tmp/packages/rpms/${rpm_arch}"
    
    if [ ! -d "$rpm_dir" ]; then
        echo "WARNING: RPM directory not found: $rpm_dir" >&2
        return 0
    fi
    
    local rpm_count
    rpm_count=$(find "$rpm_dir" -name "*.rpm" 2>/dev/null | wc -l)
    
    if [ "$rpm_count" -eq 0 ]; then
        echo "No RPMs found in $rpm_dir"
        return 0
    fi
    
    echo "Installing local RPMs from $rpm_dir"
    # Use dnf localinstall to handle dependencies
    dnf -q install -y "$rpm_dir"/*.rpm
}

# detect architecture for repo URLs
get_download_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        amd64|x86_64)
            echo "x86_64"
            ;;
        aarch64)
            echo "aarch64"
            ;;
        *)
            echo "ERROR: Unsupported architecture: $arch" >&2
            exit 1
            ;;
    esac
}

# expand environment variables in json string
expand_vars() {
    local json="$1"
    echo "$json" | sed "s/\${PYTHON_VERSION}/${PYTHON_VERSION}/g; \
                        s/\${CUDA_MAJOR}/${CUDA_MAJOR}/g; \
                        s/\${CUDA_MINOR}/${CUDA_MINOR}/g"
}

# find package mappings file (script dir or /tmp)
find_mappings_file() {
    local filename="$1"
    local script_dir="$2"

    if [ -f "${script_dir}/${filename}" ]; then
        echo "${script_dir}/${filename}"
    elif [ -f "/tmp/${filename}" ]; then
        echo "/tmp/${filename}"
    else
        echo "ERROR: ${filename} not found" >&2
        exit 1
    fi
}

# setup ubuntu repos
setup_ubuntu_repos() {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y software-properties-common
    add-apt-repository -y universe
    apt-get update -qq
    # efa uses apt for installing packages rather than apt-get
    apt update -qq
}

# setup rhel repos (EPEL and CUDA)
setup_rhel_repos() {
    local download_arch="$1"

    dnf -q install -y dnf-plugins-core
    dnf -q install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
    dnf config-manager --set-enabled epel
    dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${download_arch}/cuda-rhel9.repo"
}

# update system packages
update_system() {
    local os="$1"

    if [ "$os" = "ubuntu" ]; then
        apt-get update -qq
        apt-get upgrade -y
    elif [ "$os" = "rhel" ]; then
        dnf -q update -y
    fi
}

# install packages
install_packages() {
    local os="$1"
    shift
    local packages=("$@")

    if [ "$os" = "ubuntu" ]; then
        apt-get install -y --no-install-recommends "${packages[@]}"
    elif [ "$os" = "rhel" ]; then
        dnf -q install -y --allowerasing "${packages[@]}"
    fi
}

# cleanup package manager cache
cleanup_packages() {
    local os="$1"

    if [ "$os" = "ubuntu" ]; then
        apt-get clean
        rm -rf /var/lib/apt/lists/*
    elif [ "$os" = "rhel" ]; then
        dnf clean all
    fi
}

# autoremove unused packages
autoremove_packages() {
    local os="$1"

    if [ "$os" = "ubuntu" ]; then
        apt-get autoremove -y
    elif [ "$os" = "rhel" ]; then
        dnf autoremove -y
    fi
}

# load package list from json mappings
# usage: load_packages_from_json <os> <manifest_json> <section_key>
# returns: array of package names for the target os
load_packages_from_json() {
    local os="$1"
    local manifest="$2"
    local section="${3:-rhel_to_ubuntu}"

    local packages=()

    if [ "$os" = "ubuntu" ]; then
        # get ubuntu packages from mappings (skip null values)
        while IFS= read -r pkg; do
            [ -n "$pkg" ] && packages+=("$pkg")
        done < <(echo "$manifest" | jq -r ".${section} | to_entries[] | select(.value != null) | .value")

        # add ubuntu-only packages if they exist
        if echo "$manifest" | jq -e '.ubuntu_only' > /dev/null 2>&1; then
            while IFS= read -r pkg; do
                packages+=("$pkg")
            done < <(echo "$manifest" | jq -r '.ubuntu_only[]')
        fi
    elif [ "$os" = "rhel" ]; then
        # use rhel package names directly from json keys
        while IFS= read -r pkg; do
            packages+=("$pkg")
        done < <(echo "$manifest" | jq -r ".${section} | keys[]")
    fi

    printf '%s\n' "${packages[@]}"
}

# load packages from json file with variable expansion
# usage: load_and_expand_packages <os> <mappings_file>
# returns: package names (one per line) for the target os
load_and_expand_packages() {
    local os="$1"
    local mappings_file="$2"

    local manifest manifest_expanded
    manifest=$(cat "$mappings_file")
    manifest_expanded=$(expand_vars "$manifest")

    load_packages_from_json "$os" "$manifest_expanded"
}

# merge two json package manifests (accelerator overrides common)
# usage: merge_package_manifests <common_json> <accelerator_json>
# returns: merged json manifest
merge_package_manifests() {
    local common="$1"
    local accelerator="$2"

    # use jq to deeply merge the two manifests
    # accelerator packages override common ones with same key
    jq -s '.[0] * .[1]' <(echo "$common") <(echo "$accelerator")
}

# load and merge packages from common + accelerator-specific locations
# usage: load_layered_packages <os> <package_type> <accelerator>
# package_type: "builder-packages.json" or "runtime-packages.json"
# accelerator: "cuda", "xpu", "hpu", etc.
# returns: package names (one per line) for the target os
# if no accelerator is passed only the common manifests will be used
load_layered_packages() {
    local os="$1"
    local package_type="$2"
    local accelerator="$3"

    local common_file="/tmp/packages/common/${package_type}"
    local accelerator_file="/tmp/packages/${accelerator}/${package_type}"

    # check if files exist
    if [ ! -f "$common_file" ]; then
        echo "ERROR: Common package file not found: $common_file" >&2
        exit 1
    fi

    # load common packages
    local common_manifest common_expanded
    common_manifest=$(cat "$common_file")
    common_expanded=$(expand_vars "$common_manifest")

    # load accelerator packages if they exist
    local merged_manifest="$common_expanded"
    if [ -f "$accelerator_file" ]; then
        local accel_manifest accel_expanded
        accel_manifest=$(cat "$accelerator_file")
        accel_expanded=$(expand_vars "$accel_manifest")
        merged_manifest=$(merge_package_manifests "$common_expanded" "$accel_expanded")
    fi

    # extract package list for target OS
    load_packages_from_json "$os" "$merged_manifest"
}
