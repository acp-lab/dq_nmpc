#!/bin/bash

set -e

# Get absolute path of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Workspace root
WS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo ""
echo "Workspace detected:"
echo "$WS_DIR"
echo ""

echo "Let's build the NMPC!"
echo "Enter your platform_type"
echo "default: race"
echo "options: race race2 race_S voxl2 raxl2 iris eagle"
echo ""

read platform_type
platform_type=${platform_type:-race}

echo "Thank you!"
echo ""

CONFIG_FILE="$WS_DIR/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml"

# Run code generation
python3 "$SCRIPT_DIR/dq_nmpc/dq_controller.py" "$CONFIG_FILE"

# Create install directories
mkdir -p "$WS_DIR/install/dq_cpp/lib"

# Copy generated shared library
cp c_generated_code/libacados_ocp_solver_dq_quadrotor.so \
   "$WS_DIR/install/dq_cpp/lib/"

# Remove old generated code
rm -rf "$WS_DIR/src/dq_cpp/c_generated_code"

# Move new generated code
mv -f c_generated_code "$WS_DIR/src/dq_cpp/"

# Build package
cd "$WS_DIR"

source install/setup.bash

colcon build --symlink-install --packages-select dq_cpp

source install/setup.bash

echo ""
echo "NMPC build complete!"
echo ""
