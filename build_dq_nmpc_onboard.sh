#!/bin/bash
echo ""
echo "Let's build the NMPC!"
echo "enter your platform_type"
echo "default: race"
echo 'options: race race2 race_S voxl2 raxl2 iris eagle'
echo ""
read platform_type
platform_type=${platform_type:-race}
echo 'thank you!'
echo ""

python3 dq_nmpc/dq_controller.py /ext3/ws_acp/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml

# Creating the folder where we are going to paste the files
mkdir /ext3/ws_acp/install/dq_cpp/
mkdir /ext3/ws_acp/install/dq_cpp/lib

cp c_generated_code/libacados_ocp_solver_quadrotor.so /ext3/ws_acp/install/dq_cpp/lib

echo "Deleting old Files"
rm -rf /ext3/ws_acp/src/dq_cpp/c_generated_code
mv -f c_generated_code /ext3/ws_acp/src/dq_cpp/


source /ext3/env_setup.sh
cd /ext3/ws_acp/
source install/setup.bash
colcon build --symlink-install --packages-select dq_cpp
source install/setup.bash
