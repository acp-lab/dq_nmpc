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

python3 dq_nmpc/dq_controller.py $COLCON_WS_DIR/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml

# Creating the folder where we are going to paste the files
mkdir $COLCON_WS_DIR/install/dq_cpp/
mkdir $COLCON_WS_DIR/install/dq_cpp/lib/

cp c_generated_code/libacados_ocp_solver_dq_quadrotor.so $COLCON_WS_DIR/install/dq_cpp/lib/

#echo "Deleting old Files"
rm -rf $COLCON_WS_DIR/src/dq_cpp/c_generated_code
mv -f c_generated_code $COLCON_WS_DIR/src/dq_cpp/
#
#
cd $COLCON_WS_DIR
source $COLCON_WS_DIR/install/setup.bash
colcon build --symlink-install --packages-select dq_cpp
source $COLCON_WS_DIR/install/setup.bash
