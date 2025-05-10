#!/bin/bash
echo ""
echo "Let's build the NMPC!"
echo "enter your platform_type"
echo "default: mujoco"
echo 'options: mujoco'
echo ""
read platform_type
platform_type=${platform_type:-mujoco}
echo 'thank you!'
echo ""

python3 dq_nmpc/dq_controller.py $WS/src/dq_nmpc/config/$platform_type/default/dq_control.yaml

# Creating the folder where we are going to paste the files
mkdir $WS/install/dq_cpp/
mkdir $WS/install/dq_cpp/lib
cp c_generated_code/libacados_ocp_solver_quadrotor.so $WS/install/dq_cpp/lib

echo "Deleting old Files"
rm -rf $WS/src/dq_cpp/c_generated_code
mv -f c_generated_code $WS/src/dq_cpp/


cd $WS
source ~/.bashrc
colcon build --symlink-install --packages-select dq_cpp
source install/setup.bash