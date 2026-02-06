#!/bin/bash

#!/bin/bash
echo ""
echo "Let's install the NMPC!"
echo "enter your platform_type"
echo "default: race"
echo 'options: race race_S voxl2 raxl2 iris eagle'
echo ""
read platform_type
platform_type=${platform_type:-eagle}
echo 'thank you!'
echo ""

echo "building acados"
cd ..
git clone https://github.com/acados/acados.git
cd acados/
git checkout 37e17d31890ab54e5a855f1fe787fbf2f5d43bdb
git submodule init
git submodule update --recursive
cd ..
git apply dq_nmpc/blasfeo.patch --directory=acados/external/blasfeo
git apply dq_nmpc/hpipm.patch --directory=acados/external/hpipm
mkdir -p acados/build && cd acados/build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
cd ..

echo ""
echo "building acados_template"
pip install -e interfaces/acados_template
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/lib" >> ~/.bashrc
echo "export ACADOS_SOURCE_DIR=$(pwd)" >> ~/.bashrc
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/lib
export ACADOS_SOURCE_DIR=$(pwd)

echo ""
echo "building tera_renderer"
cd ..
git clone https://github.com/acados/tera_renderer
cd tera_renderer
cargo build --verbose --release
cp target/release/t_renderer $ACADOS_SOURCE_DIR/bin/t_renderer
cd ..

echo ""
echo "build acados controller"
cd dq_nmpc
python3 dq_nmpc/dq_controller.py ../acp-autonomy-stack/config/$platform_type/default/dq_control.yaml
source ~/.bashrc

echo ""
echo "build arpl_nmpc"
cd ../..
source install/setup.bash
colcon build

echo ""
echo "copying acados lib in workspace"
cp src/arpl_nmpc/c_generated_code/libacados_ocp_solver_quadrotor.so install/arpl_nmpc/lib/

############################################################################################
python3 dq_nmpc/dq_controller.py /home/eagle10/ws/acp_ws/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml

# Creating the folder where we are going to paste the files
mkdir /home/eagle10/ws/acp_ws/install/dq_cpp/
mkdir /home/eagle10/ws/acp_ws/install/dq_cpp/lib

cp c_generated_code/libacados_ocp_solver_quadrotor.so /home/eagle10/ws/acp_ws/install/dq_cpp/lib

echo "Deleting old Files"
rm -rf /home/eagle10/ws/acp_ws/src/dq_cpp/c_generated_code
mv -f c_generated_code /home/eagle10/ws/acp_ws/src/dq_cpp/


