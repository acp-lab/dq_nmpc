# Dual-Quaternion Model Predictive Control for Quadrotor

This repository provides an implementation of a Model Predictive Control (MPC) strategy based on Dual Quaternions (DQ) for quadrotor systems.

The controller leverages the Acados optimization framework for fast and efficient solving of optimal control problems, and the MuJoCo physics engine to simulate realistic quadrotor dynamics in a 3D environment. Dual quaternions are used to represent pose in SE(3) compactly and without singularities, making the approach well-suited for aerial robotics applications.

## Dependencies

### Python Packages

Install required Python dependencies:

```bash
pip install --no-cache-dir pyyaml pynput casadi osqp
````

### Acados Installation

To install Acados and its dependencies, execute the following:

```bash
git clone https://github.com/acados/acados.git
cd acados
git checkout 37e17d31890ab54e5a855f1fe787fbf2f5d43bdb
git submodule update --init --recursive
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON -DACADOS_INSTALL_DIR="${HOME}/acados" ..
make install -j$(nproc)
cd ${HOME}/acados 
make shared_library 
make examples_c
pip install -e ${HOME}/acados/interfaces/acados_template
```

#### Build Tera Renderer (for Acados templating)

```bash
git clone https://github.com/acados/tera_renderer
cd tera_renderer
curl https://sh.rustup.rs -sSf | sh -s -- -y
/bin/bash -c "source ${HOME}/.cargo/env && cargo build --verbose --release"
cp ${HOME}/tera_renderer/target/release/t_renderer ${HOME}/acados/bin/t_renderer
```

#### Add Environment Variables and Aliases

Append the following to your `~/.bashrc`:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/acados/lib' >> ~/.bashrc
echo 'export ACADOS_SOURCE_DIR=${HOME}/acados' >> ~/.bashrc
echo 'source ${HOME}/.cargo/env' >> ~/.bashrc
echo 'export WS=/home/ros2_ws' >> ~/.bashrc
```

Apply the changes:

```bash
source ~/.bashrc
```

## Custom Messages

This project depends on custom ROS 2 message types. Clone the message package into your workspace:

```bash
cd ~/ros2_ws/src  
git clone https://github.com/acp-lab/quadrotor_msgs.git
```

Build the workspace:

```bash
cd ~/ros2_ws
colcon build --symlink-install  
source install/setup.bash
```

## Installation of the Dual-Quaternion MPC Controller

Clone this repository into your ROS 2 workspace:

```bash
cd ~/ros2_ws/src  
git clone https://github.com/acp-lab/dq_nmpc.git
```

Build the workspace:

```bash
cd ~/ros2_ws
colcon build --symlink-install  
source install/setup.bash
```

## Simulation Environment

The simulation environment used for testing and validating the controller is available in the following repository:

Quadrotor Simulator in MuJoCo: [https://github.com/acp-lab/quadrotor\_simulator\_mujoco.git](https://github.com/acp-lab/quadrotor_simulator_mujoco.git)

**Important:** Ensure the simulator is installed and running before launching the controller. The MPC relies on published simulation data, such as odometry.

## Usage

To launch the simulation, planner, and controller together, run:

```bash
ros2 launch dq_nmpc controller_dq.launch.py
```

## Available ROS 2 Topics

| Topic Name      | Message Type              | Description                                  |
| --------------- | ------------------------- | -------------------------------------------- |
| /quadrotor/cmd  | geometry\_msgs/msg/Wrench | Control input computed by the DQ-MPC         |
| /quadrotor/odom | nav\_msgs/msg/Odometry    | Full quadrotor state estimate from simulator |

```
```
