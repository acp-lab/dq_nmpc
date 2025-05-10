#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
#import matplotlib.pyplot as plt
import casadi as ca

import threading

# Libraries of dual-quaternions
from dq_nmpc import dualquat_from_pose_casadi
from dq_nmpc import dualquat_trans_casadi, dualquat_quat_casadi, rotation_casadi, rotation_inverse_casadi, dual_velocity_casadi, velocities_from_twist_casadi
from ode_acados import compute_flatness_states
from dq_nmpc import error_dual_aux_casadi

from acados_template import AcadosOcpSolver

from ament_index_python.packages import get_package_share_path

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from quadrotor_msgs.msg import TRPYCommand
from quadrotor_msgs.msg import PositionCommand
from quadrotor_msgs.msg import TrajectoryPoint
from geometry_msgs.msg import Point, Vector3, Quaternion
from mujoco_msgs.msg import Dual
import time
import os
import sys
import utils
from dq_controller import solver

# Function to create a dualquaternion, get quaernion and translatation and returns a dualquaternion
dualquat_from_pose = dualquat_from_pose_casadi()

# Function to get the trasnlation from the dualquaternion, input dualquaternion and get a translation expressed as a quaternion [0.0, tx, ty,tz]
get_trans = dualquat_trans_casadi()

# Function to get the quaternion from the dualquaternion, input dualquaternion and get a the orientation quaternions [qw, qx, qy, qz]
get_quat = dualquat_quat_casadi()

# Function that maps linear velocities in the inertial frame and angular velocities in the body frame to both of them in the body frame, this is known as twist using dualquaternions
dual_twist = dual_velocity_casadi()

# Function that maps linear and angular velocites in the body frame to the linear velocity in the inertial frame and the angular velocity still in th body frame
velocity_from_twist = velocities_from_twist_casadi()

# Function that returns a vector from the body frame to the inertial frame
rot = rotation_casadi()

# Function that returns a vector from the inertial frame to the body frame
inverse_rot = rotation_inverse_casadi()

# Function to check for the shorthest path
error_dual_f = error_dual_aux_casadi()

class DQnmpcNode(Node):
    def __init__(self, params):
        super().__init__('DQNMPC_FINAL')
        # Lets define internal variables
        self.g = params['gravity']
        self.mQ = params['mass']

        # Inertia Matrix
        self.Jxx = params['ixx']
        self.Jyy = params['iyy']
        self.Jzz = params['izz']
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]
        self.j = 0
        self.ts = params['nmpc']['ts']
        self.t_N = params['nmpc']['horizon_time']
        self.N_prediction = params['nmpc']['horizon_steps']

        # Times for the path 
        t_inital = 4
        t_trajectory = 60
        t_final = 4
        self.initial = 4.8

        # Auxiliary variable init system
        self.t_aux = np.arange(0, 1 + self.ts, self.ts, dtype=np.double)

        # Initial States dual set zeros
        # Position of the system
        pos_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Linear velocity of the sytem respect to the inertial frame
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Angular velocity respect to the Body frame
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Initial Orientation expressed as quaternionn
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])

        # Auxiliary vector [x, v, q, w], which is used to update the odometry and the states of the system
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))

        # Define odometry subscriber for the drone
        self.subscriber_ = self.create_subscription(Odometry, "/quadrotor/odom", self.callback_get_odometry, 10)

        for k in range(0, self.t_aux.shape[0]):
            tic = time.time()
            while (time.time() - tic <= self.ts):
                pass
        # Compute desired path 
        
        self.hd, self.hd_d, self.hd_dd, self.hd_ddd, self.hd_dddd, self.qd, self.w_d, self.w_d_d, self.f_d, self.M_d, self.t = compute_flatness_states(self.L, self.x_0[0:3], t_inital, t_trajectory, t_final, self.ts, 1.0, (self.initial + 1)*0.5)

        # Define planner subscriber for the drone
        self.planner_msg = PositionCommand()
        self.planner_msg.points = [self.create_trajectory_point(i) for i in range(self.N_prediction)]

        self.publisher_planner_ = self.create_publisher(PositionCommand, "/quadrotor/position_cmd", 10)

        self.timer = self.create_timer(self.ts, self.publish_planner)  # 0.01 seconds = 100 Hz
        self.start_time = time.time()


    def create_trajectory_point(self, index):
        """Creates and returns a TrajectoryPoint with initial data."""
        point = TrajectoryPoint()
        point.position = Point(x=0.0, y=0.0, z=0.0)  # Example initial positions
        point.velocity = Vector3(x=0.0 , y=0.0, z=0.0)
        point.acceleration = Vector3(x=0.0, y=0.0, z=0.0)
        point.jerk = Vector3(x=0.0, y=0.0, z=0.0)
        point.snap = Vector3(x=0.0, y=0.0, z=0.0)
        point.quaternion = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        point.force = 0.0  # Example force
        point.angular_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        point.angular_velocity_dot = Vector3(x=0.0, y=0.0, z=0.0)
        point.torque = Vector3(x=0.0, y=0.0, z=0.0)
        return point

    def callback_get_odometry(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        vx_i = msg.twist.twist.linear.x
        vy_i = msg.twist.twist.linear.y
        vz_i = msg.twist.twist.linear.z
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w
    
        # Put values in the vector
        x[3] = vx_i
        x[4] = vy_i
        x[5] = vz_i
        self.x_0 = x
        return None

    def publish_planner(self):
        # Position aux variable
        self.planner_msg.header.stamp = self.get_clock().now().to_msg()
        for i, point in enumerate(self.planner_msg.points):
            # Positions
            point.position.x = self.hd[0, self.j + i]
            point.position.y = self.hd[1, self.j + i]
            point.position.z = self.hd[2, self.j + i]

            # Velocities
            point.velocity.x = self.hd_d[0, self.j + i]
            point.velocity.y = self.hd_d[1, self.j + i]
            point.velocity.z = self.hd_d[2, self.j + i]

            # Accelerations
            point.acceleration.x = self.hd_dd[0, self.j + i]
            point.acceleration.y = self.hd_dd[1, self.j + i]
            point.acceleration.z = self.hd_dd[2, self.j + i]

            # Jerk
            point.jerk.x = self.hd_ddd[0, self.j + i]
            point.jerk.y = self.hd_ddd[1, self.j + i]
            point.jerk.z = self.hd_ddd[2, self.j + i]

            # Snap
            point.snap.x = self.hd_dddd[0, self.j + i]
            point.snap.y = self.hd_dddd[1, self.j + i]
            point.snap.z = self.hd_dddd[2, self.j + i]
            
            # Force
            point.force =   self.f_d[0, self.j + i]

            # Orientations
            point.quaternion.w = self.qd[0, self.j + i]
            point.quaternion.x = self.qd[1, self.j + i]
            point.quaternion.y = self.qd[2, self.j + i]
            point.quaternion.z = self.qd[3, self.j + i]

            # Torques
            point.torque.x = self.M_d[0, self.j + 1]
            point.torque.y = self.M_d[1, self.j + 1]
            point.torque.z = self.M_d[2, self.j + 1]

            # Angular velocity
            point.angular_velocity.x = self.w_d[0, self.j + i]
            point.angular_velocity.y = self.w_d[1, self.j + i]
            point.angular_velocity.z = self.w_d[2, self.j + i]

            # Angular velocity dot
            point.angular_velocity_dot.x = self.w_d_d[0, self.j + i]
            point.angular_velocity_dot.y = self.w_d_d[1, self.j + i]
            point.angular_velocity_dot.z = self.w_d_d[2, self.j + i]
        self.j = self.j +1
        self.publisher_planner_.publish(self.planner_msg)
        return None

def main(arg, params):
    rclpy.init(args=arg)
    planning_node = DQnmpcNode(params)
    try:
        rclpy.spin(planning_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        planning_node.get_logger().info('Simulation stopped manually.')
        planning_node.destroy_node()
        rclpy.shutdown()
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    path_to_yaml = os.path.abspath(sys.argv[1])
    params = utils.yaml_to_dict(path_to_yaml)
    main(None, params)
