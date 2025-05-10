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
from dq_nmpc import compute_flatness_states
from dq_nmpc import error_dual_aux_casadi

from acados_template import AcadosOcpSolver

from ament_index_python.packages import get_package_share_path

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from quadrotor_msgs.msg import TRPYCommand
from quadrotor_msgs.msg import PositionCommand
import time
import os
import sys
from dq_nmpc import utils
from dq_nmpc import solver

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
    def __init__(self):
        super().__init__('DQNMPC_FINAL')
        # Lets define internal variables

        return None
        self.g = params['gravity']
        self.mQ = params['mass']

        # Inertia Matrix
        self.Jxx = params['ixx']
        self.Jyy = params['iyy']
        self.Jzz = params['izz']
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]

        # Desired sample time  self.ts, time where we want to init over trajectory t_initial, time for the trajectory t_trajectory
        # Time to go to the init state t_final
        # Initial time to established a stable connection self.initial

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

        ## Compute desired path based on read Odometry
        self.Q = np.array(params['nmpc']['Q'])
        self.Q_e = np.array(params['nmpc']['Q_e'])
        self.R = np.array(params['nmpc']['R'])
        print("Gain Values")
        print(self.Q)
        print(self.Q_e)
        print(self.R)

        self.acados_ocp_solver, self.ocp = solver(params)

        # Define Publisher odom from simulation
        self.odom_msg = Odometry()
        self.publisher_odom_ = self.create_publisher(Odometry, "odom", 10)

        # Define odometry subscriber for the drone
        self.subscriber_ = self.create_subscription(Odometry, "/quadrotor/odom", self.callback_get_odometry, 10)

        # Define planner subscriber for the drone
        self.subscriber_planner_ = self.create_subscription(PositionCommand, "/quadrotor/position_cmd", self.callback_get_planner, 10)

        # Define odometry publisher for the desired path
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "desired_frame", 10)

        # Definition of the publihser for the desired parth
        self.marker_msg = Marker()
        self.points = None
        self.publisher_ref_trajectory_ = self.create_publisher(Marker, 'desired_path', 10)

        # Definition of the publisher 
        self.trpy_msg = TRPYCommand()
        self.publisher_trpy_ = self.create_publisher(TRPYCommand, '/quadrotor/trpy_cmd', 10)

        # Definition of the prediction time in secs
        self.t_N = params['nmpc']['horizon_time']

        # Definition of the horizon
        self.N_prediction = params['nmpc']['horizon_steps']

        # Sample time
        self.ts = params['nmpc']['ts']

        # Init states formulated as dualquaternions
        self.dual_1 = dualquat_from_pose(self.x_0[6], self.x_0[7], self.x_0[8],  self.x_0[9], self.x_0[0], self.x_0[1], self.x_0[2])

        # Init linear velocity in the inertial frame and angular velocity in the body frame
        self.angular_linear_1 = np.array([self.x_0[10], self.x_0[11], self.x_0[12], self.x_0[3], self.x_0[4], self.x_0[5]]) # Angular Body linear Inertial

        # Init Dual Twist
        self.dual_twist_1 = dual_twist(self.angular_linear_1, self.dual_1)

        # Auxiliar vector where we can to save all the information formulated as dualquaternion
        self.X = np.zeros((14, 1), dtype=np.double)
        self.X[:, 0] = np.array(ca.vertcat(self.dual_1, self.dual_twist_1)).reshape((14, ))

        ## Auxiliar variables for the controller
        self.dual_1_control = dualquat_from_pose(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.angular_linear_1_control = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Angular Body linear Inertial
        self.dual_twist_1_control = dual_twist(self.angular_linear_1_control, self.dual_1_control)
        self.X_control = np.zeros((14, 1), dtype=np.double)
        self.X_control[:, 0] = np.array(ca.vertcat(self.dual_1_control, self.dual_twist_1_control)).reshape((14, ))
        self.u_control = np.zeros((4, 1), dtype=np.double)
        self.u_control[0, 0] = self.g*self.mQ
        

        # Reference signals of the nmpc
        self.x_ref = np.zeros((13, self.N_prediction), dtype=np.double)
        self.u_d = np.zeros((4, self.N_prediction), dtype=np.double)
        self.w_dot_ref = np.zeros((3, self.N_prediction), dtype=np.double)
        self.X_d = np.zeros((14, self.N_prediction), dtype=np.double)

        self.init_marker()

        self.timer = self.create_timer(self.ts, self.control_nmpc)  # 0.01 seconds = 100 Hz
        self.start_time = time.time()


    def send_odometry(self, dqd):
        # Function that send odometry
        t_d = get_trans(dqd)
        q_d = get_quat(dqd)

        self.odom_msg.header.frame_id = "world"

        self.odom_msg.header.stamp = self.get_clock().now().to_msg()

        self.odom_msg.pose.pose.position.x = float(t_d[1, 0])
        self.odom_msg.pose.pose.position.y = float(t_d[2, 0])
        self.odom_msg.pose.pose.position.z = float(t_d[3, 0])

        self.odom_msg.pose.pose.orientation.x = float(q_d[1, 0])
        self.odom_msg.pose.pose.orientation.y = float(q_d[2, 0])
        self.odom_msg.pose.pose.orientation.z = float(q_d[3, 0])
        self.odom_msg.pose.pose.orientation.w = float(q_d[0, 0])

        # Send Messag
        self.publisher_odom_.publish(self.odom_msg)
        return None 

    def callback_get_planner(self, msg):
        # Empty Vector for classical formulation
        pre_quat = np.array([1.0, 0.0, 0.0, 0.0])
        current_quat = np.zeros((4, ))

        # Filter values of quaternions
        #for k in msg.points:
        #    current_quat = np.array([k.quaternion.w, k.quaternion.x, k.quaternion.y, k.quaternion.z])
        #    aux_dot = np.dot(current_quat, pre_quat)
        #    if aux_dot < 0:
        #        k.quaternion.w = -k.quaternion.w
        #        k.quaternion.x = -k.quaternion.x
        #        k.quaternion.y = -k.quaternion.y
        #        k.quaternion.z = -k.quaternion.z
        #        current_quat = - current_quat
        #    else:
        #        k.quaternion.w = k.quaternion.w
        #        k.quaternion.x = k.quaternion.x
        #        k.quaternion.y = k.quaternion.y
        #        k.quaternion.z = k.quaternion.z
        #        current_quat =  current_quat
        #    pre_quat = current_quat
        ## Set up desired states of the stystem
        i = 0
        for k in msg.points:
            # Desired States
            self.x_ref[0:3, i] = np.array([k.position.x, k.position.y, k.position.z])
            self.x_ref[3:6, i] = np.array([k.velocity.x, k.velocity.y, k.velocity.z])
            self.x_ref[6:10, i] = np.array([k.quaternion.w, k.quaternion.x, k.quaternion.y, k.quaternion.z])
            self.x_ref[10:13, i] = np.array([k.angular_velocity.x, k.angular_velocity.y, k.angular_velocity.z])

            ## Desired Orientation
            qw1_d = self.x_ref[6, i]
            qx1_d = self.x_ref[7, i]
            qy1_d = self.x_ref[8, i]
            qz1_d = self.x_ref[9, i]
            tx1_d = self.x_ref[0, i]
            ty1_d = self.x_ref[1, i]
            tz1_d = self.x_ref[2, i]

            ## Desired dualquaternions
            dual_1_d = dualquat_from_pose(qw1_d, qx1_d, qy1_d,  qz1_d, tx1_d, ty1_d, tz1_d)

            ## Linear Velocities Inertial frame
            hxd_d = self.x_ref[3, i]
            hyd_d = self.x_ref[4, i]
            hzd_d = self.x_ref[5, i]

            ## Angular velocites body frame
            wx_d = self.x_ref[10, i]
            wy_d = self.x_ref[11, i]
            wz_d = self.x_ref[12, i]

            # We do not update the velocites as a dualtwist; instead we use just the 
            angular_linear_1_d = np.array([wx_d, wy_d, wz_d, hxd_d, hyd_d, hzd_d]) # Angular Body linear Inertial
            # Init Dual Twist
            dual_twist_1_d = dual_twist(angular_linear_1_d, dual_1_d)

            # Update Reference
            self.X_d[8:14, i] = np.array(dual_twist_1_d).reshape((6, ))
            self.X_d[0:8, i] = np.array(dual_1_d).reshape((8, ))

            # Desrired force
            self.u_d[0, i] = k.force

            # Desired Torques
            self.w_dot_ref[0:3, i] = np.array([k.angular_velocity_dot.x, k.angular_velocity_dot.y, k.angular_velocity_dot.z])
            #self.u_d[1:4, i] = self.J @ self.w_dot_ref[0:3, i] + np.cross(self.x_ref[10:13, i], self.J@self.x_ref[10:13, i])
            self.u_d[1:4, i] = np.array([k.torque.x, k.torque.y, k.torque.z])
            i = i + 1

        # Send data
        self.send_marker()
        return None
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
        
        # Compute dual quaternion
        self.dual_1 = dualquat_from_pose(self.x_0[6], self.x_0[7], self.x_0[8],  self.x_0[9], self.x_0[0], self.x_0[1], self.x_0[2])
        # Init linear velocity in the inertial frame and angular velocity in the body frame
        self.angular_linear_1 = np.array([self.x_0[10], self.x_0[11], self.x_0[12], self.x_0[3], self.x_0[4], self.x_0[5]]) # Angular Body linear Inertial
        # Init Dual Twist
        self.dual_twist_1 = dual_twist(self.angular_linear_1, self.dual_1)
        self.X[:, 0] = np.array(ca.vertcat(self.dual_1, self.dual_twist_1)).reshape((14, ))

        # send dual quaternion message
        self.dual_msg.header.stamp = self.get_clock().now().to_msg()
        self.dual_msg.d_0 = float(self.dual_1[0, 0])
        self.dual_msg.d_1 = float(self.dual_1[1, 0])
        self.dual_msg.d_2 = float(self.dual_1[2, 0])
        self.dual_msg.d_3 = float(self.dual_1[3, 0])
        self.dual_msg.d_4 = float(self.dual_1[4, 0])
        self.dual_msg.d_5 = float(self.dual_1[5, 0])
        self.dual_msg.d_6 = float(self.dual_1[6, 0])
        self.dual_msg.d_7 = float(self.dual_1[7, 0])

        self.dual_msg.twist_0 = float(self.dual_twist_1[0, 0])
        self.dual_msg.twist_1 = float(self.dual_twist_1[1, 0])
        self.dual_msg.twist_2 =  float(self.dual_twist_1[2, 0])
        self.dual_msg.twist_3 =  float(self.dual_twist_1[3, 0])
        self.dual_msg.twist_4 =  float(self.dual_twist_1[4, 0])
        self.dual_msg.twist_5 =  float(self.dual_twist_1[5, 0])

        self.publisher_dual_.publish(self.dual_msg)
        return None


    def send_ref(self, h, q):
        self.ref_msg.header.frame_id = "world"
        self.ref_msg.header.stamp = self.get_clock().now().to_msg()

        self.ref_msg.pose.pose.position.x = h[0]
        self.ref_msg.pose.pose.position.y = h[1]
        self.ref_msg.pose.pose.position.z = h[2]

        self.ref_msg.pose.pose.orientation.x = q[1]
        self.ref_msg.pose.pose.orientation.y = q[2]
        self.ref_msg.pose.pose.orientation.z = q[3]
        self.ref_msg.pose.pose.orientation.w = q[0]

        # Send Message
        self.publisher_ref_.publish(self.ref_msg)
        return None 

    def send_cmd(self, dqd, wd, u):
        t_d = get_trans(dqd)
        q_d = get_quat(dqd)

        self.trpy_msg.header.stamp = self.get_clock().now().to_msg()
        self.trpy_msg.header.frame_id = "world"
        self.trpy_msg.quaternion.x = float(q_d[1, 0])
        self.trpy_msg.quaternion.y = float(q_d[2, 0])
        self.trpy_msg.quaternion.z = float(q_d[3, 0])
        self.trpy_msg.quaternion.w = float(q_d[0, 0])

        self.trpy_msg.angular_velocity.x = float(wd[0])
        self.trpy_msg.angular_velocity.y = float(wd[1])
        self.trpy_msg.angular_velocity.z = float(wd[2])

        self.trpy_msg.thrust = u[0]

        self.trpy_msg.kom[0] = 0.13
        self.trpy_msg.kom[1] = 0.13
        self.trpy_msg.kom[2] = 1.0

        self.trpy_msg.kr[0] = 1.5
        self.trpy_msg.kr[1] = 1.5
        self.trpy_msg.kr[2] = 1.0
        self.trpy_msg.aux.enable_motors = True

        self.publisher_trpy_.publish(self.trpy_msg)
        return None 

    def init_marker(self):
        self.marker_msg.header.frame_id = "world"
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.ns = "trajectory"
        self.marker_msg.id = 0
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        self.marker_msg.pose.orientation.w = 1.0
        self.marker_msg.scale.x = 0.01  # Line width
        self.marker_msg.color.a = 1.0  # Alpha
        self.marker_msg.color.r = 0.0  # Red
        self.marker_msg.color.g = 1.0  # Green
        self.marker_msg.color.b = 0.0  # Blue
        point = Point()
        point.x = self.x_ref[0, 0]
        point.y = self.x_ref[1, 0]
        point.z = self.x_ref[2, 0]
        self.points = [point]
        self.marker_msg.points = self.points
        return None

    def send_marker(self):
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        point = Point()
        point.x = self.x_ref[0, 0]
        point.y = self.x_ref[1, 0]
        point.z = self.x_ref[2, 0]
        self.points.append(point)
        self.marker_msg.points = self.points
        self.publisher_ref_trajectory_.publish(self.marker_msg)
        return None

    def control_nmpc(self):
        # Optimal Control
        self.acados_ocp_solver.set(0, "lbx", self.X[:, 0])
        self.acados_ocp_solver.set(0, "ubx", self.X[:, 0])

        # Desired Trajectory of the system
        for j in range(self.N_prediction):
            yref = self.X_d[:,0 + j]
            uref = self.u_d[:,0 + j]
            aux_ref = np.hstack((yref, uref, self.Q, self.Q_e, self.R))
            self.acados_ocp_solver.set(j, "p", aux_ref)

        self.acados_ocp_solver.set(j + 1, "p", aux_ref)
        # Check Solution since there can be possible errors 
        self.acados_ocp_solver.solve()
        self.X_control = self.acados_ocp_solver.get(1, "x")
        self.u_control = self.acados_ocp_solver.get(0, "u")
        self.send_cmd(self.X_control[0:8], self.X_control[8:14], self.u_control)
        return None 
        

def main(arg=None):
    rclpy.init(args=arg)
    planning_node = DQnmpcNode()
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
    main()
