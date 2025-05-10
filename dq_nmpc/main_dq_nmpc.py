#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca

# Libraries of dual-quaternions
from dq_nmpc import dualquat_from_pose_casadi
from dq_nmpc import dualquat_trans_casadi, dualquat_quat_casadi, rotation_casadi, rotation_inverse_casadi, dual_velocity_casadi, velocities_from_twist_casadi
from dq_nmpc import error_dual_aux_casadi

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from quadrotor_msgs.msg import TRPYCommand
from quadrotor_msgs.msg import PositionCommand
import time
from dq_nmpc import solver
from geometry_msgs.msg import Wrench
from scipy.spatial.transform import Rotation as R

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
        self.declare_parameter('mass', 1.0)
        self.declare_parameter('gravity', 9.8)
        self.declare_parameter('ixx', 0.00305587)
        self.declare_parameter('iyy', 0.00159695)
        self.declare_parameter('izz', 0.00159687)
        self.declare_parameter('mav_name', 'quadrotor')

        # System gains
        self.declare_parameter('nmpc.Q', [0.0]*14)
        self.declare_parameter('nmpc.Q_e', [0.0]*14)
        self.declare_parameter('nmpc.R', [0.0]*4)
        self.declare_parameter('nmpc.horizon_steps', 0)
        self.declare_parameter('nmpc.horizon_time', 0.0)
        self.declare_parameter('nmpc.ts', 0.0)
        self.declare_parameter('nmpc.ubu', [0.0]*4)
        self.declare_parameter('nmpc.lbu', [0.0]*4)
        self.declare_parameter('nmpc.nx', 0)
        self.declare_parameter('nmpc.nu', 0)
        
        self.declare_parameter('flag_build', True)
        self.flag_build = self.get_parameter('flag_build').value


        # Access parameters
        self.mass = self.get_parameter('mass').value
        self.gravity = self.get_parameter('gravity').value
        self.ixx = self.get_parameter('ixx').value
        self.iyy = self.get_parameter('iyy').value
        self.izz = self.get_parameter('izz').value

        nmpc_params = self.get_parameters_by_prefix('nmpc')
        self.Q = nmpc_params['Q'].value
        self.Q_e = nmpc_params['Q_e'].value
        self.R = nmpc_params['R'].value
        self.ubu = nmpc_params['ubu'].value
        self.lbu = nmpc_params['lbu'].value
        self.horizon_time = nmpc_params['horizon_time'].value
        self.horizon_steps = nmpc_params['horizon_steps'].value
        self.ts = nmpc_params['ts'].value
        self.nx = nmpc_params['nx'].value
        self.nu = nmpc_params['nu'].value

        self.mav_name = self.get_parameter('mav_name').get_parameter_value().string_value

        # Check values
        self.get_logger().info(f'Mass: {self.mass}')
        self.get_logger().info(f'Grravity: {self.gravity}')
        self.get_logger().info(f'Ixx: {self.ixx}')
        self.get_logger().info(f'Iyy: {self.iyy}')
        self.get_logger().info(f'Izz: {self.izz}')

        self.get_logger().info(f'Q matrix: {self.Q}')
        self.get_logger().info(f'Qe matrix: {self.Q_e}')
        self.get_logger().info(f'R matrix: {self.R}')

        self.get_logger().info(f'Ubu matrix: {self.ubu}')
        self.get_logger().info(f'Lbu matrix: {self.lbu}')
        
        self.get_logger().info(f'Horizon Steps: {self.horizon_steps}')
        self.get_logger().info(f'Horizon Time: {self.horizon_time}')
        self.get_logger().info(f'Ts Time: {self.ts}')
        self.get_logger().info(f'Name: {self.mav_name}')
        self.get_logger().info(f'Nx: {self.nx}')
        self.get_logger().info(f'Nu: {self.nu}')

        # === Optional: Consolidate all parameters into a dict ===
        params = {
            'mass': self.mass,
            'gravity': self.gravity,
            'ixx': self.ixx,
            'iyy': self.iyy,
            'izz': self.izz,
            'mav_name': self.mav_name,
            'nmpc': {
                'Q': self.Q,
                'Q_e': self.Q_e,
                'R': self.R,
                'ubu': self.ubu,
                'lbu': self.lbu,
                'horizon_steps': self.horizon_steps,
                'ts': self.ts,
                'horizon_time': self.horizon_time,
                'nx': self.nx,
                'nu': self.nu,
            }
        }

        # Values of the system
        self.g = params['gravity']
        self.mQ = params['mass']

        # Inertia Matrix
        self.Jxx = params['ixx']
        self.Jyy = params['iyy']
        self.Jzz = params['izz']
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]

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

        self.acados_ocp_solver, self.ocp = solver(params, self.flag_build)

        # Define odometry subscriber for the drone
        self.subscriber_ = self.create_subscription(Odometry, "odom", self.callback_get_odometry, 10)

        # Define planner subscriber for the drone
        self.subscriber_planner_ = self.create_subscription(PositionCommand, "position_cmd", self.callback_get_planner, 10)

        # Define odometry publisher for the desired path
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "desired_frame", 10)

        # Definition of the publihser for the desired parth
        self.marker_msg = Marker()
        self.points = None
        self.publisher_ref_trajectory_ = self.create_publisher(Marker, 'desired_path', 10)

        # Definition of the publisher 
        self.control_msg = Wrench()
        self.publisher_control_ = self.create_publisher(Wrench, 'cmd', 10)

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

    def callback_get_planner(self, msg):
        # Empty Vector for classical formulation
        pre_quat = np.array([1.0, 0.0, 0.0, 0.0])
        current_quat = np.zeros((4, ))
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
            #aux_torque = J@w_p[:, k] + np.cross(w[:, k], J@w[:, k])
            self.u_d[1:4, i] = self.J @ self.w_dot_ref[0:3, i] + np.cross(self.x_ref[10:13, i], self.J@self.x_ref[10:13, i])
            #self.u_d[1:4, i] = np.array([k.torque.x, k.torque.y, k.torque.z])
            i = i + 1

        # Send data
        self.send_marker()
        self.send_ref()
        return None

    def callback_get_odometry(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z

        vb = np.array([[vx_b], [vy_b], [vz_b]])
        
        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        # Rotation inertial frame
        rotational = R.from_quat([x[7], x[8], x[9], x[6]])
        rotational_matrix = rotational.as_matrix()
        vx_i = rotational_matrix@vb
    
        # Put values in the vector
        x[3] = vx_i[0, 0]
        x[4] = vx_i[1, 0]
        x[5] = vx_i[2, 0]
        self.x_0 = x
        
        # Compute dual quaternion
        self.dual_1 = dualquat_from_pose(self.x_0[6], self.x_0[7], self.x_0[8],  self.x_0[9], self.x_0[0], self.x_0[1], self.x_0[2])
        # Init linear velocity in the inertial frame and angular velocity in the body frame
        self.angular_linear_1 = np.array([self.x_0[10], self.x_0[11], self.x_0[12], self.x_0[3], self.x_0[4], self.x_0[5]]) # Angular Body linear Inertial
        # Init Dual Twist
        self.dual_twist_1 = dual_twist(self.angular_linear_1, self.dual_1)
        self.X[:, 0] = np.array(ca.vertcat(self.dual_1, self.dual_twist_1)).reshape((14, ))
        return None


    def send_ref(self):
        self.ref_msg.header.frame_id = "world"
        self.ref_msg.header.stamp = self.get_clock().now().to_msg()

        self.ref_msg.pose.pose.position.x = self.x_ref[0, 0]
        self.ref_msg.pose.pose.position.y = self.x_ref[1, 0]
        self.ref_msg.pose.pose.position.z = self.x_ref[2, 0]

        self.ref_msg.pose.pose.orientation.x = self.x_ref[7, 0]
        self.ref_msg.pose.pose.orientation.y = self.x_ref[8, 0]
        self.ref_msg.pose.pose.orientation.z = self.x_ref[9, 0]
        self.ref_msg.pose.pose.orientation.w = self.x_ref[6, 0]

        # Send Message
        self.publisher_ref_.publish(self.ref_msg)
        return None 

    def send_cmd(self, dqd, wd, u):
        t_d = get_trans(dqd)
        q_d = get_quat(dqd)
        self.control_msg.force.x = 0.0
        self.control_msg.force.y = 0.0
        self.control_msg.force.z = u[0]
        self.control_msg.torque.x = u[1]
        self.control_msg.torque.y = u[2]
        self.control_msg.torque.z = u[3]
        self.publisher_control_.publish(self.control_msg)
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
        self.u_aux = np.array(self.u_control)
        self.send_cmd(self.X_control[0:8], self.X_control[8:14], self.u_control)
        #self.get_logger().info(f"Sent control: Thrust={self.u_control[0]:.3f}")
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
