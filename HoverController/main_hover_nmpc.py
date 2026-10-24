'''
The ROS 2 node. This is the file that actually runs.
Reads parameters (mass, inertia, weights, limits, horizon).
Subscribes to odom, converts the message into your state vector (watching quaternion order and velocity frame).
Runs a timer that sets the current state, solves, and publishes the first input as a Wrench on cmd.
Holds the fixed hover target, so it needs no planner.
Replaces main_dq_nmpc.py for your purposes.
'''

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from scipy.spatial.transform import Rotation as R

from dq_nmpc.hover_ocp import solver
from dq_nmpc import utils

import numpy as np
import casadi as ca

class HoverNmpcNode(Node):
    def __init__(self):
        super().__init__('hover_nmpc')

         # ---- Parameters ----
        self.declare_parameter('config_path', '/home/cwalczak/ws_ros2/src/dq_nmpc/config/mujoco/default/hover_config.yaml')# ABSOLUTE path to hover_config.yaml
        self.declare_parameter('flag_build', True)                 # True = (re)generate and compile the solver
        self.declare_parameter('hover_position', [0.5, 0.5, 1.0])  # where to hold, world frame [m]
        self.declare_parameter('velocity_in_body_frame', True)     # ASSUMPTION: odom linear velocity is body frame
 
        config_path = self.get_parameter('config_path').value
        if not config_path:
            raise RuntimeError("Set the 'config_path' parameter to the absolute path of hover_config.yaml")
        params = utils.yaml_to_dict(config_path)
 
        mass = params['mass']
        gravity = params['gravity']
        steps = params['nmpc']['horizon_steps']
        horizon_time = params['nmpc']['horizon_time']
        ts = params['nmpc'].get('ts', 0.0)
        if ts <= 0.0:
            ts = horizon_time / steps          # fall back to the solver's own step length
        self.ts = ts
        self.N = steps
        self.velocity_in_body_frame = self.get_parameter('velocity_in_body_frame').value


        # --- VERIFY INIT --- 
        self.get_logger().info(f'Loaded {config_path}')
        self.get_logger().info(f'mass={mass}, gravity={gravity}, N={steps}, horizon={horizon_time}s, ts={ts}s')
        


        #---- Build (or load) the solver ----
        # NOTE: the generated code and json are written relative to the current working
        # directory. If flag_build is False, launch from the same directory you built in.
        self.solver, self.ocp = solver(params, self.get_parameter('flag_build').value)

        # --- Hover Target, Constant rfernce ---
        self.u_hover = np.array([mass*gravity, 0.0, 0.0, 0.0])
        hover_pos = np.array(self.get_parameter('hover_position').value, dtype=float)
        self.x_target = np.concatenate([hover_pos,
                                        np.zeros(3),
                                        [1.0, 0.0, 0.0, 0.0],
                                        np.zeros(3)])
        ref = np.concatenate([self.x_target, self.u_hover])

        for k in range(self.N+1):
            self.solver.set(k, 'p', ref)
            self.solver.set(k, 'x', self.x_target)
        for k in range(self.N):
            self.solver.set(k, 'u', self.u_hover)

        # ---- State storage ----
        self.x_current = None          # filled by the odometry callback
        self.first_solve = True

        self.create_subscription(Odometry, 'odom', self.callback_get_odometry, 10)
        self.publisher_control = self.create_publisher(Wrench, 'cmd', 10)
        self.timer = self.create_timer(self.ts, self.control_nmpc)

        self.get_logger().info(f'Hover NMPC ready. Target position: {hover_pos}')

        # ------------------------------------------------------------------
    # Odometry -> your 13-number state  [pos(3), vel_world(3), quat_wxyz(4), omega_body(3)]
    # This is the only place the sim's message format touches your code.
    # ------------------------------------------------------------------
    def callback_get_odometry(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
 
        # ROS orientation is (x, y, z, w); your model uses (w, x, y, z)
        quat_xyzw = np.array([o.x, o.y, o.z, o.w])
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
 
        # Your model keeps velocity in the WORLD frame
        vel = np.array([lin.x, lin.y, lin.z])
        if self.velocity_in_body_frame:
            vel = R.from_quat(quat_xyzw).apply(vel)       # body -> world
 
        omega = np.array([ang.x, ang.y, ang.z])           # ASSUMPTION: body frame
 
        self.x_current = np.concatenate([[p.x, p.y, p.z], vel, quat_wxyz, omega])

        

    
    # --- Control ---

    def control_nmpc(self):
        if self.x_current is None:
            self.get_logger().warn('No odometry received yet; not publishing.', throttle_duration_sec=2.0)
            return
 
        x0 = self.x_current.copy()

        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)

        if self.first_solve:
            for k in range(self.N + 1):
                self.solver.set(k, 'x', x0)
            for k in range(self.N):
                self.solver.set(k, 'u', self.u_hover)
            self.first_solve = False

        status = self.solver.solve()
      
        
        if status == 0:
            u = np.array(self.solver.get(0, 'u')).flatten() #get next control action
        else:# solve failed, hover
            self.get_logger().warn(f'Solver status {status}; sending plain hover thrust instead.',
                                   throttle_duration_sec=1.0)
            u = self.u_hover

        self.send_cmd(u)

        pos_err = np.linalg.norm(x0[0:3] - self.x_target[0:3])
        
        self.get_logger().info(
            f'pos err {pos_err:.3f} m | thrust {u[0]:.2f} N | torque [{u[1]:.4f}, {u[2]:.4f}, {u[3]:.4f}] | current {self.x_current[0:3]}',
            throttle_duration_sec=1.0)

    
    
    def send_cmd(self, u):
        msg = Wrench()
        msg.force.x = 0.0
        msg.force.y = 0.0
        msg.force.z = float(u[0])
        msg.torque.x = float(u[1])
        msg.torque.y = float(u[2])
        msg.torque.z = float(u[3])
        self.publisher_control.publish(msg)








def main(args=None):
    rclpy.init(args=args)
    node = HoverNmpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('HALT: Keyboard Interrupt')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()