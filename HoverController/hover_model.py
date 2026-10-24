'''
Defines the drone’s physics: the 13-number state (position, velocity, quaternion, angular rate) 
and the 4 inputs (thrust, three torques).
Contains the four motion equations (position, velocity, quaternion, angular rate) 
and returns them in the form acados expects.
Takes mass, inertia, and gravity as inputs so they can match the sim.
'''

import casadi as ca
from acados_template import AcadosModel

def body_z_axis(q):
    """Third column of R(q): the body z axis (thrust direction) in the world frame.
 
    q = [qw, qx, qy, qz]. Used to point the thrust in the right direction.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx**2 + qy**2),
    )

def quat_kinematics(q, omega_b):
    """q_dot = 0.5 * q (x) [0, omega_b]   (paper Eq. 2)."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    wx, wy, wz = omega_b[0], omega_b[1], omega_b[2]
    return 0.5 * ca.vertcat(
        -qx * wx - qy * wy - qz * wz,
         qw * wx + qy * wz - qz * wy,
         qw * wy - qx * wz + qz * wx,
         qw * wz + qx * wy - qy * wx,
    )
 
# Main Acados Model
def export_hover_model(params: dict) -> AcadosModel:
    """
    Build the acados model.
 
    params needs: 'mass', 'ixx', 'iyy', 'izz', 'gravity', 'mav_name'
    (same keys the original node already reads from its parameter file, so the
    values match the sim).
    """

    m = params["mass"]
    J = ca.diag(ca.DM([params["ixx"], params["iyy"], params["izz"]]))
    J_inv = ca.inv(J)
    g_vec = ca.DM([0.0, 0.0, -params["gravity"]])

    model = AcadosModel()
    model.name = params["mav_name"] + "_hover"

     #states
    p = ca.MX.sym("p", 3) #pos
    v = ca.MX.sym("v", 3) #vel
    q = ca.MX.sym("q", 4) #quaternion
    w = ca.MX.sym("w", 3) #angular velocity (body frame)

     #assembled state vector x -------------------------
    x = ca.vertcat(p,v,q,w)

    T = ca.MX.sym("T") #total thrust
    tau = ca.MX.sym("tau", 3) # body torques
    u = ca.vertcat(T, tau)

    x_dot = ca.MX.sym("x_dot", 13)

     # Dynamics ------------------------------------------

    p_dot = v
    v_dot = (T/m) * body_z_axis(q) + g_vec # Eq1

    k_quat = params.get("k_quat", 10.0)
    q_dot = quat_kinematics(q,w) + k_quat * (1-ca.norm_2(q)) * q #Eq2
    w_dot = J_inv @ (tau - ca.cross(w, J @ w)) #Eq3


    # Excplicit form ----------------------------------

    f_expl = ca.vertcat(p_dot, v_dot, q_dot, w_dot)
    f_impl = x_dot - f_expl

    # Runtime param --------------------------------------\
    p_ref = ca.MX.sym("p_ref", 13 + 4)

    # Filling otu Acados Model ----------------------------
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.z = []
    model.p = p_ref
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model




