
import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver


# --- MATH HELPERS ---

def quat_to_R(q):
    """R(q): rotation matrix from a quaternion [qw qx qy qz]."""
    w, qx, qy, qz = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*w),     2*(qx*qz + qy*w)),
        ca.horzcat(2*(qx*qy + qz*w),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*w)),
        ca.horzcat(2*(qx*qz - qy*w),      2*(qy*qz + qx*w),      1 - 2*(qx**2 + qy**2)))

def quat_mul(a, c):
    """a o c: Hamilton quaternion product, both [w x y z]."""
    return ca.vertcat(a[0]*c[0] - a[1]*c[1] - a[2]*c[2] - a[3]*c[3],
                    a[0]*c[1] + a[1]*c[0] + a[2]*c[3] - a[3]*c[2],
                    a[0]*c[2] - a[1]*c[3] + a[2]*c[0] + a[3]*c[1],
                    a[0]*c[3] + a[1]*c[2] - a[2]*c[1] + a[3]*c[0])

def H(v):
    return ca.vertcat(0, v)

def R_X(a):
    return ca.vertcat(ca.horzcat(1,0,0),
                    ca.horzcat(0, ca.cos(a), -ca.sin(a)),
                    ca.horzcat(0, ca.sin(a), ca.cos(a)))

def R_Z(a):
    return ca.DM([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

# --- MAIN MODEL FUNCTION --- 

def export_model(params: dict) -> AcadosModel:
    """params: dict loaded from omnidrone_params.yaml"""
 
    model = AcadosModel()
    model.name = params["mav_name"] + "_omnidrone"
 
    # --- CONSTANTS (from YAML) ---
    Np = params["num_rotors"]
    mass = params["mass"]
    Ixx = params["inertia"]["Ixx"]
    Iyy = params["inertia"]["Iyy"]
    Izz = params["inertia"]["Izz"]
    t_servo = params["t_servo"]
    g = params["gravity"]
    arm_theta = np.deg2rad(params["arm_theta_deg"])
    rotor_dist = params["rotor_dist"]
    b = params["spin_dir"]
    kq_kt = params["kq_kt"]
 
    # --- PARAMETERS (disturbances + reference quaternion) ---
    W_f_d   = ca.SX.sym('W_f_d', 3)     # disturbance force, world frame
    G_tau_d = ca.SX.sym('G_tau_d', 3)   # disturbance torque, body frame
    q_r     = ca.SX.sym('q_r', 4)
    p = ca.vertcat(W_f_d, G_tau_d, q_r)
 
    # --- STATES ---
    W_p     = ca.SX.sym('W_p', 3)       # position, world frame
    W_v     = ca.SX.sym('W_v', 3)       # velocity, world frame
    WG_q    = ca.SX.sym('WG_q', 4)      # quaternion G -> W, [qw qx qy qz]
    G_omega = ca.SX.sym('G_omega', 3)   # angular velocity, body frame
    alpha   = ca.SX.sym('alpha', Np)    # servo angles
    x    = ca.vertcat(W_p, W_v, WG_q, G_omega, alpha)   # 17 states
    xdot = ca.SX.sym('xdot', x.shape[0])
 
    # --- CONTROLS ---
    f_c     = ca.SX.sym('f_c', Np)      # thrust commands [N]
    alpha_c = ca.SX.sym('alpha_c', Np)  # servo angle commands [rad]
    u = ca.vertcat(f_c, alpha_c)        # 8 controls
 
    # --- Resultant wrench (2a-b) ---
    R_f   = [ca.vertcat(0, 0, f_c[i]) for i in range(Np)]
    R_tau = [ca.vertcat(0, 0, -b[i] * f_c[i] * kq_kt) for i in range(Np)]
 
    # --- Rotor frame to adjusted frame via alpha (3) ---
    Ei_Ri_R = [R_X(alpha[i]) for i in range(Np)]
 
    # --- G_f_u and G_tau_u (4a-b) ---
    G_Ei_R = [R_Z(arm_theta[i]) for i in range(Np)]
    G_p_r  = [rotor_dist * ca.DM([np.cos(arm_theta[i]), np.sin(arm_theta[i]), 0]) for i in range(Np)]
 
    G_f_u   = ca.SX.zeros(3)
    G_tau_u = ca.SX.zeros(3)
    for i in range(Np):
        G_f_u   += G_Ei_R[i] @ Ei_Ri_R[i] @ R_f[i]
        G_tau_u += G_Ei_R[i] @ Ei_Ri_R[i] @ R_tau[i] + ca.cross(G_p_r[i], G_Ei_R[i] @ Ei_Ri_R[i] @ R_f[i])
 
    # --- Servo model (5) ---
    alpha_dot = (alpha_c - alpha) / t_servo
 
    # --- Rigid body (6) ---
    W_g = ca.DM([0, 0, -g])
    I_vec = ca.DM([Ixx, Iyy, Izz])      # diagonal inertia as a vector
 
    W_p_dot     = W_v
    W_v_dot     = (quat_to_R(WG_q) @ G_f_u + W_f_d) / mass + W_g
    WG_q_dot    = 0.5 * quat_mul(WG_q, H(G_omega))
    G_omega_dot = (-ca.cross(G_omega, I_vec * G_omega) + G_tau_u + G_tau_d) / I_vec  # elementwise
 
    # --- Explicit / implicit dynamics ---
    f_expl = ca.vertcat(W_p_dot, W_v_dot, WG_q_dot, G_omega_dot, alpha_dot)
    f_impl = xdot - f_expl
 
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = []
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model
 
 
if __name__ == "__main__":
    with open("omnidrone_params.yaml") as fh:
        params = yaml.safe_load(fh)
    model = export_model(params)
    print(model.x.shape[0], model.u.shape[0], model.p.shape[0])   # 17 8 10
 