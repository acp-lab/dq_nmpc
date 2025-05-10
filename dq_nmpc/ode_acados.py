import numpy as np
import casadi as ca
import numpy as np
from dq_nmpc import Quaternion
from dq_nmpc import DualQuaternion
from casadi import Function
from casadi import jacobian
from acados_template import AcadosModel
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
from scipy.linalg import block_diag
from scipy import sparse
import osqp

# Sample time symbolic
ts = ca.MX.sym("ts", 1, 1)

# Defining Dual Quaternion informtatio
qw_1 = ca.MX.sym('qw_1', 1, 1)
qx_1 = ca.MX.sym('qx_1', 1, 1)
qy_1 = ca.MX.sym('qy_1', 1, 1)
qz_1 = ca.MX.sym('qz_1', 1, 1)
q_1 = ca.vertcat(qw_1, qx_1, qy_1, qz_1)
dw_1 = ca.MX.sym("dw_1", 1, 1)
dx_1 = ca.MX.sym("dx_1", 1, 1)
dy_1 = ca.MX.sym("dy_1", 1, 1)
dz_1 = ca.MX.sym("dz_1", 1, 1)
d_1 = ca.vertcat(dw_1, dx_1, dy_1, dz_1)

# Creating auxiliar variables
dual_1 = ca.vertcat(qw_1, qx_1, qy_1, qz_1, dw_1,dx_1, dy_1, dz_1)

# Defining Desired Frame
qw_1d = ca.MX.sym('qw_1d', 1, 1)
qx_1d = ca.MX.sym('qx_1d', 1, 1)
qy_1d = ca.MX.sym('qy_1d', 1, 1)
qz_1d = ca.MX.sym('qz_1d', 1, 1)
q_1d = ca.vertcat(qw_1d, qx_1d, qy_1d, qz_1d)

dw_1d = ca.MX.sym("dw_1d", 1, 1)
dx_1d = ca.MX.sym("dx_1d", 1, 1)
dy_1d = ca.MX.sym("dy_1d", 1, 1)
dz_1d = ca.MX.sym("dz_1d", 1, 1)
d_1d = ca.vertcat(dw_1d, dx_1d, dy_1d, dz_1d)

# Symbolic Variables
dual_1d = ca.vertcat(qw_1d, qx_1d, qy_1d, qz_1d, dw_1d, dx_1d, dy_1d, dz_1d)

# Defining the desired Velocity using symbolics
vx_1d = ca.MX.sym("vx_1d", 1, 1)
vy_1d = ca.MX.sym("vy_1d", 1, 1)
vz_1d = ca.MX.sym("vz_1d", 1, 1)

wx_1d = ca.MX.sym("wx_1d", 1, 1)
wy_1d = ca.MX.sym("wy_1d", 1, 1)
wz_1d = ca.MX.sym("wz_1d", 1, 1)

Vd = ca.vertcat(0.0, vx_1d, vy_1d, vz_1d)
Wd = ca.vertcat(0.0, wx_1d, wy_1d, wz_1d)

# Symbolic variables desired velocities
w_1d = ca.vertcat(wx_1d, wy_1d, wz_1d, vx_1d, vy_1d, vz_1d)

# Defining the control gains using symbolic variables
kr1 = ca.MX.sym("kr1", 1, 1)
kr2 = ca.MX.sym("kr2", 1, 1)
kr3 = ca.MX.sym("kr3", 1, 1)

kd1 = ca.MX.sym("kd1", 1, 1)
kd2 = ca.MX.sym("kd2", 1, 1)
kd3 = ca.MX.sym("kd3", 1, 1)

Kr = ca.vertcat(0.0, kr1, kr2, kr3)
Kd = ca.vertcat(0.0, kd1, kd2, kd3)


# Creating states of the current dualquaternion
Q1 = DualQuaternion(q_real= Quaternion(q = q_1), q_dual = Quaternion(q = d_1))

# Creating the desired quaternion
Q1d = DualQuaternion(q_real= Quaternion(q = q_1d), q_dual = Quaternion(q = d_1d))

# Creating the Desired dualquaternion twist
W1d = DualQuaternion(q_real= Quaternion(q = Wd), q_dual= Quaternion(q= Vd))

# Quaternion rotation
def rotation_casadi():
    # Function that enables the rotation of a vector using quaternions

    # Creation of the symbolic variables for the quaternion and the vector
    quat_aux_1 = ca.MX.sym('quat_aux_1', 4, 1)
    vector_aux_1 = ca.MX.sym('vector_aux_1', 3, 1)

    # Defining the pure quaternion based on the vector information
    vector = ca.vertcat(0.0, vector_aux_1)

    # Compute conjugate of the quaternion
    quat = quat_aux_1
    quat_c = ca.vertcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0])

    # v' = q x v x q*
    # Rotation to the inertial frame

    H_plus_q = ca.vertcat(ca.horzcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0]),
                                ca.horzcat(quat[1, 0], quat[0, 0], -quat[3, 0], quat[2, 0]),
                                ca.horzcat(quat[2, 0], quat[3, 0], quat[0, 0], -quat[1, 0]),
                                ca.horzcat(quat[3, 0], -quat[2, 0], quat[1, 0], quat[0, 0]))

    # Computing the first multiplication
    aux_value = H_plus_q@vector

    # Multiplication by the conjugate part
    H_plus_aux = ca.vertcat(ca.horzcat(aux_value[0, 0], -aux_value[1, 0], -aux_value[2, 0], -aux_value[3, 0]),
                                ca.horzcat(aux_value[1, 0], aux_value[0, 0], -aux_value[3, 0], aux_value[2, 0]),
                                ca.horzcat(aux_value[2, 0], aux_value[3, 0], aux_value[0, 0], -aux_value[1, 0]),
                                ca.horzcat(aux_value[3, 0], -aux_value[2, 0], aux_value[1, 0], aux_value[0, 0]))

    # Computing the vector rotate respect the quaternion
    vector_i = H_plus_aux@quat_c

    # Create function
    f_rot =  ca.Function('f_rot', [quat_aux_1, vector_aux_1], [vector_i[1:4, 0]])
    return f_rot


def rotation_inverse_casadi():
    # Creation of the symbolic variables for the quaternion and the vector
    quat_aux_1 = ca.MX.sym('quat_aux_1', 4, 1)
    vector_aux_1 = ca.MX.sym('vector_aux_1', 3, 1)

    # Auxiliary pure quaternion based on the information of the vector
    vector = ca.vertcat(0.0, vector_aux_1)

    # Quaternion
    quat = quat_aux_1

    # Quaternion conjugate
    quat_c = ca.vertcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0])
    # v' = q* x v x q 
    # Rotation to the body Frame

    # QUaternion Multiplication vector form
    H_plus_q_c = ca.vertcat(ca.horzcat(quat_c[0, 0], -quat_c[1, 0], -quat_c[2, 0], -quat_c[3, 0]),
                                ca.horzcat(quat_c[1, 0], quat_c[0, 0], -quat_c[3, 0], quat_c[2, 0]),
                                ca.horzcat(quat_c[2, 0], quat_c[3, 0], quat_c[0, 0], -quat_c[1, 0]),
                                ca.horzcat(quat_c[3, 0], -quat_c[2, 0], quat_c[1, 0], quat_c[0, 0]))

    # First Multiplication
    aux_value = H_plus_q_c@vector

    # Quaternion multiplication second element
    H_plus_aux = ca.vertcat(ca.horzcat(aux_value[0, 0], -aux_value[1, 0], -aux_value[2, 0], -aux_value[3, 0]),
                                ca.horzcat(aux_value[1, 0], aux_value[0, 0], -aux_value[3, 0], aux_value[2, 0]),
                                ca.horzcat(aux_value[2, 0], aux_value[3, 0], aux_value[0, 0], -aux_value[1, 0]),
                                ca.horzcat(aux_value[3, 0], -aux_value[2, 0], aux_value[1, 0], aux_value[0, 0]))

    # Rotated vector repected to the body frame
    vector_b = H_plus_aux@quat

    # Defining function using casadi
    f_rot_inv =  ca.Function('f_rot_inv', [quat_aux_1, vector_aux_1], [vector_b[1:4, 0]])
    return f_rot_inv

# Creating functions which are going to be used later
# f_rotation move a vector from the body frame to the inertial frame
f_rotation = rotation_casadi()
# f_rotation move a vector from the inertial frame to the body frame
f_rotation_inverse = rotation_inverse_casadi()

def rotation(quat_aux_1, vector_aux_1):
    # Function that enables the rotation of a vector using quaternions

    # Defining the pure quaternion based on the vector information
    vector = ca.vertcat(0.0, vector_aux_1)

    # Compute conjugate of the quaternion
    quat = quat_aux_1
    quat_c = ca.vertcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0])

    # v' = q x v x q*
    # Rotation to the inertial frame

    H_plus_q = ca.vertcat(ca.horzcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0]),
                                ca.horzcat(quat[1, 0], quat[0, 0], -quat[3, 0], quat[2, 0]),
                                ca.horzcat(quat[2, 0], quat[3, 0], quat[0, 0], -quat[1, 0]),
                                ca.horzcat(quat[3, 0], -quat[2, 0], quat[1, 0], quat[0, 0]))

    # Computing the first multiplication
    aux_value = H_plus_q@vector

    # Multiplication by the conjugate part
    H_plus_aux = ca.vertcat(ca.horzcat(aux_value[0, 0], -aux_value[1, 0], -aux_value[2, 0], -aux_value[3, 0]),
                                ca.horzcat(aux_value[1, 0], aux_value[0, 0], -aux_value[3, 0], aux_value[2, 0]),
                                ca.horzcat(aux_value[2, 0], aux_value[3, 0], aux_value[0, 0], -aux_value[1, 0]),
                                ca.horzcat(aux_value[3, 0], -aux_value[2, 0], aux_value[1, 0], aux_value[0, 0]))

    # Computing the vector rotate respect the quaternion
    vector_i = H_plus_aux@quat_c

    return vector_i[1:4, 0]

def dual_quat_casadi():
    # Function that obtains the elements of the dual quaternion  real an dual part
    values = Q1.get[:, 0]
    dualquaternion_f = Function('dualquaternion_f', [dual_1], [values])
    return dualquaternion_f

def dualquat_trans_casadi():
    # Functions that computes the translation from a dual quaternion
    values = Q1.get_trans.get[:, 0]
    f_trans = Function('f_trans', [dual_1], [values])
    return f_trans

def dualquat_get_real_casadi():
    # Function that get the real part form the dual quaternon
    values = Q1.get_real.get[:, 0]
    f_real = Function('f_real', [dual_1], [values])
    return f_real

def dualquat_get_dual_casadi():
    # Function that gets the dual part of the dualquaternion
    values = Q1.get_dual.get[:, 0]
    f_dual = Function('f_dual', [dual_1], [values])
    return f_dual

def dualquat_quat_casadi():
    # Function that get the quaternion of the dualquaternion, this elemens is the same as the real part
    values = Q1.get_quat.get[:, 0]
    f_quat = Function('f_quat', [dual_1], [values])
    return f_quat

# Creating Functions
get_real = dualquat_get_real_casadi()
get_dual = dualquat_get_dual_casadi()
get_trans = dualquat_trans_casadi()
get_quat = dualquat_quat_casadi()

# Creation of dualquaternion kinemtics
def quatdot_simple(quat, omega):
    # Functions that computes the differential kinematics based on dualquaternions

    # Split values real and dual
    quat_data = quat[0:4, 0]
    dual_data =  quat[4:8, 0]

    # Auxiliary variable in order to avoid numerical issues
    #norm_r = ca.dot(quat_data, quat_data)
    K_quat = 10
    #norm_d = 2*(quat_data.T@dual_data)

    norm_r = ca.norm_2(quat_data)
    norm_d = 2* ca.dot(quat_data, dual_data)

    quat_error = 1 - norm_r
    dual_error = norm_d

    aux_1 = quat_data * (K_quat*quat_error)
    aux_2 = dual_data * (0*dual_error)

    aux_dual = ca.vertcat(aux_1, aux_2)

    # Creatin aux Variables
    H_r_plus = ca.vertcat(ca.horzcat(quat_data[0, 0], -quat_data[1, 0], -quat_data[2, 0], -quat_data[3, 0]),
                                ca.horzcat(quat_data[1, 0], quat_data[0, 0], -quat_data[3, 0], quat_data[2, 0]),
                                ca.horzcat(quat_data[2, 0], quat_data[3, 0], quat_data[0, 0], -quat_data[1, 0]),
                                ca.horzcat(quat_data[3, 0], -quat_data[2, 0], quat_data[1, 0], quat_data[0, 0]))

    H_d_plus = ca.vertcat(ca.horzcat(dual_data[0, 0], -dual_data[1, 0], -dual_data[2, 0], -dual_data[3, 0]),
                                ca.horzcat(dual_data[1, 0], dual_data[0, 0], -dual_data[3, 0], dual_data[2, 0]),
                                ca.horzcat(dual_data[2, 0], dual_data[3, 0], dual_data[0, 0], -dual_data[1, 0]),
                                ca.horzcat(dual_data[3, 0], -dual_data[2, 0], dual_data[1, 0], dual_data[0, 0]))
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros),
                        ca.horzcat(H_d_plus, H_r_plus))

    # Auxiliar variable veloicities
    omega = ca.vertcat(0.0, omega[0, 0], omega[1, 0], omega[2, 0], 0.0, omega[3, 0], omega[4, 0], omega[5, 0])
    q_dot = (1/2)*(Hplus@ omega) + aux_dual
    return q_dot


def dual_velocity_casadi(w_1d = w_1d, dual_1d = dual_1d):
    # Funtions that computes the twist based on dualquaternions
    twist = ca.vertcat(0.0, w_1d[0:3, 0], 0.0, w_1d[3:6, 0])
    w_aux = get_real(twist)
    w = w_aux[1:4, 0]

    v_aux = get_dual(twist)
    v = v_aux[1:4, 0]

    quat_aux = get_quat(dual_1d)

    real = w
    dual = f_rotation_inverse(quat_aux, v)
    
    velocity = ca.vertcat(real, dual)
    f_velocity = Function('f_velocity', [w_1d, dual_1d], [velocity])
    return f_velocity


def velocities_from_twist_casadi(w_1d = w_1d, dual_1d = dual_1d):
    # Get Real and dual values 
    twist = ca.vertcat(0.0, w_1d[0:3, 0], 0.0, w_1d[3:6, 0])

    w_aux = get_real(twist)
    w = w_aux[1:4, 0]

    dual_aux = get_dual(twist)
    dual = dual_aux[1:4, 0]

    quat_aux = get_quat(dual_1d)

    w_body = w
    v_inertial = f_rotation(quat_aux, dual)

    velocity = ca.vertcat(w_body, v_inertial)
    f_velocity = Function('f_velocity', [w_1d, dual_1d], [velocity])
    return f_velocity

def Ad(qd, v):
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    quat_d_data = qd_conjugate[0:4]
    dual_d_data =  qd_conjugate[4:8]

    wb = v[0:3, 0]
    vb = v[3:6, 0]

    vector = ca.vertcat(0.0, wb, 0.0, vb)

    H_r_plus = ca.vertcat(ca.horzcat(quat_d_data[0], -quat_d_data[1], -quat_d_data[2], -quat_d_data[3]),
                                ca.horzcat(quat_d_data[1], quat_d_data[0], -quat_d_data[3], quat_d_data[2]),
                                ca.horzcat(quat_d_data[2], quat_d_data[3], quat_d_data[0], -quat_d_data[1]),
                                ca.horzcat(quat_d_data[3], -quat_d_data[2], quat_d_data[1], quat_d_data[0]))

    H_d_plus = ca.vertcat(ca.horzcat(dual_d_data[0], -dual_d_data[1], -dual_d_data[2], -dual_d_data[3]),
                                ca.horzcat(dual_d_data[1], dual_d_data[0], -dual_d_data[3], dual_d_data[2]),
                                ca.horzcat(dual_d_data[2], dual_d_data[3], dual_d_data[0], -dual_d_data[1]),
                                ca.horzcat(dual_d_data[3], -dual_d_data[2], dual_d_data[1], dual_d_data[0]))
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros),
                        ca.horzcat(H_d_plus, H_r_plus))

    aux_dual = Hplus @ vector
    quat_aux_data = aux_dual[0:4]
    dual_aux_data = aux_dual[4:8]

    H_r_aux_plus = ca.vertcat(ca.horzcat(quat_aux_data[0], -quat_aux_data[1], -quat_aux_data[2], -quat_aux_data[3]),
                                ca.horzcat(quat_aux_data[1], quat_aux_data[0], -quat_aux_data[3], quat_aux_data[2]),
                                ca.horzcat(quat_aux_data[2], quat_aux_data[3], quat_aux_data[0], -quat_aux_data[1]),
                                ca.horzcat(quat_aux_data[3], -quat_aux_data[2], quat_aux_data[1], quat_aux_data[0]))

    H_d_aux_plus = ca.vertcat(ca.horzcat(dual_aux_data[0], -dual_aux_data[1], -dual_aux_data[2], -dual_aux_data[3]),
                                ca.horzcat(dual_aux_data[1], dual_aux_data[0], -dual_aux_data[3], dual_aux_data[2]),
                                ca.horzcat(dual_aux_data[2], dual_aux_data[3], dual_aux_data[0], -dual_aux_data[1]),
                                ca.horzcat(dual_aux_data[3], -dual_aux_data[2], dual_aux_data[1], dual_aux_data[0]))

    Haux_plus = ca.vertcat(ca.horzcat(H_r_aux_plus, zeros),
                        ca.horzcat(H_d_aux_plus, H_r_aux_plus))

    
    vector_b_dual = Haux_plus@qd
    vector_b = ca.vertcat(vector_b_dual[1:4], vector_b_dual[5:8])
    return vector_b


def error_lie(qd, q):
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    quat_d_data = qd_conjugate[0:4]
    dual_d_data =  qd_conjugate[4:8]

    H_r_plus = ca.vertcat(ca.horzcat(quat_d_data[0], -quat_d_data[1], -quat_d_data[2], -quat_d_data[3]),
                                ca.horzcat(quat_d_data[1], quat_d_data[0], -quat_d_data[3], quat_d_data[2]),
                                ca.horzcat(quat_d_data[2], quat_d_data[3], quat_d_data[0], -quat_d_data[1]),
                                ca.horzcat(quat_d_data[3], -quat_d_data[2], quat_d_data[1], quat_d_data[0]))

    H_d_plus = ca.vertcat(ca.horzcat(dual_d_data[0], -dual_d_data[1], -dual_d_data[2], -dual_d_data[3]),
                                ca.horzcat(dual_d_data[1], dual_d_data[0], -dual_d_data[3], dual_d_data[2]),
                                ca.horzcat(dual_d_data[2], dual_d_data[3], dual_d_data[0], -dual_d_data[1]),
                                ca.horzcat(dual_d_data[3], -dual_d_data[2], dual_d_data[1], dual_d_data[0]))
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros),
                        ca.horzcat(H_d_plus, H_r_plus))

    q_e_aux = Hplus @ q

    q_error = q_e_aux

    q_error_real = q_error[0:4, 0]
    q_error_real_c = ca.vertcat(q_error_real[0, 0], -q_error_real[1, 0], -q_error_real[2, 0], -q_error_real[3, 0])
    q_error_dual = q_error[4:8, 0]

    ## Real Part
    norm = ca.norm_2(q_error_real[1:4] + ca.np.finfo(np.float64).eps)
    angle = ca.atan2(norm, q_error_real[0])

    ## Dual Part
    H_error_dual_plus = ca.vertcat(ca.horzcat(q_error_dual[0, 0], -q_error_dual[1, 0], -q_error_dual[2, 0], -q_error_dual[3, 0]),
                                ca.horzcat(q_error_dual[1, 0], q_error_dual[0, 0], -q_error_dual[3, 0], q_error_dual[2, 0]),
                                ca.horzcat(q_error_dual[2, 0], q_error_dual[3, 0], q_error_dual[0, 0], -q_error_dual[1, 0]),
                                ca.horzcat(q_error_dual[3, 0], -q_error_dual[2, 0], q_error_dual[1, 0], q_error_dual[0, 0]))

    trans_error = 2 * H_error_dual_plus@q_error_real_c
    # Computing log map
    ln_quaternion = ca.vertcat(0.0,  (1/2)*angle*q_error_real[1, 0]/norm, (1/2)*angle*q_error_real[2, 0]/norm, (1/2)*angle*q_error_real[3, 0]/norm)
    ln_trans = ca.vertcat(0.0, (1/2)*trans_error[1, 0], (1/2)*trans_error[2, 0], (1/2)*trans_error[3, 0])

    q_e_ln = ca.vertcat(ln_quaternion, ln_trans)
    return q_e_ln

def ln_dual(q_error):
    q_error_real = q_error[0:4, 0]
    q_error_real_c = ca.vertcat(q_error_real[0, 0], -q_error_real[1, 0], -q_error_real[2, 0], -q_error_real[3, 0])
    q_error_dual = q_error[4:8, 0]

    ## Real Part
    norm = ca.norm_2(q_error_real[1:4] + ca.np.finfo(np.float64).eps)
    angle = 2*ca.atan2(norm, q_error_real[0])

    ## Dual Part
    H_error_dual_plus = ca.vertcat(ca.horzcat(q_error_dual[0, 0], -q_error_dual[1, 0], -q_error_dual[2, 0], -q_error_dual[3, 0]),
                                ca.horzcat(q_error_dual[1, 0], q_error_dual[0, 0], -q_error_dual[3, 0], q_error_dual[2, 0]),
                                ca.horzcat(q_error_dual[2, 0], q_error_dual[3, 0], q_error_dual[0, 0], -q_error_dual[1, 0]),
                                ca.horzcat(q_error_dual[3, 0], -q_error_dual[2, 0], q_error_dual[1, 0], q_error_dual[0, 0]))

    trans_error = 2 * H_error_dual_plus@q_error_real_c
    # Computing log map
    ln_quaternion = ca.vertcat(0.0, (1/2)*angle*q_error_real[1, 0]/norm, (1/2)*angle*q_error_real[2, 0]/norm, (1/2)*angle*q_error_real[3, 0]/norm)
    ln_trans = ca.vertcat(0.0, (1/2)*trans_error[1, 0], (1/2)*trans_error[2, 0], (1/2)*trans_error[3, 0])

    q_e_ln = ca.vertcat(ln_quaternion, ln_trans)

    return q_e_ln

def conjugate_dual(qd):
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    return qd_conjugate

def error_dual(qd, q):
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    quat_d_data = qd_conjugate[0:4]
    dual_d_data =  qd_conjugate[4:8]

    H_r_plus = ca.vertcat(ca.horzcat(quat_d_data[0], -quat_d_data[1], -quat_d_data[2], -quat_d_data[3]),
                                ca.horzcat(quat_d_data[1], quat_d_data[0], -quat_d_data[3], quat_d_data[2]),
                                ca.horzcat(quat_d_data[2], quat_d_data[3], quat_d_data[0], -quat_d_data[1]),
                                ca.horzcat(quat_d_data[3], -quat_d_data[2], quat_d_data[1], quat_d_data[0]))

    H_d_plus = ca.vertcat(ca.horzcat(dual_d_data[0], -dual_d_data[1], -dual_d_data[2], -dual_d_data[3]),
                                ca.horzcat(dual_d_data[1], dual_d_data[0], -dual_d_data[3], dual_d_data[2]),
                                ca.horzcat(dual_d_data[2], dual_d_data[3], dual_d_data[0], -dual_d_data[1]),
                                ca.horzcat(dual_d_data[3], -dual_d_data[2], dual_d_data[1], dual_d_data[0]))
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros),
                        ca.horzcat(H_d_plus, H_r_plus))

    q_e_aux = Hplus @ q

    q_error = q_e_aux

    return q_error

def error_dual_aux_casadi():
    qd = ca.MX.sym('qd', 8, 1)
    q = ca.MX.sym('q', 8, 1)
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3], qd[4], -qd[5], -qd[6], -qd[7])
    quat_d_data = qd_conjugate[0:4]
    dual_d_data =  qd_conjugate[4:8]

    H_r_plus = ca.vertcat(ca.horzcat(quat_d_data[0], -quat_d_data[1], -quat_d_data[2], -quat_d_data[3]),
                                ca.horzcat(quat_d_data[1], quat_d_data[0], -quat_d_data[3], quat_d_data[2]),
                                ca.horzcat(quat_d_data[2], quat_d_data[3], quat_d_data[0], -quat_d_data[1]),
                                ca.horzcat(quat_d_data[3], -quat_d_data[2], quat_d_data[1], quat_d_data[0]))

    H_d_plus = ca.vertcat(ca.horzcat(dual_d_data[0], -dual_d_data[1], -dual_d_data[2], -dual_d_data[3]),
                                ca.horzcat(dual_d_data[1], dual_d_data[0], -dual_d_data[3], dual_d_data[2]),
                                ca.horzcat(dual_d_data[2], dual_d_data[3], dual_d_data[0], -dual_d_data[1]),
                                ca.horzcat(dual_d_data[3], -dual_d_data[2], dual_d_data[1], dual_d_data[0]))
    zeros = ca.DM.zeros(4, 4)
    Hplus = ca.vertcat(ca.horzcat(H_r_plus, zeros),
                        ca.horzcat(H_d_plus, H_r_plus))

    q_e_aux = Hplus @ q
    
    q_error = q_e_aux

    f_error_dual = Function('f_error_dual', [qd, q], [q_error])
    return f_error_dual

def dual_aceleraction_casadi(dual, omega, u, L):
    # Split Control Actions
    force = u[0, 0]
    torques = u[1:4, 0]

    # System Matrices
    J = ca.DM.zeros(3, 3)
    J[0, 0] = L[1]
    J[1, 1] = L[2]
    J[2, 2] = L[3]
    J_1 = ca.inv(J)
    e3 = ca.DM.zeros(3, 1)
    e3[2, 0] = 1.0
    g = L[4]
    m = L[0]

    # Compute linear and angular velocity from twist velocity
    w = omega[0:3, 0]
    v = omega[3:6, 0]
    p = get_trans(dual)
    q = get_quat(dual)
    p = p[1:4, 0]

    # Compute unforced part
    a = ca.cross(-J_1@w, J@w)
    F_r = a
    F_d = ca.cross(v, w) - g*(f_rotation_inverse(q, e3))

    # Compute forced part
    U_r = J_1@torques
    U_d = (force/m)@e3

    T_r = F_r + U_r
    T_d = F_d + U_d
    T = ca.vertcat(T_r, T_d)

    return T

def export_model(params):
    # Constraints variable
    constraint = ca.types.SimpleNamespace()

    # Parameters Model
    L = [params['mass'], params['ixx'], params['iyy'], params['izz'], params['gravity']]
    print(L)

    # Model section parameters
    model = AcadosModel()
    model.name = params['mav_name']
    model.z = []

    # States of the system
    qw_1d = ca.MX.sym('qw_1d', 1, 1)
    qx_1d = ca.MX.sym('qx_1d', 1, 1)
    qy_1d = ca.MX.sym('qy_1d', 1, 1)
    qz_1d = ca.MX.sym('qz_1d', 1, 1)

    dw_1d = ca.MX.sym("dw_1d", 1, 1)
    dx_1d = ca.MX.sym("dx_1d", 1, 1)
    dy_1d = ca.MX.sym("dy_1d", 1, 1)
    dz_1d = ca.MX.sym("dz_1d", 1, 1)
    
    # Defining the desired Velocity using symbolics
    vx_1d = ca.MX.sym("vx_1d", 1, 1)
    vy_1d = ca.MX.sym("vy_1d", 1, 1)
    vz_1d = ca.MX.sym("vz_1d", 1, 1)

    wx_1d = ca.MX.sym("wx_1d", 1, 1)
    wy_1d = ca.MX.sym("wy_1d", 1, 1)
    wz_1d = ca.MX.sym("wz_1d", 1, 1)


    X = ca.vertcat(qw_1d, qx_1d, qy_1d, qz_1d, dw_1d, dx_1d, dy_1d, dz_1d,
                   wx_1d, wy_1d, wz_1d, vx_1d, vy_1d, vz_1d)
    model.x = X

    # Split States of the system
    twist_1 = X[8:14, 0]
    dualquat_1 = X[0:8, 0]

    # Auxiliary variables implicit function
    qw_1dot = ca.MX.sym('qw_1dot', 1, 1)
    qx_1dot = ca.MX.sym('qx_1dot', 1, 1)
    qy_1dot = ca.MX.sym('qy_1dot', 1, 1)
    qz_1dot = ca.MX.sym('qz_1dot', 1, 1)

    dw_1dot = ca.MX.sym("dw_1dot", 1, 1)
    dx_1dot = ca.MX.sym("dx_1dot", 1, 1)
    dy_1dot = ca.MX.sym("dy_1dot", 1, 1)
    dz_1dot = ca.MX.sym("dz_1dot", 1, 1)

    vx_1dot = ca.MX.sym("vx_1dot", 1, 1)
    vy_1dot = ca.MX.sym("vy_1dot", 1, 1)
    vz_1dot = ca.MX.sym("vz_1dot", 1, 1)

    wx_1dot = ca.MX.sym("wx_1dot", 1, 1)
    wy_1dot = ca.MX.sym("wy_1dot", 1, 1)
    wz_1dot = ca.MX.sym("wz_1dot", 1, 1)

    X_dot = ca.vertcat(qw_1dot, qx_1dot, qy_1dot, qz_1dot, dw_1dot, dx_1dot, dy_1dot, dz_1dot,
                       wx_1dot, wy_1dot, wz_1dot, vx_1dot, vy_1dot, vz_1dot)

    # Control Actions
    F_ref = ca.MX.sym('F_ref')
    tau_1_ref = ca.MX.sym('tau_1_ref')
    tau_2_ref = ca.MX.sym('tau_2_ref')
    tau_3_ref = ca.MX.sym('tau_3_ref')

    u = ca.vertcat(F_ref, tau_1_ref, tau_2_ref, tau_3_ref)
    model.u = u
    
    # System Dynamics
    dualdot = quatdot_simple(dualquat_1, twist_1)
    twistdot = dual_aceleraction_casadi(dualquat_1, twist_1, u, L)
    f_expl = ca.vertcat(dualdot, twistdot)
    f_impl = X_dot - f_expl

    # External parameters
    ref_params = ca.MX.sym('ref_params', params['nmpc']['nx'] + params['nmpc']['nu'], 1)
    cost_params = ca.MX.sym('cost_params', params['nmpc']['nx'] + params['nmpc']['nx'] + params['nmpc']['nu'], 1)
    model.p = ca.vertcat(ref_params, cost_params)

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.xdot = X_dot
    
    # Constraint system
    norm_q = ca.norm_2(get_quat(X[0:8]))
    constraint.expr = ca.vertcat(norm_q)
    constraint.min = 1.0
    constraint.max = 1.0
    return model, get_trans, get_quat, constraint, error_lie, error_dual, ln_dual, Ad, conjugate_dual, rotation

def quadrotorModel(L: list)-> AcadosModel:
    # Dynamics of the quadrotor based on unit quaternions
    # INPUT
    # L                                                          - system parameters(mass, Inertias and gravity)
    # OUTPUT                           
    # model                                                      - Acados model
    model_name = 'quadrotor'
    constraint = ca.types.SimpleNamespace()
    # Defining Desired Frame
    qw_1d = ca.MX.sym('qw_1d', 1, 1)
    qx_1d = ca.MX.sym('qx_1d', 1, 1)
    qy_1d = ca.MX.sym('qy_1d', 1, 1)
    qz_1d = ca.MX.sym('qz_1d', 1, 1)

    dw_1d = ca.MX.sym("dw_1d", 1, 1)
    dx_1d = ca.MX.sym("dx_1d", 1, 1)
    dy_1d = ca.MX.sym("dy_1d", 1, 1)
    dz_1d = ca.MX.sym("dz_1d", 1, 1)
    
    # Defining the desired Velocity using symbolics
    vx_1d = ca.MX.sym("vx_1d", 1, 1)
    vy_1d = ca.MX.sym("vy_1d", 1, 1)
    vz_1d = ca.MX.sym("vz_1d", 1, 1)

    wx_1d = ca.MX.sym("wx_1d", 1, 1)
    wy_1d = ca.MX.sym("wy_1d", 1, 1)
    wz_1d = ca.MX.sym("wz_1d", 1, 1)


    X = ca.vertcat(qw_1d, qx_1d, qy_1d, qz_1d, dw_1d, dx_1d, dy_1d, dz_1d,
                   wx_1d, wy_1d, wz_1d, vx_1d, vy_1d, vz_1d)

    # Split States of the system
    twist_1 = X[8:14, 0]
    dualquat_1 = X[0:8, 0]

    # Auxiliary variables implicit function
    qw_1dot = ca.MX.sym('qw_1dot', 1, 1)
    qx_1dot = ca.MX.sym('qx_1dot', 1, 1)
    qy_1dot = ca.MX.sym('qy_1dot', 1, 1)
    qz_1dot = ca.MX.sym('qz_1dot', 1, 1)

    dw_1dot = ca.MX.sym("dw_1dot", 1, 1)
    dx_1dot = ca.MX.sym("dx_1dot", 1, 1)
    dy_1dot = ca.MX.sym("dy_1dot", 1, 1)
    dz_1dot = ca.MX.sym("dz_1dot", 1, 1)

    vx_1dot = ca.MX.sym("vx_1dot", 1, 1)
    vy_1dot = ca.MX.sym("vy_1dot", 1, 1)
    vz_1dot = ca.MX.sym("vz_1dot", 1, 1)

    wx_1dot = ca.MX.sym("wx_1dot", 1, 1)
    wy_1dot = ca.MX.sym("wy_1dot", 1, 1)
    wz_1dot = ca.MX.sym("wz_1dot", 1, 1)

    X_dot = ca.vertcat(qw_1dot, qx_1dot, qy_1dot, qz_1dot, dw_1dot, dx_1dot, dy_1dot, dz_1dot,
                       wx_1dot, wy_1dot, wz_1dot, vx_1dot, vy_1dot, vz_1dot)

    # Control Actions
    F_ref = ca.MX.sym('F_ref')
    tau_1_ref = ca.MX.sym('tau_1_ref')
    tau_2_ref = ca.MX.sym('tau_2_ref')
    tau_3_ref = ca.MX.sym('tau_3_ref')

    u = ca.vertcat(F_ref, tau_1_ref, tau_2_ref, tau_3_ref)

    dualdot = quatdot_simple(dualquat_1, twist_1)
    twistdot = dual_aceleraction_casadi(dualquat_1, twist_1, u, L)

    norm_q = ca.norm_2(get_quat(X[0:8]))
    dot_real_dual = 2* ca.dot(X[0:4], X[4:8])
    constraint.norm = Function("norm", [X], [norm_q])
    constraint.expr = ca.vertcat(norm_q)
    constraint.min = 1.0
    constraint.max = 1.0
    constraint.min2 = 0.0
    constraint.max2 = 0.0
    # Explicit and implicit functions
    f_expl = ca.vertcat(dualdot, twistdot)
    f_impl = X_dot - f_expl
    p = ca.MX.sym('p', 18, 1)

    # Algebraic variables
    z = []

    # Dynamics
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = X_dot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    return model, get_trans, get_quat, constraint, error_lie, error_dual, ln_dual, Ad, conjugate_dual, rotation

def noise(x, noise):
    # Get position and quaternion
    dual = x[0:8]
    twist = x[8:14]
    trans = get_trans(dual)
    trans_np = np.array(trans[1:4]).reshape((3, ))
    quat_data = get_quat(dual)

    # Split noise
    noise_position = noise[0:3]

    # Translation part
    trans_noise = trans_np + noise_position
    trans_noise_aux = np.array([0.0, trans_noise[0], trans_noise[1], trans_noise[2]])

    # Rotational part
    noise_quat = noise[3:6]
    squared_norm_delta = noise_quat[0]*noise_quat[0] + noise_quat[1]*noise_quat[1] + noise_quat[2]*noise_quat[2]
    q_delta = np.zeros((4, 1))

    if squared_norm_delta > 0:
        norm_delta = np.sqrt(squared_norm_delta)
        sin_delta_by_delta = np.sin(norm_delta) / norm_delta
        q_delta[0, 0] = np.cos(norm_delta)
        q_delta[1, 0] = sin_delta_by_delta * noise_quat[0]
        q_delta[2, 0] = sin_delta_by_delta * noise_quat[1]
        q_delta[3, 0] = sin_delta_by_delta * noise_quat[2]
    else:
        q_delta[0, 0] = 1.0
        q_delta[1, 0] = 0.0
        q_delta[2, 0] = 0.0
        q_delta[3, 0] = 0.0

    H_r_plus = ca.vertcat(ca.horzcat(quat_data[0, 0], -quat_data[1, 0], -quat_data[2, 0], -quat_data[3, 0]),
                                ca.horzcat(quat_data[1, 0], quat_data[0, 0], -quat_data[3, 0], quat_data[2, 0]),
                                ca.horzcat(quat_data[2, 0], quat_data[3, 0], quat_data[0, 0], -quat_data[1, 0]),
                                ca.horzcat(quat_data[3, 0], -quat_data[2, 0], quat_data[1, 0], quat_data[0, 0]))
    H_r_plus = np.array(H_r_plus)
    quat_noise_aux = H_r_plus@q_delta
    Q1_pose =  DualQuaternion.from_pose(quat = quat_noise_aux, trans = trans_noise_aux)
    values_pose = np.array(Q1_pose.get[:, 0]).reshape((8, ))

    # Twist Noise
    values_twist_aux = noise[6:12]
    values_twist = np.array(twist).reshape((6, )) + values_twist_aux
    x_noise = np.array([values_pose[0], values_pose[1], values_pose[2], values_pose[3],
                        values_pose[4], values_pose[5], values_pose[6], values_pose[7],
                        values_twist[0], values_twist[1], values_twist[2],
                        values_twist[3], values_twist[4], values_twist[5]])
    return x_noise
def cost_quaternion_casadi():
    qd = ca.MX.sym('qd', 4, 1)
    q = ca.MX.sym('q', 4, 1)
    qd_conjugate = ca.vertcat(qd[0, 0], -qd[1, 0], -qd[2, 0], -qd[3, 0])
    quat_d_data = qd_conjugate[0:4, 0]
    quaternion = q[0:4, 0]

    H_r_plus = ca.vertcat(ca.horzcat(quat_d_data[0, 0], -quat_d_data[1, 0], -quat_d_data[2, 0], -quat_d_data[3, 0]),
                                ca.horzcat(quat_d_data[1, 0], quat_d_data[0, 0], -quat_d_data[3, 0], quat_d_data[2, 0]),
                                ca.horzcat(quat_d_data[2, 0], quat_d_data[3, 0], quat_d_data[0, 0], -quat_d_data[1, 0]),
                                ca.horzcat(quat_d_data[3, 0], -quat_d_data[2, 0], quat_d_data[1, 0], quat_d_data[0, 0]))



    q_e_aux = H_r_plus @ quaternion
    
    #q_error = ca.if_else(condition1, expr1, expr2) 
    q_error = q_e_aux

    qw = q_error[0, 0] + ca.np.finfo(np.float64).eps
    angle = 2*ca.acos(qw)
    denominator = ca.sqrt(1 - qw*qw)

    ln_quaternion = ca.vertcat(angle*q_error[1, 0]/denominator, angle*q_error[2, 0]/denominator, angle*q_error[3, 0]/denominator)

    cost = ca.norm_2(ln_quaternion)

    # Sux variable in roder to get a norm
    f_cost = Function('f_cost', [qd, q], [cost])
    return f_cost

def cost_translation_casadi():
    td = ca.MX.sym('td', 4, 1)
    t = ca.MX.sym('t', 4, 1)

    te = td - t
    
    cost = te.T@te
    f_cost = Function('f_cost', [td, t], [cost])
    return f_cost

def skew_matrix(x):
    a1 = x[0]
    a2 = x[1]
    a3 = x[2]
    A = np.array([[0.0, -a3, a2], [a3, 0.0, -a1], [-a2, a1, 0.0]], dtype=np.double)
    return A

def ref_circular_trajectory(t, p, w_c):
    # Compute the desired Trajecotry of the system
    # COmpute Desired Positions
    xd = p * np.cos(w_c*t)
    yd = p * np.sin(w_c*t)
    zd = 0 * np.zeros((t.shape[0]))

    # Compute velocities
    xd_p = - p * w_c * np.sin(w_c * t)
    yd_p =   p * w_c * np.cos(w_c * t)
    zd_p = 0 * np.zeros((t.shape[0]))

    # Compute acceleration
    xd_pp = - p * w_c * w_c * np.cos(w_c * t)
    yd_pp = - p * w_c * w_c * np.sin(w_c * t) 
    zd_pp = 0 * np.zeros((t.shape[0]))

    # Compute jerk
    xd_ppp =  p * w_c * w_c * w_c * np.sin(w_c * t)
    yd_ppp = - p * w_c * w_c * w_c * np.cos(w_c * t) 
    zd_ppp = 0 * np.zeros((t.shape[0]))

    # Compute snap
    xd_pppp = p * w_c * w_c * w_c * w_c * np.cos(w_c * t)
    yd_pppp = p * w_c * w_c * w_c * w_c * np.sin(w_c * t)
    zd_pppp = 0 * np.zeros((t.shape[0]))

    # Compute angular displacement
    theta = 0 * np.zeros((t.shape[0]))

    # Compute angular velocity
    theta_p = 0 * np.zeros((t.shape[0]))
    #theta = np.arctan2(yd_p, xd_p)
    #theta = theta

    # Compute angular velocity
    #theta_p = (1. / ((yd_p / xd_p) ** 2 + 1)) * ((yd_pp * xd_p - yd_p * xd_pp) / xd_p ** 2)
    #theta_p[0] = 0.0

    theta_pp = 0 * np.zeros((theta.shape[0]))

    hd = np.vstack((xd, yd, zd))
    hd_p = np.vstack((xd_p, yd_p, zd_p))
    hd_pp = np.vstack((xd_pp, yd_pp, zd_pp))
    hd_ppp = np.vstack((xd_ppp, yd_ppp, zd_ppp))
    hd_pppp = np.vstack((xd_pppp, yd_pppp, zd_pppp))
    return hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_pp

def trajectory(t, zi, w_c):
    p, theta, p_d, theta_d, p_dd, p_ddd, p_dddd, theta_dd = ref_circular_trajectory(t, zi, w_c)
    a = np.pi/2
    b = 0.05
    r = np.zeros((3, p_d.shape[1]), dtype=np.double)
    r_d = np.zeros((3, p_d.shape[1]), dtype=np.double)
    r_dd = np.zeros((3, p_d.shape[1]), dtype=np.double)
    r_ddd = np.zeros((3, p_d.shape[1]), dtype=np.double)
    r_dddd = np.zeros((3, p_d.shape[1]), dtype=np.double)

    for k in range(0, p_d.shape[1]):
        w = np.array([a*np.sin(b*t[k]), 0.0, 0.0], dtype=np.double)
        w_d = np.array([b*a*np.cos(b*t[k]), 0.0, 0.0], dtype=np.double)
        w_dd = np.array([-b*b*a*np.sin(b*t[k]), 0.0, 0.0], dtype=np.double)
        w_ddd = np.array([-b*b*b*a*np.cos(b*t[k]), 0.0, 0.0], dtype=np.double)
        w_dddd = np.array([b*b*b*b*a*np.sin(b*t[k]), 0.0, 0.0], dtype=np.double)

        # Aux variables first derivative
        skew_w_d = skew_matrix(w_d)
        skew_w_d_2 = skew_matrix(w_d)@skew_matrix(w_d)
        skew_w_d_3 = skew_matrix(w_d)@skew_matrix(w_d)@skew_matrix(w_d)
        skew_w_d_4 = skew_matrix(w_d)@skew_matrix(w_d)@skew_matrix(w_d)@skew_matrix(w_d)

        # Aux second derivative
        skew_w_dd = skew_matrix(w_dd)
        skew_w_dd_2 = skew_matrix(w_dd)@skew_matrix(w_dd)

        # Aux third derivative
        skew_w_ddd = skew_matrix(w_ddd)

        # Aux fourth derivative
        skew_w_dddd = skew_matrix(w_dddd)

        # New Desired reference
        r[:, k] = expm(skew_matrix(w))@p[:, k]
        r_d[:, k] = expm(skew_matrix(w))@(p_d[:, k] + skew_w_d@p[:, k])
        r_dd[:, k] = expm(skew_matrix(w))@(skew_w_d_2@p[:, k] + 2*skew_w_d@p_d[:, k] + p_dd[:, k] + skew_w_dd@p[:, k])
        r_ddd[:, k] = expm(skew_matrix(w))@(p_ddd[:, k] + skew_w_ddd@p[:, k] + 3*skew_w_dd@p_d[:, k] + 3*skew_w_d@p_dd[:, k] + skew_w_d_3@p[:, k] + 3*skew_w_d_2@p_d[:, k] + 3 * skew_w_d@skew_w_dd@p[:, k])
        r_dddd[:, k] = expm(skew_matrix(w))@(p_dddd[:, k] + skew_w_dddd@p[:, k] + 4 * skew_w_ddd@p_d[:, k] + 6*skew_w_dd@p_dd[:, k] + 4 * skew_w_d@p_ddd[:, k] + skew_w_d_4@p[:, k] + 3*skew_w_dd_2@p[:, k] + 4*skew_w_d_3@p_d[:, k] + 6*skew_w_d_2@p_dd[:, k] + 6*skew_w_d_2@skew_w_dd@p[:, k] + 4*skew_w_d@skew_w_ddd@p[:, k] + 12*skew_w_d@skew_w_dd@p_d[:, k])

    x0 = 0 * np.zeros((t.shape[0]))
    y0 = 0 * np.zeros((t.shape[0]))
    z0 = 6 * np.ones((t.shape[0]))

    h0 = np.vstack((x0, y0, z0))
    r = r + h0
    return r, r_d, r_dd, r_ddd, r_dddd, theta, theta_d, theta_dd

def compute_flatness_states(L, x, t_initial, t_trajectory, t_final, sample_time, zi, w_c):

    # Drone Parameters
    m = L[0]
    Jxx = L[1]
    Jyy = L[2]
    Jzz = L[3]
    g = L[4]
    J = np.array([[Jxx, 0.0, 0.0], [0.0, Jyy, 0.0], [0.0, 0.0, Jzz]])

    # Inertial Frame 
    Zw = np.array([[0.0], [0.0], [1.0]])
    Xw = np.array([[1.0], [0.0], [0.0]])
    Yw = np.array([[0.0], [1.0], [0.0]])

    #hd, hd_p, hd_pp, hd_ppp, hd_pppp, theta, theta_p, theta_pp = trajectory(t, zi, w_c)
    hd, hd_p, hd_pp, hd_ppp, hd_pppp, theta, theta_p, theta_pp, t = minimum_snap_final(t_initial, t_trajectory, t_final, sample_time, x, zi, w_c)

    # Empty vector for the internal values
    alpha =  np.zeros((3, hd.shape[1]), dtype=np.double)
    beta =  np.zeros((3, hd.shape[1]), dtype=np.double)

    # Desired Orientation matrix
    Yc = np.zeros((3, hd.shape[1]), dtype=np.double)
    Xc = np.zeros((3, hd.shape[1]), dtype=np.double)
    Zc = np.zeros((3, hd.shape[1]), dtype=np.double)

    # Auxiliary Body frame
    Yb = np.zeros((3, hd.shape[1]), dtype=np.double)
    Xb = np.zeros((3, hd.shape[1]), dtype=np.double)
    Zb = np.zeros((3, hd.shape[1]), dtype=np.double)

    q = np.zeros((4, hd.shape[1]), dtype=np.double)

    f = np.zeros((1, hd.shape[1]), dtype=np.double)
    f_p = np.zeros((1, hd.shape[1]), dtype=np.double)

    # Angular vlocity
    w = np.zeros((3, hd.shape[1]), dtype=np.double)

    # Angular acceleration
    w_p = np.zeros((3, hd.shape[1]), dtype=np.double)
    M = np.zeros((3, hd.shape[1]), dtype=np.double)

    for k in range(0, hd.shape[1]):
        # Auxiliary variables
        alpha[:, k] = m*hd_pp[:, k] + m*g*Zw[:, 0]
        beta[:, k] = m*hd_pp[:, k] + m*g*Zw[:, 0]
        aux = beta[:, k]

        # Components Desired Orientation matrix
        Yc[:, k] = np.array([-np.sin(theta[k]), np.cos(theta[k]), 0])
        Xc[:, k] = np.array([ np.cos(theta[k]), np.sin(theta[k]), 0])
        Zc[:, k] = np.array([0.0, 0.0, 1.0])

        # Body frame that is projected to the desired orientation
        Xb[:, k] = (np.cross(Yc[:, k], alpha[:, k]))/(np.linalg.norm(np.cross(Yc[:, k], alpha[:, k])))
        Yb[:, k] = (np.cross(beta[:, k], Xb[:, k]))/(np.linalg.norm(np.cross(beta[:, k], Xb[:, k])))
        Zb[:, k] = np.cross(Xb[:, k], Yb[:, k])

        R_d = np.array([[Xb[0, k], Yb[0, k], Zb[0, k]], [Xb[1, k], Yb[1, k], Zb[1, k]], [Xb[2, k], Yb[2, k], Zb[2, k]]])
        r_d = R.from_matrix(R_d)
        quad_d_aux = r_d.as_quat()
        q[:, k] = np.array([quad_d_aux[3], quad_d_aux[0], quad_d_aux[1], quad_d_aux[2]])
        if k > 0:
            aux_dot = np.dot(q[:, k], q[:, k-1])
            if aux_dot < 0:
                q[:, k] = -q[:, k]
            else:
                q[:, k] = q[:, k]
        else:
            pass
        q[:, k] = q[:, k]/np.linalg.norm(q[:, k])
        # Compute nominal force of the in the body frame
        f[:, k] = np.dot(Zb[:, k], m*hd_pp[:, k] + m*g*Zw[:, 0])

        
        # Compute angular velocities
        # Elements of the vecto b
        b1 = m*np.dot(Xb[:, k], hd_ppp[:, k])
        b2 = -m*np.dot(Yb[:, k], hd_ppp[:, k])
        b3 = theta_p[k] * np.dot(Xc[:, k], Xb[:, k])

        b = np.array([[b1], [b2], [b3]], dtype=np.double)

        # Elements of the matrix A
        a11 = 0.0
        a12 = f[:, k]
        a13 = 0.0

        a21 = f[:, k]
        a22 = 0.0
        a23 = 0.0
        
        a31 = 0.0
        a32 = -np.dot(Yc[:, k], Zb[:, k])
        a33 = np.linalg.norm(np.cross(Yc[:, k], Zb[:, k]))

        # Inverse Matrix A
        A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], dtype=np.double)
        A_1 = np.linalg.inv(A)

        # Compute nominal angular velocity
        aux_angular_velocity = A_1@b
        w[:, k] = aux_angular_velocity[:, 0]
        wx = w[0, k]
        wy = w[1, k]
        wz = w[2, k]
        
        # Time derivative of the force respect with the body axis
        f_p[:, k] = m*np.dot(Zb[:, k], hd_ppp[:, k])

        chi_1 = theta_pp[k] * np.dot(Xc[:, k], Xb[:, k])
        chi_2 = -2*theta_p[k] * wy * np.dot(Xc[:, k], Zb[:, k])
        chi_3 =  -wy * wx * np.dot(Yc[:, k], Yb[:, k])
        chi_4 =  2* theta_p[k] * wz * np.dot(Xc[:, k], Yb[:, k])
        chi_5 =  -wz*wx*np.dot(Yc[:, k], Zb[:, k])

        chi = chi_1 + chi_2 + chi_3 + chi_4 + chi_5

        # Compute angular accelerations of the system
        B1 = m*np.dot(Xb[:, k], hd_pppp[:, k]) - f[:, k]*wx*wz - 2*f_p[:, k]*wy
        B2 = -m*np.dot(Yb[:, k], hd_pppp[:, k]) -2 * f_p[:, k] * wx + f[:, k]*wy*wz
        B3 = chi

        B = np.array([[B1], [B2], [B3]], dtype=np.double)

        # Computing angular acceleration
        aux_angular_acce = A_1@B
        w_p[:, k] = aux_angular_acce[:, 0]
        aux_torque = J@w_p[:, k] + np.cross(w[:, k], J@w[:, k])
        # Compute torque
        M[:, k] = aux_torque
        # Compute nominal force of the in the body frame
    return hd, hd_p, hd_pp, hd_ppp, hd_pppp, q, w, w_p, f, M, t

def position_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack([1 * np.ones(t.shape), t, t**2, t**3, t**4, t**5, t**6, t**7, t**8, t**9])
    return vector

def velocity_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack([
        0 * np.ones(t.shape), 
        1 * np.ones(t.shape), 
        2 * t, 
        3 * t**2, 
        4 * t**3, 
        5 * t**4, 
        6 * t**5, 
        7 * t**6, 
        8 * t**7, 
        9 * t**8
    ])
    return vector

def acceleration_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack([
        0 * np.ones(t.shape), 
        0 * np.ones(t.shape), 
        2 * np.ones(t.shape), 
        6 * t, 
        12 * t**2, 
        20 * t**3, 
        30 * t**4, 
        42 * t**5, 
        56 * t**6, 
        72 * t**7
    ])
    return vector

def jerk_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack([
        0 * np.ones(t.shape), 
        0 * np.ones(t.shape), 
        0 * np.ones(t.shape), 
        6 * np.ones(t.shape), 
        24 * t, 
        60 * t**2, 
        120 * t**3, 
        210 * t**4, 
        336 * t**5, 
        504 * t**6
    ])
    return vector

def snap_time(t):
    t = np.array(t)  # Ensure t is a NumPy array
    vector = np.vstack([
        0 * np.ones(t.shape), 
        0 * np.ones(t.shape), 
        0 * np.ones(t.shape), 
        0 * np.ones(t.shape), 
        24 * np.ones(t.shape), 
        120 * t, 
        360 * t**2, 
        840 * t**3, 
        1680 * t**4, 
        3024 * t**5
    ])
    return vector

def A(t):
    ## Compute Values For matrix A
    ## Equality position and final velocities, accelerations and jerks
    A_zeros_data = A_zeros()
    A_zeros_aux_data = A_zeros_aux()
    A_t_1 = A_aux(t[0])
    A_t_2 = A_aux(t[1])
    A_t_3 = A_aux(t[2])
    A_t_4 = A_aux(t[3])
    A_init = A_start()
    A_med_velocities = A_med()
    A_pos = position_time(0.0).T

    # Equality velicties accelerations final first segment
    A_vel_eq_1 = velocity_time(t[0]).T
    A_acc_eq_1 = acceleration_time(t[0]).T
    A_jerk_eq_1 = jerk_time(t[0]).T
    A_snap_eq_1 = snap_time(t[0]).T

    # Equality velicties accelerations final second segment
    A_vel_eq_2 = velocity_time(t[1]).T
    A_acc_eq_2 = acceleration_time(t[1]).T
    A_jerk_eq_2 = jerk_time(t[1]).T
    A_snap_eq_2 = snap_time(t[1]).T

    # Equality velicties accelerations init third segment
    A_vel_eq_3 = velocity_time(0).T
    A_acc_eq_3 = acceleration_time(0).T
    A_jerk_eq_3 = jerk_time(0).T
    A_snap_eq_3 = snap_time(0).T

    # Equality velicties accelerations init third segment
    A_vel_eq_4 = velocity_time(0).T
    A_acc_eq_4 = acceleration_time(0).T
    A_jerk_eq_4 = jerk_time(0).T
    A_snap_eq_4 = snap_time(0).T

    A = np.block([
        [A_init, A_zeros_data, A_zeros_data, A_zeros_data],
        [A_t_1, A_med_velocities, A_zeros_data, A_zeros_data],
        [A_zeros_data, A_t_2, A_med_velocities, A_zeros_data],
        [A_zeros_data, A_zeros_data, A_t_3, A_med_velocities],
        [A_zeros_data, A_zeros_data, A_zeros_data, A_t_4],
        [A_zeros_aux_data, A_pos, A_zeros_aux_data, A_zeros_aux_data],
        [A_zeros_aux_data, A_zeros_aux_data, A_pos, A_zeros_aux_data],
        [A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data, A_pos],
        [A_vel_eq_1, A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data],
        [A_acc_eq_1, A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data],
        [A_jerk_eq_1, A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data],
        [A_snap_eq_1, A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data],
        [A_zeros_aux_data, A_vel_eq_2, A_zeros_aux_data, A_zeros_aux_data],
        [A_zeros_aux_data, A_acc_eq_2, A_zeros_aux_data, A_zeros_aux_data],
        [A_zeros_aux_data, A_jerk_eq_2, A_zeros_aux_data, A_zeros_aux_data],
        [A_zeros_aux_data, A_snap_eq_2, A_zeros_aux_data, A_zeros_aux_data],
        [A_zeros_aux_data, A_zeros_aux_data, A_vel_eq_3, A_zeros_aux_data],
        [A_zeros_aux_data, A_zeros_aux_data, A_acc_eq_3, A_zeros_aux_data],
        [A_zeros_aux_data, A_zeros_aux_data, A_jerk_eq_3, A_zeros_aux_data],
        [A_zeros_aux_data, A_zeros_aux_data, A_snap_eq_3, A_zeros_aux_data],
        [A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data, A_vel_eq_4],
        [A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data, A_acc_eq_4],
        [A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data, A_jerk_eq_4],
        [A_zeros_aux_data, A_zeros_aux_data, A_zeros_aux_data, A_snap_eq_4]
    ])
    return A

def B(points, h_init, h_final):
    b_1 = np.array([points[0], 0, 0, 0, 0])
    b_2 = np.array([points[1], 0, 0, 0, 0])
    b_3 = np.array([points[2], 0, 0, 0, 0])
    b_4 = np.array([points[3], 0, 0, 0, 0])
    b_5 = np.array([points[4], 0, 0, 0, 0])
    # Define the vector b_5
    b_6 = np.array([points[1], points[2], points[3]])

    b_first = np.array([h_init[1], h_init[2], h_init[3], h_init[4]]);
    b_second = np.array([h_final[1], h_final[2], h_final[3], h_final[4]]);
    b_third = b_second
    b_fourth = np.array([0.0, 0.0, 0.0, 0.0]);
    b = np.concatenate((b_1, b_2, b_3, b_4, b_5, b_6, b_first, b_second, b_third, b_fourth))
    return b

def H(t):
    H_f_1 = hessian_cost(t[0])
    H_f_2 = hessian_cost(t[1])
    H_f_3 = hessian_cost(t[2])
    H_f_4 = hessian_cost(t[3])
    H_i = hessian_cost(0.0)

    H1 = H_f_1 - H_i
    H2 = H_f_2 - H_i
    H3 = H_f_3 - H_i
    H4 = H_f_4 - H_i

    H = block_diag(H1, H2, H3, H4)
    return H

def quadratic_program(t, waypoints, h_init, h_final):
    A_data = A(t)
    b_data = B(waypoints, h_init, h_final)
    H_data = H(t)
    P = sparse.csc_matrix(H_data)  # Quadratic term
    q = np.zeros((A_data.shape[1]))  # Linear term

    # Equality constraint: A_eq * x = b_eq
    A_eq = sparse.csc_matrix(A_data)  # Equality constraint matrix
    b_eq = np.array(b_data)  # Equality constraint vector

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup the problem in OSQP format
    prob.setup(P, q, A_eq, b_eq, b_eq, verbose=False)

    # Solve the problem
    res = prob.solve()
    return res.x

def A_aux(t_final):
    
    # Construct the Ax matrix
    A = np.vstack([
        position_time(t_final).T,
        velocity_time(t_final).T,
        acceleration_time(t_final).T,
        jerk_time(t_final).T,
        snap_time(t_final).T
    ])
    return A

def A_zeros():
    # Construct the Ax matrix
    A = np.zeros((5, 10), dtype=np.double)
    return A

def A_zeros_aux():
    # Construct the Ax matrix
    A = np.zeros((1, 10), dtype=np.double)
    return A

def A_start():
    # Construct the matrix A
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 24, 0, 0, 0, 0, 0]
    ])
    
    # Return the constructed matrix A
    return A

def A_med():
    # Construct the matrix A
    A = np.array([
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0, -2,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0, -6,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0, -24, 0,  0,  0,  0,  0]
    ])
    
    # Return the constructed matrix A
    return A

def hessian_cost(t):
    H = np.array([
        [0, 0, 0, 0, 0,             0,               0,               0,                0,                0],
        [0, 0, 0, 0, 0,             0,               0,               0,                0,                0],
        [0, 0, 0, 0, 0,             0,               0,               0,                0,                0],
        [0, 0, 0, 0, 0,             0,               0,               0,                0,                0],
        [0, 0, 0, 0, 0,             0,               0,               0,                0,                0],
        [0, 0, 0, 0, 0,     14400*t,     43200*t**2,    100800*t**3,    201600*t**4,     362880*t**5],
        [0, 0, 0, 0, 0,    43200*t**2,   172800*t**3,   453600*t**4,    967680*t**5,    1814400*t**6],
        [0, 0, 0, 0, 0,   100800*t**3,   453600*t**4,  1270080*t**5,   2822400*t**6,    5443200*t**7],
        [0, 0, 0, 0, 0,   201600*t**4,   967680*t**5,  2822400*t**6,   6451200*t**7,   12700800*t**8],
        [0, 0, 0, 0, 0,   362880*t**5,  1814400*t**6,  5443200*t**7,  12700800*t**8,   25401600*t**9]
    ])
    return H

def minimum_snap_final(t_initial, t_trajectory, t_final, sample_time, x, zi, w_c):
    # Reshape states
    x = x.reshape((3, 1))

    ##  Compute array with time
    traj_flight_time = np.array([[t_initial, t_trajectory, t_final, t_final]], dtype=np.double)

    r_init, r_d_init, r_dd_init, r_ddd_init, r_dddd_init, _, _, _ = trajectory(traj_flight_time[:, 0], zi, w_c)
    #r_init, _, r_d_init, _, r_dd_init, r_ddd_init, r_dddd_init, _ = ref_circular_trajectory(traj_flight_time[:, 0], zi, w_c)
    h_init = np.hstack((r_init, r_d_init, r_dd_init, r_ddd_init, r_dddd_init))

    r_final, r_d_final, r_dd_final, r_ddd_final, r_dddd_final, _, _, _ = trajectory(traj_flight_time[:, 0] + traj_flight_time[:, 1], zi, w_c)
    #r_final, _, r_d_final, _, r_dd_final, r_ddd_final, r_dddd_final, _ = ref_circular_trajectory(traj_flight_time[:, 0] + traj_flight_time[:, 1], zi, w_c)
    h_final = np.hstack((r_final, r_d_final, r_dd_final, r_ddd_final, r_dddd_final))

    # Definition waypoints
    waypoints_1 = np.hstack((x, r_init, r_final, x, x))
    traj_size = waypoints_1.shape[1] - 1

    # Number Point over trajectory and polynomials
    number_points = 1/sample_time
    number_polynomial = 9
    number_coeff = number_polynomial + 1

    ## Time trajectory auxiliar variable
    t_trajectory_values  = np.arange(traj_flight_time[0, 0], traj_flight_time[0, 1] + traj_flight_time[0, 0], sample_time)
    r, r_d, r_dd, r_ddd, r_dddd, _, _, _ = trajectory(t_trajectory_values, zi, w_c)
    #r, _, r_d, _, r_dd, r_ddd, r_dddd, _ = ref_circular_trajectory(t_trajectory_values, zi, w_c)

    ## Coeff
    coeff_x = quadratic_program(traj_flight_time[0, :], waypoints_1[0, :], h_init[0, :], h_final[0, :])
    coeff_x = coeff_x.reshape(traj_size, number_coeff)
    coeff_y = quadratic_program(traj_flight_time[0, :], waypoints_1[1, :], h_init[1, :], h_final[1, :])
    coeff_y = coeff_y.reshape(traj_size, number_coeff)
    coeff_z = quadratic_program(traj_flight_time[0, :], waypoints_1[2, :], h_init[2, :], h_final[2, :])
    coeff_z = coeff_z.reshape(traj_size, number_coeff)

    ## Compute Trajectories using Coeff and the predefined Trajectory
    position_x = []
    velocity_x = []
    acceleration_x = []
    jerk_x = []
    snap_x = []

    position_y = []
    velocity_y = []
    acceleration_y = []
    jerk_y = []
    snap_y = []

    position_z = []
    velocity_z = []
    acceleration_z = []
    jerk_z = []
    snap_z = []

    for k in range(0, traj_size):
        plot_time = traj_flight_time[0, k] * number_points
        time_step = traj_flight_time[0, k]/plot_time
        if k != 1:
            for j in range(0, int(plot_time)):
                ## X section
                position_x.append(np.dot(coeff_x[k, :],  position_time(j*time_step))[0])
                velocity_x.append(np.dot(coeff_x[k, :],  velocity_time(j*time_step))[0])
                acceleration_x.append(np.dot(coeff_x[k, :], acceleration_time(j*time_step))[0])
                jerk_x.append(np.dot(coeff_x[k, :], jerk_time(j*time_step))[0])
                snap_x.append(np.dot(coeff_x[k, :], snap_time(j*time_step))[0])

                ## Y section
                position_y.append(np.dot(coeff_y[k, :],  position_time(j*time_step))[0])
                velocity_y.append(np.dot(coeff_y[k, :],  velocity_time(j*time_step))[0])
                acceleration_y.append(np.dot(coeff_y[k, :], acceleration_time(j*time_step))[0])
                jerk_y.append(np.dot(coeff_y[k, :], jerk_time(j*time_step))[0])
                snap_y.append(np.dot(coeff_y[k, :], snap_time(j*time_step))[0])

                ## Z section
                position_z.append(np.dot(coeff_z[k, :],  position_time(j*time_step))[0])
                velocity_z.append(np.dot(coeff_z[k, :],  velocity_time(j*time_step))[0])
                acceleration_z.append(np.dot(coeff_z[k, :], acceleration_time(j*time_step))[0])
                jerk_z.append(np.dot(coeff_z[k, :], jerk_time(j*time_step))[0])
                snap_z.append(np.dot(coeff_z[k, :], snap_time(j*time_step))[0])
        else:
            for j in range(0, t_trajectory_values.shape[0]):
                ## X section
                position_x.append(r[0, j])
                velocity_x.append(r_d[0, j])
                acceleration_x.append(r_dd[0, j])
                jerk_x.append(r_ddd[0, j])
                snap_x.append(r_dddd[0, j])

                ## Y section
                position_y.append(r[1, j])
                velocity_y.append(r_d[1, j])
                acceleration_y.append(r_dd[1, j])
                jerk_y.append(r_ddd[1, j])
                snap_y.append(r_dddd[1, j])

                ## Z section
                position_z.append(r[2, j])
                velocity_z.append(r_d[2, j])
                acceleration_z.append(r_dd[2, j])
                jerk_z.append(r_ddd[2, j])
                snap_z.append(r_dddd[2, j])

    position_x = np.array(position_x)
    position_y = np.array(position_y)
    position_z = np.array(position_z)

    # Convert the lists to numpy arrays
    velocity_x = np.array(velocity_x)
    velocity_y = np.array(velocity_y)
    velocity_z = np.array(velocity_z)

    acceleration_x = np.array(acceleration_x)
    acceleration_y = np.array(acceleration_y)
    acceleration_z = np.array(acceleration_z)

    jerk_x = np.array(jerk_x)
    jerk_y = np.array(jerk_y)
    jerk_z = np.array(jerk_z)

    snap_x = np.array(snap_x)
    snap_y = np.array(snap_y)
    snap_z = np.array(snap_z)

    ## Final Data
    h = np.vstack([position_x.T, position_y.T, position_z.T])
    h_d = np.vstack([velocity_x.T, velocity_y.T, velocity_z.T])
    h_dd = np.vstack([acceleration_x.T, acceleration_y.T, acceleration_z.T])
    h_ddd = np.vstack([jerk_x.T, jerk_y.T, jerk_z.T])
    h_dddd = np.vstack([snap_x.T, snap_y.T, snap_z.T])
    t = np.arange(0, h.shape[1]*sample_time, sample_time);

    ## We can also compute the angle using the trajectories
    theta = np.zeros((h.shape[1]))
    theta_d = np.zeros((h.shape[1]))
    theta_dd = np.zeros((h.shape[1]))
    return h, h_d, h_dd, h_ddd, h_dddd, theta, theta_d, theta_dd, t
