import numpy as np
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from dq_nmpc import Quaternion
from dq_nmpc import DualQuaternion
from casadi import Function
from casadi import jacobian
from acados_template import AcadosModel

# Auxiliar values to create the dual quaternion
qw_1_aux = ca.MX.sym('qw_1_aux', 1, 1)
qx_1_aux = ca.MX.sym('qx_1_aux', 1, 1)
qy_1_aux = ca.MX.sym('qy_1_aux', 1, 1)
qz_1_aux = ca.MX.sym('qz_1_aux', 1, 1)
q_1_aux = ca.vertcat(qw_1_aux, qx_1_aux, qy_1_aux, qz_1_aux)
tx_1_aux = ca.MX.sym("tx_1_aux", 1, 1)
ty_1_aux = ca.MX.sym("ty_1_aux", 1, 1)
tz_1_aux = ca.MX.sym("tz_1_aux", 1, 1)
t_1_aux = ca.vertcat(0.0, tx_1_aux, ty_1_aux, tz_1_aux)

# DualQuaternion from axis and position
Q1_pose =  DualQuaternion.from_pose(quat = q_1_aux, trans = t_1_aux)

def dualquat_from_pose_casadi():
    # Compute the Pose based of the quaternion and the trasnation
    values = Q1_pose.get[:, 0]
    f_pose = Function('f_pose', [qw_1_aux, qx_1_aux, qy_1_aux, qz_1_aux, tx_1_aux, ty_1_aux, tz_1_aux], [values])
    return f_pose