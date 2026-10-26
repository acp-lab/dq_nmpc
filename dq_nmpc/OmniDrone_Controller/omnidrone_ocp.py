'''
Builds the optimal control problem around omnidrone model
Holds the cost (weights on position error, velocity, tilt, angular rate, and input), 
the input bounds, the horizon length and step count, and the solver options.

'''




import os
import sys
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver                                   
from omnidrone_model import export_omnidrone_model, quat_mul
from casadi import Function, MX, vertcat, sin, cos, fabs, DM   
import utils


 
def quat_conj(q):
    """q^-1 for a unit quaternion."""
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])



def solver(params, flag = True):    
    nmpc = params['nmpc']
    N = nmpc['horizon_steps']
    Tf = nmpc['horizon_time']
    Np = params['num_rotors']

    # ---------------- STEP 1: model and problem container ----------------
    model = export_omnidrone_model(params)
    nx = model.x.size()[0]                       # 17
    nu = model.u.size()[0]                       # 8
    npar = model.p.size()[0]                     # 10
 
    ocp = AcadosOcp()
    ocp.model = model
    ocp.code_export_directory = 'c_generated_code_omni'   # new build directory


    # ---------------- STEP 2: horizon ----------------

    ocp.dims.N = N
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
 
    # some variables
    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p
 
    W_p, W_v, WG_q, G_omega, alpha = x[0:3], x[3:6], x[6:10], x[10:13], x[13:17]
    f_c, alpha_c = u[0:Np], u[Np:2 * Np]
    q_r = p[6:10]                                # reference quaternion, a parameter

    # ---------------- STEP 3: cost (paper Eq. 9a, error definitions of Sec. III-B) ----------------
    # error state  x~ = [p~, v~, V(q_e), w~, alpha~]       q_e = q o q_r^-1
    # error input  u~ = [f_c~, alpha_c - alpha]
    q_e = quat_mul(WG_q, quat_conj(q_r))
    y_x = ca.vertcat(W_p, W_v, q_e[1:4], G_omega, alpha)
    y   = ca.vertcat(y_x, f_c, alpha_c - alpha_c)
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = y_x

    w = nmpc['weights']
    Q_x = np.r_[w['p'], w['v'], w['q'], w['omega'], [w['alpha']] * Np] # state cost
    R_u = np.r_[[w['f']] * Np, [w['dalpha']] * Np] # terminal cost
    ocp.cost.W = np.diag(np.r_[Q_x, R_u])
    ocp.cost.W_e = np.diag(Q_x * w['terminal_scale'])


    f_hover = params['mass'] * params['gravity'] / Np
    ocp.cost.yref = np.r_[np.zeros(9), np.zeros(3), np.zeros(Np), [f_hover] * Np, np.zeros(Np)]
    ocp.cost.yref_e = np.zeros(16)

    # ---------------- STEP 4: parameters (disturbance force/torque, reference quaternion) ----------------
    ocp.parameter_values = np.r_[np.zeros(6), 1.0, 0.0, 0.0, 0.0]

    # ---------------- STEP 5: constraints ----------------
    lim = nmpc['limits']
    ocp.constraints.x0 = np.r_[np.zeros(6), 1.0, 0.0, 0.0, 0.0, np.zeros(3), np.zeros(Np)]   # placeholder start

    # (9e) input bounds
    ocp.constraints.lbu = np.r_[[lim['f_min']] * Np, [-lim['alpha_max']] * Np]
    ocp.constraints.ubu = np.r_[[lim['f_max']] * Np, [lim['alpha_max']] * Np]
    ocp.constraints.idxbu = np.arange(nu)
    # (9d) velocity limits: v_x,y,z (states 3-5) and omega_x,y,z (states 10-12)
    ocp.constraints.lbx = np.r_[[-lim['v_max']] * 3, [-lim['omega_max']] * 3]
    ocp.constraints.ubx = np.r_[[lim['v_max']] * 3, [lim['omega_max']] * 3]
    ocp.constraints.idxbx = np.array([3, 4, 5, 10, 11, 12])
 
    # ---------------- STEP 6: solver options ----------------
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
 
    # ---------------- STEP 7: build ----------------
    ocp_solver = AcadosOcpSolver(ocp, json_file='omnidrone_ocp.json', generate=flag, build=flag)
    return ocp_solver, ocp

    
    


                                  



                        

if __name__ == "__main__":                                     
  path_to_yaml = os.path.abspath(sys.argv[1])
  params = utils.yaml_to_dict(path_to_yaml)
  print(params)
  ocp_solver, ocp = solver(params)