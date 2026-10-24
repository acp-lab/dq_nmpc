'''
Builds the optimal control problem around hover model
Holds the cost (weights on position error, velocity, tilt, angular rate, and input), 
the input bounds, the horizon length and step count, and the solver options.
Includes the quaternion error calculation.
'''




import os
import sys
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from dq_nmpc import utils                                      
from dq_nmpc.hover_model import export_hover_model
from casadi import Function, MX, vertcat, sin, cos, fabs, DM   


# Eq. (12):  q - q_ref = [ q (x) q_ref^-1 ]_xyz
def quat_error(q, q_ref):
    r = vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])     # q_ref inverse 
    return vertcat(
        q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
        q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
        q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
    )


def solver(params, flag = True):                              
    # get dynamical model
    model = export_hover_model(params)

    # Get size of the system
    nx = model.x.size()[0]                                     
    nu = model.u.size()[0]                                    

    # initialize ocp
    ocp = AcadosOcp()                                      
    ocp.model = model
    ocp.dims.N = params['nmpc']['horizon_steps']
    ocp.dims.nx = nx
    ocp.dims.nbx = nx
    ocp.dims.nbu = nu
    ocp.dims.nbx_e = nx
    ocp.dims.nu = nu
    ocp.dims.np = model.p.size()[0]                           
    ocp.dims.nbxe_0 = nx
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # New build directory
    ocp.code_export_directory = 'c_generated_code_hover'

    # some variables
    x = ocp.model.x                                          
    u = ocp.model.u
    p = ocp.model.p

    # Internal states of the system
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    w_b = x[10:13]

    # Desired states of the system
    ref_pos = p[0:3]
    ref_vel = p[3:6]
    ref_quat = p[6:10]
    ref_w_b = p[10:13]
    ref_control = p[13:17]

    # Control actions limits
    f_max = params['nmpc']['ubu'][0]                      
    tau_1_max = params['nmpc']['ubu'][1]
    tau_2_max = params['nmpc']['ubu'][2]
    tau_3_max = params['nmpc']['ubu'][3]

    # Cost function weights
    R_u = MX.zeros(4, 4)
    R_u[0, 0] = 20 / f_max
    R_u[1, 1] = 60 / tau_1_max
    R_u[2, 2] = 60 / tau_2_max
    R_u[3, 3] = 60 / tau_3_max

    Q_pos = MX.zeros(3, 3)
    Q_pos[0, 0] = 20
    Q_pos[1, 1] = 20
    Q_pos[2, 2] = 20

    Q_vel = MX.zeros(3, 3)
    Q_vel[0, 0] = 1
    Q_vel[1, 1] = 1
    Q_vel[2, 2] = 1

    Q_quat = MX.zeros(3, 3)
    Q_quat[0, 0] = 5
    Q_quat[1, 1] = 5
    Q_quat[2, 2] = 5

    Q_w = MX.zeros(3, 3)                                       
    Q_w[0, 0] = 1
    Q_w[1, 1] = 1
    Q_w[2, 2] = 1

    # Compute errrors
    error_pos = pos - ref_pos
    error_vel = vel - ref_vel
    error_quat = quat_error(quat, ref_quat)
    error_w = w_b - ref_w_b                                    
    error_u = ref_control - u                                  

    # Cost Function
    ocp.model.cost_expr_ext_cost = (error_pos.T @ Q_pos @ error_pos
                                    + error_quat.T @ Q_quat @ error_quat
                                    + error_u.T @ R_u @ error_u
                                    + error_vel.T @ Q_vel @ error_vel
                                    + error_w.T @ Q_w @ error_w)
    ocp.model.cost_expr_ext_cost_e = (error_pos.T @ Q_pos @ error_pos
                                      + error_quat.T @ Q_quat @ error_quat
                                      + error_vel.T @ Q_vel @ error_vel
                                      + error_w.T @ Q_w @ error_w)

    # Init condition
    hover_pos = np.array([0.0, 0.0, 1.0])
    hover_thrust = params['mass'] * params['gravity']
    ref_params = np.concatenate([
        hover_pos,                          # position
        np.zeros(3),                        # velocity
        np.array([1.0, 0.0, 0.0, 0.0]),     # quaternion (level, w first)
        np.zeros(3),                        # angular rate
        np.array([hover_thrust, 0.0, 0.0, 0.0]),   # nominal input: thrust = m*g, zero torques
    ])

    ocp.parameter_values = ref_params                         

    # Constraints
    ocp.constraints.constr_type = 'BGH'                     

    # Set constraints
    ocp.constraints.lbu = np.array(params['nmpc']['lbu'])    
    ocp.constraints.ubu = np.array(params['nmpc']['ubu'])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array(ref_params[:nx])            

    # Set options  -- everything below is SAME as the original
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'

    ## Regularization (stabilizes optimization)
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'

    ## Levenberg-Marquardt regularization (optional)
    ocp.solver_options.levenberg_marquardt = 10.0

    ## NLP Solver Settings
    # ocp.solver_options.nlp_solver_max_iter = 200
    # ocp.solver_options.nlp_solver_tol_stat = 1e-2
    ocp.solver_options.print_level = 0

    ocp.solver_options.tf = params['nmpc']['horizon_time']

    ## QP Solver Options
    # ocp.solver_options.qp_solver_cond_N = params['nmpc']['horizon_steps']
    # ocp.solver_options.qp_solver_ric_alg = 1

    ## Compilation flags for external functions (optional for performance)
    # ocp.solver_options.ext_fun_compile_flags = '-Ofast -march=native'
    # ocp.solver_options.hpipm_mode = 'SPEED'

    # Parallelization
    # ocp.solver_options.cg_use_openmp = True
    # ocp.solver_options.cg_hardcode_constraints = False
    # ocp.solver_options.cg_use_variable_weighting_matrix = True

    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.sim_method_newton_iter = 2

   
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_hover.json', build = flag, generate = flag)
    return acados_solver, ocp                                  

if __name__ == "__main__":                                     
  path_to_yaml = os.path.abspath(sys.argv[1])
  params = utils.yaml_to_dict(path_to_yaml)
  print(params)
  ocp_solver, ocp = solver(params)