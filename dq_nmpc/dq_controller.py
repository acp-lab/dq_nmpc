import os
import sys
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from dq_nmpc import utils
from dq_nmpc import export_model
from casadi import Function, MX, vertcat, sin, cos, fabs, DM
def solver(params, flag = True):
    # get dynamical model
    model, get_trans, get_quat, constraint, error_lie_2, dual_error, ln, Ad, conjugate, rotation = export_model(params)

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

    # some variables
    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p
    Q   = p[-(nx+nx+nu):-(nx+nu)]
    Q_e = p[-(nx+nu):-(nu)]
    
    # Internal states of the system
    dual = x[0:8]
    w_b = x[8:11]
    v_b = x[11:14]
    #v_i = rotation(x[0:4], v_b)

    # Desired states of the system
    ref_dual = p[0:8]
    ref_w_b = p[8:11]
    ref_v_i = p[11:14]
    ref_control = p[14:18]

    # Control actions limits
    f_max = params['nmpc']['ubu'][0]
    tau_1_max = params['nmpc']['ubu'][1]
    tau_2_max = params['nmpc']['ubu'][2]
    tau_3_max = params['nmpc']['ubu'][3]
    R = p[-(nu):]
    R[0] = R[0]/f_max
    R[1] = R[1]/tau_1_max
    R[2] = R[2]/tau_2_max
    R[3] = R[3]/tau_3_max

    Q_dual = MX.zeros(8, 8)
    Q_dual[0, 0] = Q[7]
    Q_dual[1, 1] = Q[8]
    Q_dual[2, 2] = Q[9]
    Q_dual[3, 3] = Q[10]

    Q_dual[4, 4] = Q[0]
    Q_dual[5, 5] = Q[1]
    Q_dual[6, 6] = Q[2]
    Q_dual[7, 7] = Q[3]
    
    # Linear velocity inertial frame
    Q_v = MX.zeros(3, 3)
    Q_v[0, 0] = Q[4]
    Q_v[1, 1] = Q[5]
    Q_v[2, 2] = Q[6]
    
    # Angular velocity body frame
    Q_w = MX.zeros(3, 3)
    Q_w[0, 0] = Q[11]
    Q_w[1, 1] = Q[12]
    Q_w[2, 2] = Q[13]

    # Control Actions
    R_u = MX.zeros(4, 4)
    R_u[0, 0] = R[0]
    R_u[1, 1] = R[1]
    R_u[2, 2] = R[2]
    R_u[3, 3] = R[3]
    
    # Compute errrors
    error_dual = dual_error(ref_dual, dual)
    ln_error = ln(error_dual)
    error_w = w_b - ref_w_b
    error_v = v_b - ref_v_i
    error_u = ref_control - u 


    # Cost Function
    ocp.model.cost_expr_ext_cost = ln_error.T@Q_dual@ln_error + error_u.T@R_u@error_u + error_v.T@Q_v@error_v + error_w.T@Q_w@error_w
    ocp.model.cost_expr_ext_cost_e = ln_error.T@Q_dual@ln_error + error_v.T@Q_v@error_v + error_w.T@Q_w@error_w

    # Init condition
    ref_params = np.array([1.0, 0.0, 0.0, 0.0,    # Primary part dualquaternion
                                     0.0, 0.0, 0.0, 0.0,    # Dual part dualquaternion
                                     0.0, 0.0, 0.0,         # Angular velocity body frame
                                     0.0, 0.0, 0.0,         # Linear velocity body frame
                                     0.0, 0.0, 0.0, 0.0
                                     ])  
    # Init Values for the cost
    cost_params = np.ones(nx + nx + nu)

    ocp.parameter_values = np.concatenate([ref_params, cost_params])
    # Constraints
    ocp.constraints.constr_type = 'BGH'

    # Set constraints
    ocp.constraints.lbu = np.array(params['nmpc']['lbu'])
    ocp.constraints.ubu = np.array(params['nmpc']['ubu'])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.array(ref_params[:nx])

    # Set options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # Efficient QP solver
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Fast real-time SQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # Gauss-Newton approximation
    ocp.solver_options.integrator_type = 'IRK'  # Implicit Runge-Kutta (IRK)

    ## Regularization (stabilizes optimization)
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'  

    ## Levenberg-Marquardt regularization (optional)
    ocp.solver_options.levenberg_marquardt = 10.0

    ## NLP Solver Settings
    ocp.solver_options.nlp_solver_max_iter = 200  # Maximum iterations
    ocp.solver_options.nlp_solver_tol_stat = 1e-2  # Tolerance for stationarity
    ocp.solver_options.print_level = 0  # Suppress print output

    ocp.solver_options.tf = params['nmpc']['horizon_time']

    ## QP Solver Options
    ##ocp.solver_options.qp_solver_warm_start = 1  # Enable QP hot-start
    ocp.solver_options.qp_solver_cond_N = params['nmpc']['horizon_steps']  # Number of QP stages for partial condensing
    ocp.solver_options.qp_solver_ric_alg = 1  # Use Riccati-based algorithm

    ## Compilation flags for external functions (optional for performance)
    ocp.solver_options.ext_fun_compile_flags = '-Ofast -march=native'
    ocp.solver_options.hpipm_mode = 'SPEED'  # Prioritize speed in QP solver

    # Parallelization
    ocp.solver_options.cg_use_openmp = True  # Enable OpenMP parallelization
    ocp.solver_options.cg_hardcode_constraints = False  # Allow runtime constraint changes
    ocp.solver_options.cg_use_variable_weighting_matrix = True  # Support time-varying costs

    ocp.solver_options.sim_method_num_stages = 4  # IRK-GL4: 4 stages for accuracy
    ocp.solver_options.sim_method_num_steps = 1  # Number of integration steps
    ocp.solver_options.sim_method_newton_iter = 2  # Newton iterations for convergence

    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_mpc.json', build = flag, generate = flag)
    return acados_solver, ocp

if __name__ == "__main__":
  path_to_yaml = os.path.abspath(sys.argv[1])
  params = utils.yaml_to_dict(path_to_yaml)
  print(params)
  ocp_solver, ocp = solver(params)