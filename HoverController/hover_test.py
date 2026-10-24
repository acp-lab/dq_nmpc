'''
test file to verify ocp solving behavior
'''

import numpy as np
import casadi as ca

from dq_nmpc import utils
from dq_nmpc.hover_model import export_hover_model
from dq_nmpc.hover_ocp import solver as build_solver

# Load your existing configuration
params = utils.yaml_to_dict("config/mujoco/default/hover_config.yaml")

mass = params["mass"]
gravity = params["gravity"]
thrust = mass * gravity

def test_hover_init():
        

    print("\n=== Test 1.1: Physics at hover ===")

    model = export_hover_model(params)
    dynamics = ca.Function(
        "hover_dynamics",
        [model.x, model.u],
        [model.f_expl_expr]
    )

    # State: [position, velocity, quaternion (w,x,y,z), angular velocity]
    x_hover = np.array([
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    # Input: [thrust, torque_x, torque_y, torque_z]
    u_hover = np.array([thrust, 0.0, 0.0, 0.0])

    derivative = np.array(dynamics(x_hover, u_hover)).flatten()

    print(f"Expected hover thrust: {thrust:.5f} N")
    print("State derivatives:", derivative)
    print("Maximum absolute derivative:", np.max(np.abs(derivative)))

    assert np.allclose(derivative, 0.0, atol=1e-8), \
        "FAIL: the model is not at hover equilibrium"

    print("PASS: model is at hover equilibrium")

    print("\n=== Test 1.2: NMPC solver at hover ===")

    # Reuse the compiled solver.
    # If this fails because generated files are missing, rebuild first.
    ocp_solver, ocp = build_solver(params, flag=False)

    N = params["nmpc"]["horizon_steps"]

    # Reference: desired state followed by nominal control
    reference = np.concatenate([x_hover, u_hover])

    for k in range(N + 1):
        ocp_solver.set(k, "p", reference)
        ocp_solver.set(k, "x", x_hover)

    for k in range(N):
        ocp_solver.set(k, "u", u_hover)

    status = ocp_solver.solve()
    print("Solver status:", status)

    if status != 0:
        raise RuntimeError(f"FAIL: acados solver returned status {status}")

    u_cmd = np.array(ocp_solver.get(0, "u")).flatten()

    print("Command [thrust, tau_x, tau_y, tau_z]:", u_cmd)
    print(f"Expected thrust: {thrust:.5f} N")

    assert np.isclose(u_cmd[0], thrust, atol=0.1), \
        "FAIL: thrust is not close to the expected hover thrust"

    assert np.allclose(u_cmd[1:], 0.0, atol=0.05), \
        "FAIL: hover torques are not close to zero"

    print("PASS: solver returns approximately the expected hover command")

    ocp_solver = None
    print("\nAll standalone hover checks passed.")

def test_hover_below():
    print("\n=== Test 2.1: Physics at hover ===")

    model = export_hover_model(params)

    dynamics = ca.Function("hover_dynamics_below",
                           [model.x, model.u],
                           [model.f_expl_expr]
                           )

    # Desired State: z = 1m
    x_ref = np.array([
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0])

    x0 = x_ref.copy()
    x0[2] = 0.5

    # Nominal hover control
    u_hover = np.array([thrust, 0.0, 0.0, 0.0])

    print(f"Starting altitude: {x0[2]:.2f} m")
    print(f"Desired altitude:  {x_ref[2]:.2f} m")
    print(f"Altitude error:    {x_ref[2] - x0[2]:.2f} m")

    # Check the physical model at the starting state.
    # At hover thrust, the drone has zero vertical acceleration,
    # so the controller must command additional thrust to climb.
    derivative = np.array(dynamics(x0, u_hover)).flatten()

    print("Vertical acceleration with hover thrust:",
        f"{derivative[5]:.4f} m/s^2")

      # Build the existing solver using the generated acados code.
    ocp_solver, ocp = build_solver(params, flag=False)

    N = params["nmpc"]["horizon_steps"]

    # Reference contains the desired state and nominal control.
    reference = np.concatenate([x_ref, u_hover])

    for k in range(N + 1):
        ocp_solver.set(k, "p", reference)
        ocp_solver.set(k, "x", x_ref)

    for k in range(N):
        ocp_solver.set(k, "u", u_hover)

    # Allow the drone's initial state to be below the target.
    # This assumes the OCP has full-state initial bounds configured.
    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)

    status = ocp_solver.solve()
    print("Solver status:", status)

    if status != 0:
        raise RuntimeError(
            f"FAIL: acados solver returned status {status}"
        )

    u_cmd = np.array(ocp_solver.get(0, "u")).flatten()

    print("Command [thrust, tau_x, tau_y, tau_z]:", u_cmd)
    print(f"Hover thrust: {thrust:.4f} N")

    if u_cmd[0] > thrust + 0.1:
        print("PASS: solver commands additional thrust to climb.")
    else:
        print(
            "FAIL: solver did not command enough additional thrust "
            "to climb."
        )

    ocp_solver = None




    

if __name__ == "__main__":
    print("=== TEST 1: Hover in place and make sure solver keeps the same thrust ===")
    test_hover_init() 

    print("=== TEST 2: Start 0.5m under 1m to see if drone corrects itself ===")
    test_hover_below()

