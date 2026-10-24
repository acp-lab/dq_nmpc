'''
Closed-loop test: repeated solves, with your own model acting as the drone.

Your first two tests solve ONCE. The node solves ~33 times a second, warm-starting
each solve from the last one. This script does the same thing without ROS or MuJoCo,
so a controller that "stops responding after the first few ticks" shows up here.

Run from the repo root, same as your other test:
    python3 -m dq_nmpc.test_hover_closed_loop        (or however you run test_hover)
Rebuilds the solver from the CURRENT hover_ocp.py by default, so it can never test stale
compiled code. Pass --reuse to skip the rebuild and load the existing compiled solver.
'''

import sys
import numpy as np
import casadi as ca

from dq_nmpc import utils
from dq_nmpc.hover_model import export_hover_model
from dq_nmpc.hover_ocp import solver as build_solver

# ---------------------------------------------------------------- scenario
START = np.array([0.0, 0.0, 2.0])     # where the drone starts
TARGET = np.array([0.0, 0.0, 1.0])    # where it should hold
T_END = 10.0                          # seconds to simulate
PRINT_EVERY = 0.5                     # seconds between printed lines

params = utils.yaml_to_dict("config/mujoco/default/hover_config.yaml")
mass = params["mass"]
gravity = params["gravity"]
N = params["nmpc"]["horizon_steps"]
ts = params["nmpc"].get("ts", 0.0) or params["nmpc"]["horizon_time"] / N
u_hover = np.array([mass * gravity, 0.0, 0.0, 0.0])

# ---------------------------------------------------------------- the "drone"
model = export_hover_model(params)
f = ca.Function("f", [model.x, model.u], [model.f_expl_expr])


def plant_step(x, u, dt, substeps=10):
    """Advance your own model by dt seconds (RK4), holding the command u constant."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = np.array(f(x, u)).flatten()
        k2 = np.array(f(x + 0.5 * h * k1, u)).flatten()
        k3 = np.array(f(x + 0.5 * h * k2, u)).flatten()
        k4 = np.array(f(x + h * k3, u)).flatten()
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10])
    return x


def make_state(pos):
    return np.concatenate([pos, np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(3)])


# ---------------------------------------------------------------- set up the solver like the node does
BUILD = "--reuse" not in sys.argv
print("Rebuilding solver from current source..." if BUILD else "Reusing previously compiled solver (--reuse)")
ocp_solver, ocp = build_solver(params, flag=BUILD)

x_target = make_state(TARGET)
reference = np.concatenate([x_target, u_hover])
x = make_state(START)

for k in range(N + 1):
    ocp_solver.set(k, "p", reference)
    ocp_solver.set(k, "x", x)          # first-solve guess: stay where we are
for k in range(N):
    ocp_solver.set(k, "u", u_hover)

# ---------------------------------------------------------------- closed loop
print(f"start {START}  ->  target {TARGET}   (control period {ts:.3f} s, hover thrust {u_hover[0]:.4f} N)")
print(f"{'t[s]':>6} {'z[m]':>7} {'vz':>7} {'thrust':>8} {'tau':>22} {'status':>6} {'plan z_N':>9}")

steps = int(round(T_END / ts))
every = max(1, int(round(PRINT_EVERY / ts)))
log = []
bad_status = 0

for i in range(steps):
    ocp_solver.set(0, "lbx", x)
    ocp_solver.set(0, "ubx", x)
    status = ocp_solver.solve()
    u = np.array(ocp_solver.get(0, "u")).flatten()
    plan_end_z = float(ocp_solver.get(N, "x")[2])
    if status != 0:
        bad_status += 1                # unlike the node, do NOT hide failures behind a fallback

    log.append((i * ts, x[2], u[0]))
    if i % every == 0:
        print(f"{i * ts:6.2f} {x[2]:7.3f} {x[5]:7.3f} {u[0]:8.3f} "
              f"[{u[1]:6.3f},{u[2]:6.3f},{u[3]:6.3f}] {status:6d} {plan_end_z:9.3f}")

    x = plant_step(x, u, ts)

# ---------------------------------------------------------------- verdict
t, z, thrust = (np.array(c) for c in zip(*log))
err_end = abs(z[-1] - TARGET[2])
print("\n=== Summary ===")
print(f"solver failures: {bad_status} of {steps} ticks")
print(f"thrust range: {thrust.min():.3f} .. {thrust.max():.3f} N   (hover = {u_hover[0]:.3f})")
print(f"height range: {z.min():.3f} .. {z.max():.3f} m,  final error {err_end:.3f} m")

if bad_status:
    print("FAIL: the solver reported failures. Check the status codes above.")
elif thrust.max() - thrust.min() < 0.1:
    print("FAIL: thrust never moved away from hover. The controller is ignoring the position error.")
elif err_end > 0.05:
    print("FAIL: it responded but did not settle on the target.")
else:
    print("PASS: it moved toward the target and settled.")