#!/usr/bin/env python3
"""
Panda visibility-maximization controller with fixed camera and workspace limits.

- Franka Emika Panda manipulator
- Fixed camera in environment
- QP-based velocity controller with slack variables using ProxSuite
- Vision collision avoidance ensures robot does not block camera FOV
- Workspace limits in x, y, z are enforced via inequality constraints
- Updated: allow soft orientation, position slack, strong occlusion penalty,
  and ProxSuite initialization that supports box + linear inequalities.
"""

import math
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import swift

# Use ProxSuite instead of qpsolvers
import proxsuite
USE_PROXSUITE = True

# ------------------- Utility: robust shortest-rotation between vectors ----------
def transform_between_vectors(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return sm.SE3()  # identity if degenerate
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = math.acos(dot)
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        if dot > 0.0:
            return sm.SE3()
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        axis /= np.linalg.norm(axis)
        return sm.SE3.AngleAxis(math.pi, axis)
    axis = axis / axis_norm
    return sm.SE3.AngleAxis(angle, axis)

# ------------------- Setup simulator & robot --------------------------
env = swift.Swift()
env.launch(realtime=True)

panda = rtb.models.Panda()

# Better initial configuration - use ready pose instead of zero pose
panda.q = panda.qr.copy()
env.add(panda)

print(f"Initial robot configuration: {panda.q}")
print(f"Initial end-effector position: {panda.fkine(panda.q).t}")
T_init = panda.fkine(panda.q)   # initial EE pose

# ------------------- Workspace limits ----------------
workspace_limits = {
    "x": (0.1, 0.9),
    "y": (-0.4, 0.4),
    "z": (0.01, 0.8)
}

# ------------------- Target in environment ---------------------------
target_x = 0.5
target_y = 0.0
target_z = 0.05
target_pose = sm.SE3(target_x, target_y, target_z)
target = sg.Sphere(radius=0.03, pose=target_pose)
env.add(target)

print(f"Target position: ({target_x}, {target_y}, {target_z})")

# ------------------- Fixed camera in environment ---------------------
camera_pose = sm.SE3(0.5, 0.4, 1.0) * sm.SE3.Rx(-0.4)
camera_fov = sg.Cylinder(radius=0.005, length=2.5, pose=camera_pose.A)
env.add(camera_fov)

# ------------------- Visualization of robot FOV (optional) ----------
sight_length = 0.7
sight_cyl = sg.Cylinder(radius=0.002, length=sight_length, pose=panda.fkine(panda.q).A)
env.add(sight_cyl)

# ------------------- Desired end-effector references -----------------------
# Tep_vis: camera-facing orientation used only for vision/occlusion checks
Tep_vis = sm.SE3(target_pose.t) * sm.SE3.Rz(np.pi)
# Tep_des: position-only desired pose (orientation will be soft)
Tep_des = sm.SE3(target_pose.t)

# ------------------- Controller setup --------------------------------
n_robot = panda.n
n_slack = 10
n = n_robot = n_robot = n_robot = panda.n  # ensure consistent naming

S_ARM_LIN = slice(n, n + 3)
S_ARM_ANG = slice(n + 3, n + 6)
S_CAM = slice(n + 6, n + 9)
S_OCC = n + 9

ps = 0.1
pi = 0.9
manip_gain = 1.0
camera_gain = 25.0

# Workspace constraint parameters
workspace_margin = 0.06
workspace_activation_distance = 0.65

# Workspace boundaries
x_min, x_max = workspace_limits["x"]
y_min, y_max = workspace_limits["y"]
z_min, z_max = workspace_limits["z"]

wall_thickness = 0.1
alpha = 0.2
wall_color = [0.8, 0.1, 0.1, alpha]

walls = []
centers = [
    # [(x_min - wall_thickness/2), (y_min+y_max)/2, (z_min+z_max)/2],  # left wall
    [(x_max + wall_thickness/2), (y_min+y_max)/2, (z_min+z_max)/2],  # right wall
    [(x_min+x_max)/2, (y_max + wall_thickness/2), (z_min+z_max)/2],  # front wall
    [(x_min+x_max)/2, (y_min - wall_thickness/2), (z_min+z_max)/2],  # back wall
    [(x_min+x_max)/2, (y_min+y_max)/2, (z_min - wall_thickness/2)],  # FLOOR
    # [(x_min+x_max)/2, (y_min+y_max)/2, (z_max + wall_thickness/2)]   # CEILING
]
sizes = [
    # [wall_thickness, y_max-y_min, z_max-z_min],  # left wall
    [wall_thickness, y_max-y_min, z_max-z_min],  # right wall
    [x_max-x_min, wall_thickness, z_max-z_min],  # front wall
    [x_max-x_min, wall_thickness, z_max-z_min],  # back wall
    [x_max-x_min, y_max-y_min, wall_thickness],  # FLOOR
    # [x_max-x_min, y_max-y_min, wall_thickness]   # CEILING
]

for c, s in zip(centers, sizes):
    wall = sg.Cuboid(s)
    wall.T = sm.SE3(*c)
    wall.color = wall_color
    walls.append(wall)

for w in walls:
    env.add(w)

env.step()

# ------------------- ProxSuite QP solver setup ----------------------------
if USE_PROXSUITE:
    n_var = n + n_slack
    n_eq = 6
    n_ineq_max = 300  # safe upper bound for linear inequalities
    qp_solver = proxsuite.proxqp.dense.QP(n_var, n_eq, n_ineq_max)

# ------------------- Improved workspace constraint function ----------------------------
def get_workspace_constraints(xyz, J_xyz, dt=0.01):
    """
    Generate inequality constraints to enforce workspace boundaries.
    Ensures min-constraints (including floor) produce sufficiently negative RHS to force corrections.
    """
    C_rows = []
    d_vals = []

    limits = [
        (workspace_limits["x"], 0),
        (workspace_limits["y"], 1),
        (workspace_limits["z"], 2)
    ]

    for (min_lim, max_lim), axis_idx in limits:
        current_pos = float(xyz[axis_idx])

        dist_to_min = current_pos - (min_lim + workspace_margin)
        dist_to_max = (max_lim - workspace_margin) - current_pos

        # MIN constraint (e.g., floor): -J_axis * qd <= - (min + margin - current)/dt
        if axis_idx == 2:
            # floor (always active)
            C_row = -J_xyz[axis_idx, :].copy()
            # compute rhs; we clamp to a minimum negative value so it is effective
            raw = - (min_lim + workspace_margin - current_pos) / max(dt, 1e-4)
            d_val = raw
            if current_pos < min_lim + workspace_margin:
                # stronger if already violating margin
                d_val = min(d_val, -0.25)   # force more upward velocity
            else:
                # when nearby, ensure at least small negative value that forces correction
                d_val = min(d_val, -0.02)
            C_rows.append(C_row)
            d_vals.append(d_val)

            # ceiling (max) when close
            if dist_to_max < workspace_activation_distance:
                C_row = J_xyz[axis_idx, :].copy()
                raw_max = (max_lim - workspace_margin - current_pos) / max(dt, 1e-4)
                d_val_max = max(raw_max, 0.02)
                C_rows.append(C_row)
                d_vals.append(d_val_max)
        else:
            # X/Y min
            if dist_to_min < workspace_activation_distance or current_pos <= min_lim + workspace_margin:
                C_row = -J_xyz[axis_idx, :].copy()
                raw = - (min_lim + workspace_margin - current_pos) / max(dt, 1e-4)
                d_val = raw
                if current_pos < min_lim + workspace_margin:
                    d_val = min(d_val, -0.05)
                else:
                    d_val = min(d_val, -0.01)
                C_rows.append(C_row)
                d_vals.append(d_val)

            # X/Y max
            if dist_to_max < workspace_activation_distance or current_pos >= max_lim - workspace_margin:
                C_row = J_xyz[axis_idx, :].copy()
                raw_max = (max_lim - workspace_margin - current_pos) / max(dt, 1e-4)
                d_val_max = max(raw_max, 0.02)
                C_rows.append(C_row)
                d_vals.append(d_val_max)

    return C_rows, d_vals



# ------------------- ProxSuite QP solver ----------------------------
def solve_qp_proxsuite(H, g, A_eq, b_eq, C_in, d_in, lb, ub):
    """
    Robust ProxSuite wrapper that creates a qp object with exact sizes every call.
    This avoids \"wrong model setup\" errors when the number of inequalities or
    box constraints changes between solves.
    """
    try:
        n_var = int(H.shape[0])

        # Make A_eq / b_eq consistent arrays
        if A_eq is None:
            A_eq_local = np.empty((0, n_var))
            b_eq_local = np.empty((0,))
            n_eq = 0
        else:
            A_eq_local = np.asarray(A_eq, dtype=float)
            b_eq_local = np.asarray(b_eq, dtype=float)
            n_eq = A_eq_local.shape[0]

        # Make C_in / d_in consistent arrays
        if C_in is None:
            C_in_local = np.empty((0, n_var))
            d_in_local = np.empty((0,))
            n_ineq = 0
        else:
            C_in_local = np.asarray(C_in, dtype=float)
            d_in_local = np.asarray(d_in, dtype=float)
            n_ineq = C_in_local.shape[0]

        # Create a fresh QP sized exactly for this problem
        qp_local = proxsuite.proxqp.dense.QP(n_var, n_eq, n_ineq)

        # init and solve
        qp_local.init(H, g, A_eq_local, b_eq_local, C_in_local, d_in_local, lb, ub)

        qp_local.settings.eps_abs = 1e-6
        qp_local.settings.eps_rel = 1e-6
        qp_local.settings.max_iter = 1000

        qp_local.solve()

        status = qp_local.results.info.status
        if status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
            return qp_local.results.x
        else:
            print(f"ProxSuite solver status: {status}")
            return None

    except Exception as e:
        print(f"ProxSuite solver error: {e}")
        return None

# ------------------- Step function -----------------------------------

def step():
    dt = 0.01
    wTe = panda.fkine(panda.q).A

    # Position error (for termination)
    eTep = np.linalg.inv(wTe) @ Tep_des.A
    et = np.linalg.norm(eTep[:3, -1])

    # desired cartesian velocities (position + orientation reference)
    # use Tep_des for motion target; Tep_vis used only inside vision damper checks
    v_des_cart, _ = rtb.p_servo(sm.SE3(wTe), Tep_des, manip_gain)
    # scale orientation desire smaller (we want orientation to be optimized but not dominate)
    v_des_cart[3:] *= 0.5

    # Jacobian
    J_e = panda.jacobe(panda.q)
    J_pos = J_e[:3, :]    # 3 x n
    J_rot = J_e[3:, :]    # 3 x n

    # --- Cost weights (tune these) ---
    # High weight on position accuracy, moderate on orientation
    wp_base = 200.0       # position importance
    wor_base = 5.0        # orientation importance
    # make them depend on distance (closer -> more precise)
    wp = wp_base * (1.0 + 2.0 / (1.0 + et))      # bigger when close
    wor = wor_base

    Wp = np.diag([wp, wp, wp])
    Wor = np.diag([wor, wor, wor])

    reg = 1e-3  # small regularizer on qd

    # Build H and g for variables qd (size n) and slack variables (n_slack)
    # H_q = J_pos^T Wp J_pos + J_rot^T Wor J_rot + reg*I
    H_q = J_pos.T @ Wp @ J_pos + J_rot.T @ Wor @ J_rot + reg * np.eye(n)
    g_q = - (J_pos.T @ Wp @ v_des_cart[:3] + J_rot.T @ Wor @ v_des_cart[3:])

    # Expand to full H (n + n_slack)
    H = np.zeros((n + n_slack, n + n_slack))
    H[:n, :n] = H_q
    g = np.zeros(n + n_slack)
    g[:n] = g_q

    # Add diagonal cost for slacks (penalize using slacks)
    slack_penalty = 1e3
    for i in range(n, n + n_slack):
        H[i, i] = slack_penalty
    # Make occlusion slack extremely expensive
    H[S_OCC, S_OCC] = 1e7

    # ----------------- Inequality constraints ----------------
    C_in_rows = []
    d_in_rows = []

    # Joint velocity damper
    Gf, hf = panda.joint_velocity_damper(ps, pi, panda.n)
    if Gf is not None and hf is not None:
        Gf_padded = np.c_[Gf, np.zeros((Gf.shape[0], n_slack))]
        C_in_rows.append(Gf_padded)
        d_in_rows.append(hf)

    # Vision collision damper (use Tep_vis for LOS checks internally if required)
    c_Ain, c_bin = panda.vision_collision_damper(
        target,
        camera=camera_pose,
        q=panda.q,
        di=0.15,
        ds=0.25,
        xi=1.0,
        end=panda.link_dict["panda_hand"],
        start=panda.link_dict["panda_link0"],
    )
    if c_Ain is not None and c_bin is not None:
        m = c_Ain.shape[0]
        # attach slacks: last column is occlusion slack (S_OCC index)
        c_block = np.c_[c_Ain, np.zeros((m, n_slack))]
        c_block[:, S_OCC] = -1.0
        C_in_rows.append(c_block)
        d_in_rows.append(c_bin)

    # Workspace constraints
    xyz = wTe[:3, 3]
    ws_C_rows, ws_d_vals = get_workspace_constraints(xyz, J_pos, dt)
    for C_row, d_val in zip(ws_C_rows, ws_d_vals):
        C_full = np.zeros((1, n + n_slack))
        C_full[0, :n] = C_row
        C_in_rows.append(C_full)
        d_in_rows.append(np.array([d_val]))

    # Stack inequalities
    if C_in_rows:
        C_in = np.vstack(C_in_rows)
        d_in = np.hstack(d_in_rows)
    else:
        C_in = None
        d_in = None

    # ----------------- Box bounds ----------------
    lb = np.r_[-panda.qdlim[:n] * 0.8, -100*np.ones(n_slack)]
    ub = np.r_[ panda.qdlim[:n] * 0.8, 100*np.ones(n_slack)]

    # ----------------- Solve QP ----------------
    # The QP variable is x = [qd; s]
    # Minimize 1/2 x^T H x + g^T x  subject to C_in x <= d_in, lb <= x <= ub
    if USE_PROXSUITE:
        # ProxSuite expects numpy arrays, convert empty inequalities to correct shapes
        try:
            qd_full = solve_qp_proxsuite(H, g, None, None, C_in, d_in, lb, ub)
        except Exception as e:
            print("ProxSuite solve exception:", e)
            qd_full = None
    else:
        try:
            qd_full = qp.solve_qp(H, g, C_in, d_in, None, None, lb=lb, ub=ub)
        except Exception as e:
            print("qpsolvers error:", e)
            qd_full = None

    if qd_full is None:
        # fallback: damped pseudo-inverse to follow v_des_cart
        lambda_d = 0.05
        dq = J_e.T @ np.linalg.inv(J_e @ J_e.T + lambda_d*np.eye(6)) @ v_des_cart
        # floor safety
        xyz_next = xyz + J_pos @ dq * dt
        if xyz_next[2] < z_min + workspace_margin:
            violation = (z_min + workspace_margin) - xyz_next[2]
            correction_vel = violation / dt
            J_z = J_pos[2, :]
            correction_dq = (J_z * correction_vel) / (np.dot(J_z, J_z) + 1e-6)
            dq = dq + correction_dq
        qd = dq
    else:
        qd = qd_full[:n]

    # Conservative scaling
    max_scale = 0.4
    scale = min(max_scale, 0.6/et if et > 0.3 else max_scale)
    qd = qd * scale

    panda.qd = qd
    sight_cyl.pose = panda.fkine(panda.q).A

    # Debug prints
    if iters % 50 == 0:
        violations = []
        if xyz[0] > x_max: violations.append(f"x > {x_max:.3f}")
        if xyz[1] < y_min: violations.append(f"y < {y_min:.3f}")
        if xyz[1] > y_max: violations.append(f"y > {y_max:.3f}")
        if xyz[2] < z_min: violations.append(f"z < {z_min:.3f} (FLOOR!)")
        status = "VIOLATIONS: " + ', '.join(violations) if violations else "OK"
        print(f"Iter {iters}: Pos({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) - {status}, Err: {et:.4f}")

    return et < 0.03


# ------------------- Main loop ----------------
arrived = False
iters = 0
print(f"Starting control loop...")
print(f"Target: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")
print(f"Workspace: x{workspace_limits['x']}, y{workspace_limits['y']}, z{workspace_limits['z']}")

while not arrived and iters < 4000:
    arrived = step()
    iters += 1
    env.step(0.01)

print(f"Finished after {iters} iterations; arrived = {arrived}")
final_pos = panda.fkine(panda.q).t
print(f"Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
