#!/usr/bin/env python3
"""
Panda controller for picking up a dice from the floor with optimal approach angle.

- Franka Emika Panda manipulator
- Dice object on the floor 
- QP optimization for approach angle and reaching trajectory
- Camera visibility optimization
- Workspace limits enforcement
- Dual-phase approach: 1) Reach above dice, 2) Descend to grasp
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
print("Using ProxSuite QP solver")


# ------------------- Setup simulator & robot --------------------------
env = swift.Swift()
env.launch(realtime=True)

panda = rtb.models.Panda()
panda.q = panda.qr.copy()  # Start in ready pose
env.add(panda)

print(f"Initial end-effector position: {panda.fkine(panda.q).t}")

# ------------------- Global variables ----------------
workspace_limits = {
    "x": (-0.2, 1.2),    
    "y": (-0.8, 0.8),   
    "z": (0.01, 1.2)
}

x_min, x_max = workspace_limits["x"]
y_min, y_max = workspace_limits["y"]
z_min, z_max = workspace_limits["z"]

# ------------------- Dice on the floor ---------------------------
dice_x = 0.5
dice_y = 0.1  
dice_z = 0.05 + 0.025  # Floor + half dice height
dice_size = 0.05

dice_pose = sm.SE3(dice_x, dice_y, dice_z)
dice = sg.Cuboid([dice_size, dice_size, dice_size], pose=dice_pose)
dice.color = [0.2, 0.8, 0.2, 0.8]
env.add(dice)

print(f"Dice position: ({dice_x}, {dice_y}, {dice_z})")

# ------------------- Fixed camera ---------------------
camera_pose = sm.SE3(0.3, 0.6, 0.8) * sm.SE3.Rx(-0.5)
camera_fov = sg.Cylinder(radius=0.005, length=2.0, pose=camera_pose.A)
env.add(camera_fov)

print(f"Camera position: {camera_pose.t}")

# ------------------- Visualization elements ----------
sight_length = 0.3
sight_cyl = sg.Cylinder(radius=0.002, length=sight_length, pose=panda.fkine(panda.q).A)
env.add(sight_cyl)

# ------------------- Workspace visualization ----------------
wall_thickness = 0.1
alpha = 0.15
wall_color = [0.8, 0.1, 0.1, alpha]

walls = []
centers = [
    # [(x_min - wall_thickness/2), (y_min+y_max)/2, (z_min+z_max)/2],  # left wall
    [(x_max + wall_thickness/2), (y_min+y_max)/2, (z_min+z_max)/2],  # right wall
    [(x_min+x_max)/2, (y_max + wall_thickness/2), (z_min+z_max)/2],  # front wall
    [(x_min+x_max)/2, (y_min - wall_thickness/2), (z_min+z_max)/2],  # back wall
    [(x_min+x_max)/2, (y_min+y_max)/2, (z_min - wall_thickness/2)],  # FLOOR
]
sizes = [
    # [wall_thickness, y_max-y_min, z_max-z_min],  
    [wall_thickness, y_max-y_min, z_max-z_min],  
    [x_max-x_min, wall_thickness, z_max-z_min],  
    [x_max-x_min, wall_thickness, z_max-z_min],  
    [x_max-x_min, y_max-y_min, wall_thickness],
]

for c, s in zip(centers, sizes):
    wall = sg.Cuboid(s)
    wall.T = sm.SE3(*c)
    wall.color = wall_color
    walls.append(wall)

for w in walls:
    env.add(w)

env.step()

# ------------------- Controller setup --------------------------------
n_robot = panda.n
n_slack = 6
n = n_robot

# Control phases
PHASE_APPROACH = 0
PHASE_DESCEND = 1   
PHASE_LIFT = 2

current_phase = PHASE_APPROACH
approach_height = 0.25
grasp_height = dice_z + 0.02

# Controller parameters
manip_gain = 1.5  # Reduced for stability
workspace_margin = 0.02
max_joint_vel_scale = 0.6  # More conservative

# ------------------- ProxSuite QP solver setup ----------------------------
if USE_PROXSUITE:
    qp_solver = proxsuite.proxqp.dense.QP(n + n_slack, 6, 0)

# ------------------- Comprehensive Visibility Analysis Class ----------------------------
class VisibilityAnalyzer:
    def __init__(self, camera_pos, dice_pos):
        self.camera_pos = camera_pos
        self.dice_pos = dice_pos
        self.cam_to_dice = dice_pos - camera_pos
        self.cam_to_dice_norm = self.cam_to_dice / np.linalg.norm(self.cam_to_dice)
        
    def analyze_visibility(self, ee_pos, J_ee, tolerance=0.12):
        """
        Complete visibility analysis returning all needed metrics
        """
        # Vector from camera to end-effector
        cam_to_ee = ee_pos - self.camera_pos
        
        # Project end-effector position onto camera-dice line
        projection_length = np.dot(cam_to_ee, self.cam_to_dice_norm)
        projection_point = self.camera_pos + projection_length * self.cam_to_dice_norm
        
        # Distance from end-effector to the camera-dice line
        distance_to_line = np.linalg.norm(ee_pos - projection_point)
        
        # Check if end-effector is between camera and dice
        dice_distance = np.linalg.norm(self.cam_to_dice)
        ee_is_between = 0 < projection_length < dice_distance
        
        # Occlusion occurs if end-effector is close to line AND between camera and dice
        is_occluded = ee_is_between and (distance_to_line < tolerance)
        
        # Calculate severity
        if ee_is_between:
            depth_factor = 1.0 - (projection_length / dice_distance)
            distance_factor = max(0, (tolerance * 1.5 - distance_to_line) / (tolerance * 1.5))
            severity = depth_factor * distance_factor
        else:
            severity = 0.0
        
        # Calculate visibility gradient (direction to move away from line)
        if distance_to_line > 1e-6:
            away_direction = (ee_pos - projection_point) / distance_to_line
        else:
            # If exactly on line, use arbitrary lateral direction
            arbitrary_lateral = np.array([self.cam_to_dice_norm[1], -self.cam_to_dice_norm[0], 0])
            away_direction = arbitrary_lateral / (np.linalg.norm(arbitrary_lateral) + 1e-6)
        
        # Remove component along camera-dice line (only lateral movement)
        away_direction = away_direction - np.dot(away_direction, self.cam_to_dice_norm) * self.cam_to_dice_norm
        away_direction = away_direction / (np.linalg.norm(away_direction) + 1e-6)
        
        # Convert spatial gradient to joint velocity gradient
        visibility_gradient_joints = J_ee.T @ away_direction
        
        # Calculate cost
        if is_occluded or distance_to_line < 0.15:
            base_cost = 5000.0 if is_occluded else 2000.0
            distance_penalty = max(0, (0.2 - distance_to_line) / 0.2) * base_cost
            total_cost = base_cost + distance_penalty
        else:
            total_cost = max(0, (0.25 - distance_to_line) / 0.25) * 500.0
        
        return {
            'is_occluded': is_occluded,
            'distance_to_line': distance_to_line,
            'severity': severity,
            'cost': total_cost,
            'gradient_spatial': away_direction,
            'gradient_joints': visibility_gradient_joints,
            'projection_point': projection_point
        }

# Create global visibility analyzer
visibility_analyzer = VisibilityAnalyzer(camera_pose.t, np.array([dice_x, dice_y, dice_z]))

# ------------------- Optimal approach angle calculation ----------------------------
def calculate_optimal_approach_pose(dice_pos, camera_pos, phase, current_ee_pos=None):
    """
    Calculate optimal end-effector pose with visibility constraints
    """
    target_pos = dice_pos.copy()
    
    if phase == PHASE_APPROACH:
        # Phase 1: Move to optimal position above dice with maximum visibility
        target_pos[2] = dice_pos[2] + approach_height
        
        # Calculate lateral offset for visibility
        cam_to_dice = dice_pos - camera_pos
        cam_to_dice_2d = cam_to_dice[:2] / np.linalg.norm(cam_to_dice[:2])
        
        # Choose lateral direction based on workspace constraints
        perpendicular_left = np.array([-cam_to_dice_2d[1], cam_to_dice_2d[0]])
        perpendicular_right = np.array([cam_to_dice_2d[1], -cam_to_dice_2d[0]])
        
        # Try both directions and choose the one that stays in workspace
        lateral_offset = 0.15
        for direction in [perpendicular_left, perpendicular_right]:
            candidate_pos = target_pos.copy()
            candidate_pos[0] = dice_pos[0] + direction[0] * lateral_offset
            candidate_pos[1] = dice_pos[1] + direction[1] * lateral_offset
            
            # Check workspace bounds
            if (x_min + 0.1 <= candidate_pos[0] <= x_max - 0.1 and
                y_min + 0.1 <= candidate_pos[1] <= y_max - 0.1):
                target_pos = candidate_pos
                break
        
        # Ensure within workspace bounds
        target_pos[0] = np.clip(target_pos[0], x_min + 0.05, x_max - 0.05)
        target_pos[1] = np.clip(target_pos[1], y_min + 0.05, y_max - 0.05)
        
        # Orientation: angled toward dice
        dice_direction = dice_pos[:2] - target_pos[:2]
        dice_direction = dice_direction / (np.linalg.norm(dice_direction) + 1e-6)
        
        angle = np.arctan2(dice_direction[1], dice_direction[0])
        R = sm.SE3.Rz(angle) * sm.SE3.Ry(np.pi/4) * sm.SE3.Rx(np.pi)
        
    elif phase == PHASE_DESCEND:
        # Phase 2: Descend with slight lateral offset
        target_pos[2] = grasp_height
        
        # Maintain small lateral offset
        cam_to_dice_2d = (dice_pos[:2] - camera_pos[:2])
        cam_to_dice_2d = cam_to_dice_2d / np.linalg.norm(cam_to_dice_2d)
        perpendicular = np.array([-cam_to_dice_2d[1], cam_to_dice_2d[0]])
        
        lateral_offset = 0.03
        target_pos[0] = dice_pos[0] + perpendicular[0] * lateral_offset
        target_pos[1] = dice_pos[1] + perpendicular[1] * lateral_offset
        
        R = sm.SE3.Ry(np.pi/8) * sm.SE3.Rx(np.pi)
        
    else:  # PHASE_LIFT
        # Phase 3: Lift dice up
        target_pos[2] = dice_pos[2] + 0.15
        R = sm.SE3.Rx(np.pi)
    
    T_target = sm.SE3(target_pos) * R
    return T_target, target_pos

# ------------------- Workspace constraint function ----------------------------
def get_workspace_constraints(xyz, J_xyz, dt=0.01):
    """
    Generate workspace constraints - only when necessary
    """
    C_rows = []
    d_vals = []
    
    margin = workspace_margin
    activation_dist = 0.1
    
    # Z constraints (floor and ceiling)
    if xyz[2] < z_min + activation_dist:
        C_row = -J_xyz[2, :]
        d_val = (z_min + margin - xyz[2]) / dt
        if xyz[2] < z_min + margin:
            d_val = max(d_val, 0.5)
        C_rows.append(C_row)
        d_vals.append(d_val)
        
    if xyz[2] > z_max - activation_dist:
        C_row = J_xyz[2, :]
        d_val = (z_max - margin - xyz[2]) / dt
        C_rows.append(C_row)
        d_vals.append(d_val)
    
    # X and Y constraints
    for axis in range(2):
        min_lim = [x_min, y_min][axis]
        max_lim = [x_max, y_max][axis]
        
        if xyz[axis] < min_lim + activation_dist:
            C_row = -J_xyz[axis, :]
            d_val = (min_lim + margin - xyz[axis]) / dt
            C_rows.append(C_row)
            d_vals.append(d_val)
            
        if xyz[axis] > max_lim - activation_dist:
            C_row = J_xyz[axis, :]
            d_val = (max_lim - margin - xyz[axis]) / dt
            C_rows.append(C_row)
            d_vals.append(d_val)
    
    return C_rows, d_vals

# ------------------- ProxSuite solver ----------------------------
def solve_qp_proxsuite(H, g, A_eq, b_eq, C_in, d_in, lb, ub):
    """Solve QP using ProxSuite with relaxed settings"""
    try:
        qp_solver.init(H, g, A_eq, b_eq, C_in, d_in, lb, ub)
        qp_solver.settings.eps_abs = 1e-4  # Relaxed
        qp_solver.settings.eps_rel = 1e-4  # Relaxed
        qp_solver.settings.max_iter = 500   # Reduced
        qp_solver.solve()
        
        if qp_solver.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
            return qp_solver.results.x
        else:
            return None
    except:
        return None

# ------------------- Main control step function -----------------------------------
def step():
    global current_phase
    
    dt = 0.01
    wTe = panda.fkine(panda.q).A
    xyz = wTe[:3, 3]
    
    # Calculate target pose
    T_target, target_xyz = calculate_optimal_approach_pose(
        np.array([dice_x, dice_y, dice_z]), 
        camera_pose.t, 
        current_phase,
        xyz
    )
    
    # Calculate error
    eTep = np.linalg.inv(wTe) @ T_target.A
    position_error = np.linalg.norm(eTep[:3, -1])
    
    # Get Jacobian
    J_e = panda.jacobe(panda.q)
    
    # COMPREHENSIVE visibility analysis - ALL variables defined here
    vis_result = visibility_analyzer.analyze_visibility(xyz, J_e[:3, :])
    is_occluded = vis_result['is_occluded']
    distance_to_line = vis_result['distance_to_line']
    visibility_cost = vis_result['cost']
    visibility_gradient_joints = vis_result['gradient_joints']
    
    # Phase transitions
    if current_phase == PHASE_APPROACH and position_error < 0.05:
        current_phase = PHASE_DESCEND
        print(f"Switching to DESCEND phase at iteration {iters}")
    elif current_phase == PHASE_DESCEND and position_error < 0.02:
        current_phase = PHASE_LIFT
        print(f"Switching to LIFT phase at iteration {iters}")
    
    # Desired velocity
    v_manip, _ = rtb.p_servo(sm.SE3(wTe), T_target, manip_gain)
    
    # Adaptive gains based on phase and visibility
    if is_occluded:
        v_manip *= 0.3  # Very slow when occluded
    elif current_phase == PHASE_APPROACH:
        v_manip *= 1.0
    elif current_phase == PHASE_DESCEND:
        v_manip *= 0.6
        v_manip[2] = max(v_manip[2], -0.08)
    else:  # LIFT
        v_manip *= 0.8
        v_manip[2] = min(v_manip[2], 0.15)

    # ----------------- Simplified QP Formulation ----------------
    # Quadratic cost - much simpler and more stable
    Q = np.eye(n + n_slack) * 1e-3
    
    # Basic position tracking
    position_weight = 100.0 if not is_occluded else 20.0
    Q[:n, :n] += J_e[:3, :].T @ J_e[:3, :] * position_weight
    
    # Basic orientation tracking  
    orientation_weight = 10.0 if not is_occluded else 2.0
    Q[:n, :n] += J_e[3:, :].T @ J_e[3:, :] * orientation_weight
    
    # Visibility quadratic cost - only when needed
    if is_occluded or distance_to_line < 0.20:
        vis_weight = 2000.0 if is_occluded else 500.0
        Q[:n, :n] += np.outer(visibility_gradient_joints, visibility_gradient_joints) * vis_weight

    # Equality constraints
    A_eq = np.zeros((6, n + n_slack))
    A_eq[:, :n] = J_e
    b_eq = v_manip
    
    # Inequality constraints - simplified
    C_in_rows = []
    d_in_rows = []
    
    # Joint limits only
    Gf, hf = panda.joint_velocity_damper(0.1, 0.8, panda.n)
    if Gf is not None and hf is not None:
        Gf_padded = np.c_[Gf, np.zeros((Gf.shape[0], n_slack))]
        C_in_rows.append(Gf_padded)
        d_in_rows.append(hf)
    
    # Workspace constraints
    ws_C_rows, ws_d_vals = get_workspace_constraints(xyz, J_e[:3, :], dt)
    for C_row, d_val in zip(ws_C_rows, ws_d_vals):
        C_full = np.zeros((1, n + n_slack))
        C_full[0, :n] = C_row
        C_in_rows.append(C_full)
        d_in_rows.append(np.array([d_val]))
    
    # Hard visibility constraint when severely occluded
    if is_occluded and distance_to_line < 0.08:
        # Force movement away from camera-dice line
        C_row = -visibility_gradient_joints / (np.linalg.norm(visibility_gradient_joints) + 1e-6)
        d_val = -0.05  # Minimum speed away from line
        
        C_full = np.zeros((1, n + n_slack))
        C_full[0, :n] = C_row
        C_in_rows.append(C_full)
        d_in_rows.append(np.array([d_val]))
    
    # Stack constraints
    C_in = np.vstack(C_in_rows) if C_in_rows else None
    d_in = np.hstack(d_in_rows) if d_in_rows else None
    
    # Linear cost - simplified
    g = np.zeros(n + n_slack)
    
    # Strong visibility gradient when needed
    if is_occluded or distance_to_line < 0.15:
        vis_linear_weight = 1000.0 if is_occluded else 200.0
        g[:n] += visibility_gradient_joints * vis_linear_weight
    
    # Joint velocity limits
    velocity_scale = 0.3 if is_occluded else 0.6
    joint_vel_limit = panda.qdlim[:n] * velocity_scale
    lb = np.r_[-joint_vel_limit, -1000*np.ones(n_slack)]
    ub = np.r_[ joint_vel_limit, 1000*np.ones(n_slack)]
    
    # Solve QP
    if USE_PROXSUITE:
        qd_full = solve_qp_proxsuite(Q, g, A_eq, b_eq, C_in, d_in, lb, ub)
    else:
        try:
            qd_full = qp.solve_qp(Q, g, C_in, d_in, A_eq, b_eq, lb=lb, ub=ub)
        except:
            qd_full = None
    
    # Enhanced fallback - now visibility_gradient_joints is guaranteed to be defined
    if qd_full is None:
        print("QP solver failed - using visibility-aware fallback")
        lambda_reg = 0.05
        
        if is_occluded or distance_to_line < 0.12:
            print("EMERGENCY: Using pure visibility mode")
            # Pure visibility motion
            qd = visibility_gradient_joints * 0.08
            qd = np.clip(qd, -joint_vel_limit, joint_vel_limit)
        else:
            # Normal fallback with visibility weighting
            J_reg = J_e.T @ np.linalg.inv(J_e @ J_e.T + lambda_reg * np.eye(6))
            qd = J_reg @ v_manip
            
            # Add visibility correction
            if distance_to_line < 0.20:
                visibility_correction = visibility_gradient_joints * 0.03
                qd = qd * 0.8 + visibility_correction * 0.2
        
        # Workspace protection
        xyz_next = xyz + J_e[:3, :] @ qd * dt
        if xyz_next[2] < z_min + workspace_margin:
            correction = (z_min + workspace_margin - xyz_next[2]) / dt
            qd += J_e[2, :] * correction / (np.linalg.norm(J_e[2, :]) + 1e-6)
    else:
        qd = qd_full[:n]
    
    # Apply velocity
    panda.qd = qd
    sight_cyl.pose = panda.fkine(panda.q).A
    
    # Check completion
    if current_phase == PHASE_LIFT and position_error < 0.03:
        return True
    
    # Debug output with phase timing
    if iters % 50 == 0:
        phase_names = ["APPROACH", "DESCEND", "LIFT"]
        visibility_status = "OCCLUDED" if is_occluded else f"CLEAR (d={distance_to_line:.3f}m)"
        
        print(f"Iter {iters}: Phase {phase_names[current_phase]} (err={position_error:.4f}), "
              f"Pos({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}), "
              f"Target({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}), "
              f"Visibility: {visibility_status}")
        
        # Phase-specific warnings
        if current_phase == PHASE_APPROACH and iters > 1500:
            print("  Warning: Taking long time in APPROACH phase")
        elif current_phase == PHASE_DESCEND and iters > 3500:
            print("  Warning: Taking long time in DESCEND phase") 
        elif current_phase == PHASE_LIFT and iters > 5500:
            print("  Warning: Taking long time in LIFT phase")
        
        # Workspace violations
        violations = []
        if xyz[0] < x_min or xyz[0] > x_max: violations.append("X")
        if xyz[1] < y_min or xyz[1] > y_max: violations.append("Y") 
        if xyz[2] < z_min or xyz[2] > z_max: violations.append("Z")
        if violations:
            print(f"  Workspace violations: {', '.join(violations)}")
    
    return False

# ------------------- Main execution ----------------
print(f"Starting dice grasping task...")
print(f"Dice at: ({dice_x:.3f}, {dice_y:.3f}, {dice_z:.3f})")
print(f"Workspace: x{workspace_limits['x']}, y{workspace_limits['y']}, z{workspace_limits['z']}")

arrived = False
iters = 0
max_iters = 8000

while not arrived and iters < max_iters:
    arrived = step()
    iters += 1
    env.step(0.01)

final_pos = panda.fkine(panda.q).t
print(f"\nTask completed: {arrived} after {iters} iterations")
print(f"Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
print(f"Final phase: {['APPROACH', 'DESCEND', 'LIFT'][current_phase]}")

if final_pos[2] < z_min:
    print(f"*** FLOOR VIOLATION: Z = {final_pos[2]:.3f} < {z_min:.3f} ***")
else:
    print(f"Workspace respected: Z = {final_pos[2]:.3f} >= {z_min:.3f}")