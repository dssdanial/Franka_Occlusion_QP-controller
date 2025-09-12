import numpy as np
from scipy.spatial.transform import Rotation as R
import roboticstoolbox as rtb
from spatialmath import SE3
import spatialmath as sm
import spatialgeometry as sg
import qpsolvers
import warnings
import swift
import time

class Obstacle:
    """Represents a static obstacle in the environment"""
    def __init__(self, position, radius=0.1):
        self.position = np.array(position)
        self.radius = radius
    
    def get_distance_to_point(self, point):
        """Get distance from obstacle surface to a point (negative if inside)"""
        return np.linalg.norm(point - self.position) - self.radius

class Workspace:
    """Defines rectangular workspace with hard boundaries"""
    def __init__(self, x_limits, y_limits, z_limits):
        self.x_min, self.x_max = x_limits
        self.y_min, self.y_max = y_limits
        self.z_min, self.z_max = z_limits
    
    def is_inside(self, point, margin=0.0):
        """Check if point is inside workspace with optional margin"""
        x, y, z = point
        return (self.x_min + margin <= x <= self.x_max - margin and 
                self.y_min + margin <= y <= self.y_max - margin and 
                self.z_min + margin <= z <= self.z_max - margin)
    
    def get_distance_to_boundaries(self, point):
        """Get minimum distance to any workspace boundary (negative if outside)"""
        x, y, z = point
        distances = [
            #x - self.x_min,      # Distance to left wall
            self.x_max - x,      # Distance to right wall
            y - self.y_min,      # Distance to back wall  
            self.y_max - y,      # Distance to front wall
            z - self.z_min,      # Distance to floor
            # Note: No ceiling constraint in original code
        ]
        return min(distances)

class NEOController:
    """
    Enhanced NEO Controller with improved constraint handling
    """
    
    def __init__(self, robot, workspace=None, obstacles=None):
        self.robot = robot
        self.workspace = workspace  
        self.obstacles = obstacles if obstacles is not None else []
        self.n_joints = robot.n
        
        # NEO parameters - adjusted for better performance
        self.beta = 0.8                    # End-effector gain (reduced for stability)
        self.lambda_q = 0.1                # Velocity minimization (increased)
        self.lambda_delta = 1000.0         # Slack penalty (increased)
        
        # Enhanced velocity damper parameters
        self.xi = 1.5          # Damper gain (increased)
        self.di = 0.20         # Influence distance (increased for more margin)
        self.ds = 0.08         # Stopping distance (increased safety margin)
        
        # Safety margins
        self.workspace_margin = 0.05       # Additional workspace margin
        self.obstacle_margin = 0.02        # Additional obstacle margin
        
        print(f"Enhanced NEO Controller")
        print(f"Robot: {robot.name} ({self.n_joints} joints)")
        print(f"Velocity damper: ξ={self.xi}, di={self.di}m, ds={self.ds}m")
        print(f"Safety margins: workspace={self.workspace_margin}m, obstacle={self.obstacle_margin}m")
    
    def _get_robot_collision_points(self, q):
        """Get robot collision points and their Jacobians with better link coverage"""
        points = []
        jacobians = []
        
        try:
            # Get individual link transforms for better collision detection
            link_transforms = []
            for i in range(self.n_joints):
                try:
                    T_link = self.robot.fkine(q, end=i+1)  # Transform to each joint
                    link_transforms.append(T_link)
                except:
                    continue
            
            # End-effector (most important)
            T_ee = self.robot.fkine(q)
            points.append(T_ee.t)
            J_ee = self.robot.jacob0(q)[:3, :]
            jacobians.append(J_ee)
            
            # Add several intermediate points along the kinematic chain
            if len(link_transforms) >= 3:
                # Joint 3 position (shoulder/elbow area)
                points.append(link_transforms[2].t)
                J_joint3 = self.robot.jacob0(q, end=3)[:3, :]
                jacobians.append(J_joint3)
            
            if len(link_transforms) >= 5:
                # Joint 5 position (wrist area)
                points.append(link_transforms[4].t)
                J_joint5 = self.robot.jacob0(q, end=5)[:3, :]
                jacobians.append(J_joint5)
            
            # Add additional interpolated points for better coverage
            if len(link_transforms) >= 2:
                # Midpoint between base and end-effector
                mid_point = (link_transforms[1].t + T_ee.t) / 2
                points.append(mid_point)
                # Approximate Jacobian for midpoint
                J_mid = (self.robot.jacob0(q, end=2)[:3, :] + J_ee) / 2
                jacobians.append(J_mid)
                
        except Exception as e:
            print(f"Warning - using fallback collision points: {e}")
            # Fallback to just end-effector
            T_ee = self.robot.fkine(q)
            J_ee = self.robot.jacob0(q)[:3, :]
            points = [T_ee.t]
            jacobians = [J_ee]
        
        return points, jacobians
    
    def _formulate_workspace_constraints(self, points, jacobians):
        """Enhanced workspace constraints with better margins"""
        if self.workspace is None:
            return np.empty((0, self.n_joints)), np.empty(0)
        
        A_constraints = []
        b_constraints = []
        
        for point, J in zip(points, jacobians):
            x, y, z = point
            
            # Check each workspace boundary with enhanced margins
            boundaries = [
                # ('x_min', x - (self.workspace.x_min + self.workspace_margin), np.array([-1, 0, 0])),
                ('x_max', (self.workspace.x_max - self.workspace_margin) - x, np.array([1, 0, 0])),   
                ('y_min', y - (self.workspace.y_min + self.workspace_margin), np.array([0, -1, 0])),
                ('y_max', (self.workspace.y_max - self.workspace_margin) - y, np.array([0, 1, 0])),   
                ('z_min', z - (self.workspace.z_min + self.workspace_margin), np.array([0, 0, -1]))
            ]
            
            for boundary_name, distance, normal in boundaries:
                if distance < self.di:  # Within influence distance
                    # Enhanced velocity damper constraint
                    if distance > self.ds:
                        # ḋ ≤ ξ(d-ds)/(di-ds) with enhanced damping
                        damper_limit = self.xi * (distance - self.ds) / (self.di - self.ds)
                    else:
                        # Very close to boundary - strong stopping constraint
                        damper_limit = -0.1  # Allow slight movement away from boundary
                    
                    # Distance rate: ḋ = ∇d · q̇ = normal^T · J · q̇
                    distance_jacobian = normal @ J
                    
                    # Scale constraint based on proximity (closer = stronger constraint)
                    constraint_weight = max(1.0, (self.di - distance) / self.di * 2.0)
                    
                    A_constraints.append(constraint_weight * distance_jacobian)
                    b_constraints.append(constraint_weight * damper_limit)
                    
                    if np.random.rand() < 0.02:  # Debug print occasionally
                        print(f"Workspace: {boundary_name}, d={distance:.3f}, limit={damper_limit:.3f}, weight={constraint_weight:.2f}")
        
        if len(A_constraints) > 0:
            return np.vstack(A_constraints), np.array(b_constraints)
        else:
            return np.empty((0, self.n_joints)), np.empty(0)
    
    def _formulate_obstacle_constraints(self, points, jacobians):
        """Enhanced obstacle constraints with better detection"""
        A_constraints = []
        b_constraints = []
        
        for point, J in zip(points, jacobians):
            for i, obs in enumerate(self.obstacles):
                # Distance to obstacle surface (with additional margin)
                raw_distance = obs.get_distance_to_point(point)
                distance = raw_distance - self.obstacle_margin
                
                if distance < self.di:  # Within influence distance
                    # Direction from obstacle center to robot point
                    direction = point - obs.position
                    dist_to_center = np.linalg.norm(direction)
                    
                    if dist_to_center > 1e-6:
                        normal = direction / dist_to_center  # Outward normal
                        
                        # Enhanced velocity damper constraint  
                        if distance > self.ds:
                            damper_limit = self.xi * (distance - self.ds) / (self.di - self.ds)
                        else:
                            # Very close to obstacle - strong repulsion
                            damper_limit = -0.2
                        
                        # Distance rate: ḋ = normal^T · J · q̇
                        distance_jacobian = normal @ J
                        
                        # Scale constraint based on proximity and obstacle size
                        proximity_factor = max(1.0, (self.di - distance) / self.di * 3.0)
                        size_factor = obs.radius / 0.05  # Scale with obstacle size
                        constraint_weight = proximity_factor * size_factor
                        
                        A_constraints.append(constraint_weight * distance_jacobian)
                        b_constraints.append(constraint_weight * damper_limit)
                        
                        if np.random.rand() < 0.02:
                            print(f"Obstacle {i}: d={distance:.3f}, raw_d={raw_distance:.3f}, limit={damper_limit:.3f}, weight={constraint_weight:.2f}")
        
        if len(A_constraints) > 0:
            return np.vstack(A_constraints), np.array(b_constraints)
        else:
            return np.empty((0, self.n_joints)), np.empty(0)
    
    def solve(self, q, T_desired, q_dot_limits=None):
        """Enhanced QP solver with better constraint handling"""
        if q_dot_limits is None:
            q_dot_limits = np.ones(self.n_joints) * 0.8  # Reduced for safety
        
        try:
            # Check if target is reachable and safe
            if self.workspace is not None:
                target_safe = self.workspace.is_inside(T_desired.t, margin=self.workspace_margin)
                if not target_safe:
                    print(f"WARNING: Target outside safe workspace!")
                    # Modify target to be inside workspace
                    target_pos = T_desired.t.copy()
                    target_pos[0] = np.clip(target_pos[0], 
                                          self.workspace.x_min + self.workspace_margin,
                                          self.workspace.x_max - self.workspace_margin)
                    target_pos[1] = np.clip(target_pos[1],
                                          self.workspace.y_min + self.workspace_margin, 
                                          self.workspace.y_max - self.workspace_margin)
                    target_pos[2] = np.clip(target_pos[2],
                                          self.workspace.z_min + self.workspace_margin,
                                          self.workspace.z_max)
                    T_desired = SE3.Trans(target_pos) * SE3.RPY(T_desired.rpy())
                    print(f"Target clipped to: {target_pos}")
            
            # Current end-effector pose
            T_current = self.robot.fkine(q)
            
            # Pose error with saturation
            pose_error = T_desired.t - T_current.t
            pose_error = np.clip(pose_error, -0.3, 0.3)  # Limit large errors
            
            # Rotation error (simplified and saturated)
            R_error = T_desired.R @ T_current.R.T
            try:
                rot_error = R.from_matrix(R_error).as_rotvec()
                rot_error = np.clip(rot_error, -0.2, 0.2)
            except:
                rot_error = np.zeros(3)
            
            # Desired end-effector velocity
            nu_desired = self.beta * np.concatenate([pose_error, rot_error])
            
            # Robot Jacobian
            J = self.robot.jacob0(q)
            
            # Get robot collision points
            points, jacobians = self._get_robot_collision_points(q)
            
            # Formulate constraints
            A_workspace, b_workspace = self._formulate_workspace_constraints(points, jacobians)
            A_obstacles, b_obstacles = self._formulate_obstacle_constraints(points, jacobians)
            
            # QP variables: [q_dot, slack]
            n_vars = self.n_joints + 6
            
            # Enhanced objective with regularization
            P = np.zeros((n_vars, n_vars))
            P[:self.n_joints, :self.n_joints] = self.lambda_q * np.eye(self.n_joints)
            P[self.n_joints:, self.n_joints:] = self.lambda_delta * np.eye(6)
            
            # Add small regularization for numerical stability
            P += 1e-6 * np.eye(n_vars)
            
            q_obj = np.zeros(n_vars)
            
            # Equality constraint: J * q_dot + slack = nu_desired  
            A_eq = np.zeros((6, n_vars))
            A_eq[:, :self.n_joints] = J
            A_eq[:, self.n_joints:] = np.eye(6)
            b_eq = nu_desired
            
            # Inequality constraints
            constraints = []
            
            # Joint velocity limits with safety factor
            safety_factor = 0.8
            A_vel = np.zeros((2 * self.n_joints, n_vars))
            A_vel[:self.n_joints, :self.n_joints] = np.eye(self.n_joints)
            A_vel[self.n_joints:, :self.n_joints] = -np.eye(self.n_joints)
            b_vel = np.concatenate([safety_factor * q_dot_limits, safety_factor * q_dot_limits])
            constraints.append((A_vel, b_vel))
            
            # Workspace constraints
            if A_workspace.shape[0] > 0:
                A_ws_full = np.zeros((A_workspace.shape[0], n_vars))
                A_ws_full[:, :self.n_joints] = A_workspace
                constraints.append((A_ws_full, b_workspace))
            
            # Obstacle constraints  
            if A_obstacles.shape[0] > 0:
                A_obs_full = np.zeros((A_obstacles.shape[0], n_vars))
                A_obs_full[:, :self.n_joints] = A_obstacles
                constraints.append((A_obs_full, b_obstacles))
            
            # Combine all inequality constraints
            if constraints:
                A_ineq = np.vstack([A for A, b in constraints])
                b_ineq = np.concatenate([b for A, b in constraints])
            else:
                A_ineq = np.empty((0, n_vars))
                b_ineq = np.empty(0)
            
            # Variable bounds
            lb = np.full(n_vars, -np.inf)
            ub = np.full(n_vars, np.inf)
            
            # Joint velocity bounds
            lb[:self.n_joints] = -safety_factor * q_dot_limits
            ub[:self.n_joints] = safety_factor * q_dot_limits
            
            # Slack bounds (allow reasonable slack)
            lb[self.n_joints:] = -2.0
            ub[self.n_joints:] = 2.0
            
            # Solve QP with multiple solvers as fallback
            solution = None
            solvers_to_try = ['quadprog', 'osqp', 'cvxopt']
            
            for solver in solvers_to_try:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        solution = qpsolvers.solve_qp(
                            P=P,
                            q=q_obj,
                            A=A_ineq,
                            b=b_ineq,
                            G=A_eq,
                            h=b_eq,
                            lb=lb,
                            ub=ub,
                            solver=solver
                        )
                    
                    if solution is not None and not np.any(np.isnan(solution)):
                        break
                except:
                    continue
            
            if solution is not None and not np.any(np.isnan(solution)):
                q_dot = solution[:self.n_joints]
                slack = solution[self.n_joints:]
                
                # Monitor and report constraint violations
                if A_workspace.shape[0] > 0:
                    violations = A_workspace @ q_dot - b_workspace
                    max_violation = np.max(violations)
                    if max_violation > 0.01:
                        print(f"Workspace violation: {max_violation:.3f}")
                
                if A_obstacles.shape[0] > 0:
                    violations = A_obstacles @ q_dot - b_obstacles
                    max_violation = np.max(violations)
                    if max_violation > 0.01:
                        print(f"Obstacle violation: {max_violation:.3f}")
                
                return np.clip(q_dot, -q_dot_limits, q_dot_limits)
            else:
                raise Exception("All QP solvers failed")
        
        except Exception as e:
            if np.random.rand() < 0.1:
                print(f"QP failed: {e}")
            
            # Enhanced emergency fallback
            points, jacobians = self._get_robot_collision_points(q)
            
            # Check for critical violations
            critical_violations = False
            emergency_vel = np.zeros(self.n_joints)
            
            # Workspace violations
            for point, J in zip(points, jacobians):
                if self.workspace is not None:
                    if not self.workspace.is_inside(point, margin=0):
                        critical_violations = True
                        # Emergency push toward workspace center
                        workspace_center = np.array([
                            (self.workspace.x_min + self.workspace.x_max) / 2,
                            (self.workspace.y_min + self.workspace.y_max) / 2,
                            (self.workspace.z_min + self.workspace.z_max) / 2
                        ])
                        direction_to_center = workspace_center - point
                        direction_to_center = direction_to_center / (np.linalg.norm(direction_to_center) + 1e-6)
                        try:
                            emergency_vel += 0.3 * J.T @ direction_to_center
                        except:
                            pass
                
                # Obstacle violations
                for obs in self.obstacles:
                    distance = obs.get_distance_to_point(point)
                    if distance < 0:  # Inside obstacle
                        critical_violations = True
                        direction = point - obs.position
                        direction = direction / (np.linalg.norm(direction) + 1e-6)
                        try:
                            emergency_vel += 0.5 * J.T @ direction
                        except:
                            pass
            
            if critical_violations:
                print("EMERGENCY: Critical violations detected!")
                return np.clip(emergency_vel, -q_dot_limits * 0.5, q_dot_limits * 0.5)
            else:
                # Regular fallback - pseudo-inverse with damping
                try:
                    T_current = self.robot.fkine(q)
                    pose_error = T_desired.t - T_current.t
                    nu_desired = self.beta * 0.5 * np.concatenate([pose_error, np.zeros(3)])
                    
                    J = self.robot.jacob0(q)
                    J_reg = J.T @ J + 0.1 * np.eye(self.n_joints)
                    q_dot = np.linalg.solve(J_reg, J.T @ nu_desired)
                    return np.clip(q_dot, -q_dot_limits * 0.3, q_dot_limits * 0.3)
                except:
                    return np.zeros(self.n_joints)

def create_workspace_visual(env, workspace):
    """Create workspace visualization with better visibility"""
    x_min, x_max = workspace.x_min, workspace.x_max
    y_min, y_max = workspace.y_min, workspace.y_max
    z_min, z_max = workspace.z_min, workspace.z_max
    
    wall_thickness = 0.03
    alpha = 0.7
    wall_color = [0.9, 0.2, 0.2, alpha]
    
    walls = []
    wall_specs = [
        # Right wall (x_max)
        [[x_max + wall_thickness/2, (y_min+y_max)/2, (z_min+z_max)/2], 
         [wall_thickness, y_max-y_min, z_max-z_min]],
        # Left wall (x_min) 
        # [[x_min - wall_thickness/2, (y_min+y_max)/2, (z_min+z_max)/2],
        #  [wall_thickness, y_max-y_min, z_max-z_min]],
        # Front wall (y_max)
        [[(x_min+x_max)/2, y_max + wall_thickness/2, (z_min+z_max)/2],
         [x_max-x_min, wall_thickness, z_max-z_min]],
        # Back wall (y_min)
        [[(x_min+x_max)/2, y_min - wall_thickness/2, (z_min+z_max)/2],
         [x_max-x_min, wall_thickness, z_max-z_min]],
        # Floor (z_min)
        [[(x_min+x_max)/2, (y_min+y_max)/2, z_min - wall_thickness/2],
         [x_max-x_min, y_max-y_min, wall_thickness]]
    ]
    
    for center, size in wall_specs:
        wall = sg.Cuboid(size)
        wall.T = sm.SE3(*center)
        wall.color = wall_color
        walls.append(wall)
        env.add(wall)
    
    return walls

def create_obstacle_visual(env, obstacle):
    """Create obstacle visualization with enhanced visibility"""
    obs_sphere = sg.Sphere(radius=obstacle.radius)
    obs_sphere.T = sm.SE3.Trans(*obstacle.position)
    obs_sphere.color = [1, 0.3, 0.3, 0.9]  # Brighter red
    env.add(obs_sphere)
    return obs_sphere

# Main simulation
if __name__ == "__main__":
    env = swift.Swift()
    env.launch(realtime=True)
    
    # Create Panda robot
    panda = rtb.models.Panda()
    # Start with robot safely inside workspace
    panda.q = np.array([0.0, -0.3, 0.0, -1.2, 0.0, 0.9, 0.0])
    env.add(panda)
    
    # Define more conservative workspace with better margins
    workspace = Workspace(
        x_limits=(0.15, 0.65),    # 50cm depth with margin
        y_limits=(-0.35, 0.35),   # 70cm width with margin  
        z_limits=(0.0, 0.55)      # 55cm height with margin
    )
    
    # Verify robot starts inside workspace
    T_initial = panda.fkine(panda.q)
    print(f"Initial robot position: {T_initial.t}")
    print(f"Workspace: X=[{workspace.x_min:.2f}, {workspace.x_max:.2f}], Y=[{workspace.y_min:.2f}, {workspace.y_max:.2f}], Z=[{workspace.z_min:.2f}, {workspace.z_max:.2f}]")
    print(f"Robot starts inside workspace: {workspace.is_inside(T_initial.t)}")
    
    # Create obstacles with better positioning
    obstacles = [
        Obstacle([0.4, 0.15, 0.3], radius=0.06),   # Slightly larger for better visibility
        Obstacle([0.5, -0.15, 0.25], radius=0.05),
        Obstacle([0.35, 0.0, 0.4], radius=0.04),   # Additional obstacle
    ]
    
    # Add visuals
    walls = create_workspace_visual(env, workspace)
    
    obstacle_visuals = []
    for i, obs in enumerate(obstacles):
        obs_visual = create_obstacle_visual(env, obs)
        obstacle_visuals.append(obs_visual)
    
    # Create enhanced NEO controller
    controller = NEOController(panda, workspace, obstacles)
    
    # Set target INSIDE the safe workspace
    T_desired = SE3.Trans(0.2, 0.17, 0.05) * SE3.RPY(np.pi, 0, 0)
    print(f"Target position: {T_desired.t}")
    print(f"Target inside workspace: {workspace.is_inside(T_desired.t, margin=controller.workspace_margin)}")
    
    # Target visualization
    target = sg.Sphere(radius=0.025)
    target.T = T_desired
    target.color = [0, 1, 0, 1.0]
    env.add(target)
    
    try:
        env.step()
        time.sleep(1.0)  # Give time to see initial setup
    except:
        pass
    
    print(f"\n{'='*60}")
    print("ENHANCED NEO CONTROLLER")
    print("Features: Enhanced margins, better obstacle detection, robust QP solving")
    print(f"{'='*60}")
    
    # Control loop
    dt = 0.04  # Slightly faster updates
    max_iterations = 500
    tolerance = 0.03  # Tighter tolerance
    
    positions = []
    errors = []
    violations_log = []
    
    try:
        for i in range(max_iterations):
            q = panda.q.copy()
            
            # Get control input
            q_dot = controller.solve(q, T_desired)
            
            # Update robot
            panda.q = q + q_dot * dt
            
            # Track progress and violations
            T_current = panda.fkine(panda.q)
            positions.append(T_current.t.copy())
            error = np.linalg.norm(T_current.t - T_desired.t)
            errors.append(error)
            
            # Check violations with enhanced detection
            points, _ = controller._get_robot_collision_points(panda.q)
            workspace_violations = sum([1 for p in points if not workspace.is_inside(p, margin=0.01)])
            obstacle_violations = 0
            for p in points:
                for obs in obstacles:
                    if obs.get_distance_to_point(p) < 0.01:
                        obstacle_violations += 1
            
            total_violations = workspace_violations + obstacle_violations
            violations_log.append(total_violations)
            
            if error < tolerance:
                print(f"SUCCESS: Target reached in {i} iterations!")
                break
            
            if i % 25 == 0:
                print(f"Iter {i:3d}: error={error:.4f}m, ws_viol={workspace_violations}, obs_viol={obstacle_violations}, |q̇|={np.linalg.norm(q_dot):.3f}")
            
            try:
                env.step()
                time.sleep(0.025)
            except:
                continue
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Results analysis
    if len(errors) > 0:
        total_violations = sum(violations_log)
        final_error = errors[-1]
        success = final_error < tolerance
        
        print(f"\n{'='*60}")
        print("ENHANCED NEO CONTROLLER RESULTS")
        print(f"{'='*60}")
        print(f"Final error: {final_error:.4f}m")
        print(f"Target tolerance: {tolerance:.4f}m")
        print(f"Total violations: {total_violations}")
        print(f"Success: {success}")
        
        if total_violations == 0:
            print("🎉 PERFECT: No constraint violations!")
        elif total_violations < 5:
            print("✅ GOOD: Minimal violations")
        elif total_violations < 20:
            print("⚠️  ACCEPTABLE: Some violations but controlled")
        else:
            print("❌ FAILED: Too many violations")
        
        if success and total_violations < 10:
            print("🏆 MISSION ACCOMPLISHED: Target reached safely!")
    
    print(f"\nPress Enter to exit...")
    input()
    
    try:
        env.close()
    except:
        pass
