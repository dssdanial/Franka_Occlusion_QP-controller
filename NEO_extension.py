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

class Workspace:
    """
    Defines a rectangular workspace with 4 walls (front, left, right, floor)
    """
    def __init__(self, x_limits, y_limits, z_limits):
        self.x_min, self.x_max = x_limits
        self.y_min, self.y_max = y_limits
        self.z_min, self.z_max = z_limits
    
    def is_inside(self, point):
        """Check if a point is inside the workspace"""
        x, y, z = point
        return (self.x_min <= x <= self.x_max and 
                self.y_min <= y <= self.y_max and 
                self.z_min <= z <= self.z_max)
    
    def get_violations_and_jacobians(self, positions, jacobians):
        """
        Get workspace violations and corresponding constraint jacobians
        Returns constraint matrices for QP: A_ineq @ q_dot <= b_ineq
        """
        A_constraints = []
        b_constraints = []
        
        for i, (pos, J) in enumerate(zip(positions, jacobians)):
            x, y, z = pos
            
            # Right wall (x <= x_max): violation if x > x_max
            if x > self.x_max - 0.05:  # 5cm buffer
                # Constraint: x_dot <= -gain * (x - x_max)
                # x_dot = [1,0,0] @ J @ q_dot, so: [1,0,0] @ J @ q_dot <= -gain * violation
                violation = x - self.x_max + 0.05
                A_constraints.append(np.array([1, 0, 0]) @ J)
                b_constraints.append(-2.0 * violation)
            
            # Left wall (x >= x_min): violation if x < x_min  
            if x < self.x_min + 0.05:
                violation = self.x_min - x + 0.05
                A_constraints.append(-np.array([1, 0, 0]) @ J)
                b_constraints.append(-2.0 * violation)
            
            # Front wall (y <= y_max): violation if y > y_max
            if y > self.y_max - 0.05:
                violation = y - self.y_max + 0.05
                A_constraints.append(np.array([0, 1, 0]) @ J)
                b_constraints.append(-2.0 * violation)
            
            # Back wall (y >= y_min): violation if y < y_min
            if y < self.y_min + 0.05:
                violation = self.y_min - y + 0.05
                A_constraints.append(-np.array([0, 1, 0]) @ J)
                b_constraints.append(-2.0 * violation)
            
            # Floor (z >= z_min): violation if z < z_min
            if z < self.z_min + 0.05:
                violation = self.z_min - z + 0.05
                A_constraints.append(-np.array([0, 0, 1]) @ J)
                b_constraints.append(-2.0 * violation)
        
        if len(A_constraints) > 0:
            return np.vstack(A_constraints), np.array(b_constraints)
        else:
            return np.empty((0, jacobians[0].shape[1])), np.empty(0)

class NEOController:
    """
    Simplified NEO Controller with Proper Workspace Constraints
    """
    
    def __init__(self, robot, workspace=None):
        self.robot = robot
        self.workspace = workspace
        self.n_joints = robot.n
        
        # Control parameters
        self.beta = 1.5                    # End-effector velocity gain
        self.lambda_q = 0.01              # Joint velocity regularization
        self.lambda_manip = 0.1           # Manipulability weight
        
        # Collision points to check (simplified)
        self.check_points = [
            {'name': 'end_effector', 'link': 6},    # End-effector
            {'name': 'wrist', 'link': 5},           # Wrist
            {'name': 'elbow', 'link': 3},           # Elbow
        ]
        
        print(f"Initialized NEO controller for {robot.name} with {self.n_joints} joints")
    
    def _get_manipulability_measure(self, q):
        """Get manipulability measure and its gradient"""
        try:
            J = self.robot.jacob0(q)
            Jv = J[:3, :]  # Translational Jacobian
            
            # Manipulability measure (Yoshikawa)
            manipulability = np.sqrt(np.linalg.det(Jv @ Jv.T + 1e-6 * np.eye(3)))
            
            # Gradient using finite differences
            grad_manip = np.zeros(self.n_joints)
            epsilon = 1e-6
            
            for i in range(self.n_joints):
                q_plus = q.copy()
                q_plus[i] += epsilon
                
                J_plus = self.robot.jacob0(q_plus)
                Jv_plus = J_plus[:3, :]
                manip_plus = np.sqrt(np.linalg.det(Jv_plus @ Jv_plus.T + 1e-6 * np.eye(3)))
                
                grad_manip[i] = (manip_plus - manipulability) / epsilon
            
            return manipulability, grad_manip
            
        except:
            return 1.0, np.zeros(self.n_joints)
    
    def _get_collision_points_info(self, q):
        """Get positions and jacobians for collision checking points"""
        positions = []
        jacobians = []
        
        for point_info in self.check_points:
            link_idx = point_info['link']
            
            # Get position
            if link_idx == 6:  # End-effector
                T = self.robot.fkine(q)
                pos = T.t
                J_full = self.robot.jacob0(q)
                J_trans = J_full[:3, :]
            else:
                # Get intermediate link position and Jacobian
                T = self.robot.fkine(q)  # We'll use a simpler approach
                pos = T.t  # This is a simplification - in reality you'd compute each link
                J_full = self.robot.jacob0(q)
                J_trans = J_full[:3, :]
                # Zero out joints beyond this link
                J_trans[:, link_idx+1:] = 0
            
            positions.append(pos)
            jacobians.append(J_trans)
        
        return positions, jacobians
    
    def solve(self, q, T_desired, q_dot_limits=None):
        """
        Solve for joint velocities using QP with workspace constraints
        """
        if q_dot_limits is None:
            q_dot_limits = np.ones(self.n_joints) * 2.0
        
        try:
            # Current end-effector pose
            T_current = self.robot.fkine(q)
            
            # Pose error
            pos_error = T_desired.t - T_current.t
            
            # Rotation error (simplified)
            R_error = T_desired.R @ T_current.R.T
            rot_error = R.from_matrix(R_error).as_rotvec()
            rot_error = np.clip(rot_error, -0.5, 0.5)  # Limit rotation
            
            # Desired end-effector velocity
            nu_desired = self.beta * np.concatenate([pos_error, rot_error])
            
            # Robot Jacobian
            J = self.robot.jacob0(q)
            
            # Get manipulability
            manip, grad_manip = self._get_manipulability_measure(q)
            
            # Get workspace constraints
            positions, jacobians = self._get_collision_points_info(q)
            
            if self.workspace is not None:
                A_workspace, b_workspace = self.workspace.get_violations_and_jacobians(positions, jacobians)
            else:
                A_workspace = np.empty((0, self.n_joints))
                b_workspace = np.empty(0)
            
            # Set up QP problem: minimize 0.5 * x^T * P * x + q^T * x
            # subject to: A * x <= b, and Aeq * x = beq
            
            n_vars = self.n_joints + 6  # q_dot + slack variables for end-effector constraint
            
            # Objective matrix P and vector q
            P = np.zeros((n_vars, n_vars))
            P[:self.n_joints, :self.n_joints] = self.lambda_q * np.eye(self.n_joints)  # Regularization
            P[self.n_joints:, self.n_joints:] = 1000.0 * np.eye(6)  # Heavy penalty on slack
            
            q_obj = np.zeros(n_vars)
            q_obj[:self.n_joints] = -self.lambda_manip * grad_manip  # Maximize manipulability
            
            # Equality constraint: J * q_dot + slack = nu_desired
            A_eq = np.zeros((6, n_vars))
            A_eq[:, :self.n_joints] = J
            A_eq[:, self.n_joints:] = np.eye(6)
            b_eq = nu_desired
            
            # Inequality constraints
            constraints = []
            
            # Joint velocity limits: -q_dot_max <= q_dot <= q_dot_max
            A_vel = np.zeros((2 * self.n_joints, n_vars))
            A_vel[:self.n_joints, :self.n_joints] = np.eye(self.n_joints)
            A_vel[self.n_joints:2*self.n_joints, :self.n_joints] = -np.eye(self.n_joints)
            b_vel = np.concatenate([q_dot_limits, q_dot_limits])
            
            constraints.append((A_vel, b_vel))
            
            # Workspace constraints
            if A_workspace.shape[0] > 0:
                A_workspace_full = np.zeros((A_workspace.shape[0], n_vars))
                A_workspace_full[:, :self.n_joints] = A_workspace
                constraints.append((A_workspace_full, b_workspace))
            
            # Combine all inequality constraints
            if constraints:
                A_ineq = np.vstack([A for A, b in constraints])
                b_ineq = np.concatenate([b for A, b in constraints])
            else:
                A_ineq = np.empty((0, n_vars))
                b_ineq = np.empty(0)
            
            # Solve QP
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
                        solver='quadprog'
                    )
                
                if solution is not None and not np.any(np.isnan(solution)):
                    q_dot = solution[:self.n_joints]
                    slack = solution[self.n_joints:]
                    
                    # Check if slack is reasonable (small)
                    if np.linalg.norm(slack) > 0.5:
                        print(f"Large slack detected: {np.linalg.norm(slack):.3f}")
                    
                    return np.clip(q_dot, -q_dot_limits, q_dot_limits)
                else:
                    raise Exception("QP returned None or NaN")
            
            except Exception as e:
                print(f"QP failed: {e}")
                # Fallback to damped pseudo-inverse
                J_pinv = np.linalg.pinv(J.T @ J + 0.01 * np.eye(self.n_joints)) @ J.T
                q_dot = J_pinv @ nu_desired
                return np.clip(q_dot, -q_dot_limits, q_dot_limits)
                
        except Exception as e:
            print(f"Controller error: {e}")
            return np.zeros(self.n_joints)

def create_workspace_visual(env, workspace):
    """Create workspace visualization"""
    x_min, x_max = workspace.x_min, workspace.x_max
    y_min, y_max = workspace.y_min, workspace.y_max
    z_min, z_max = workspace.z_min, workspace.z_max
    
    wall_thickness = 0.02
    alpha = 0.4
    wall_color = [0.8, 0.1, 0.1, alpha]
    
    walls = []
    
    # Wall specifications: [center, size]
    wall_specs = [
        # Right wall (front boundary - x_max)
        [[x_max + wall_thickness/2, (y_min+y_max)/2, (z_min+z_max)/2], 
         [wall_thickness, y_max-y_min, z_max-z_min]],
        
        # # Left wall (back boundary - x_min) 
        # [[x_min - wall_thickness/2, (y_min+y_max)/2, (z_min+z_max)/2],
        #  [wall_thickness, y_max-y_min, z_max-z_min]],
        
        # Front wall (left boundary - y_max)
        [[(x_min+x_max)/2, y_max + wall_thickness/2, (z_min+z_max)/2],
         [x_max-x_min, wall_thickness, z_max-z_min]],
        
        # Back wall (right boundary - y_min)
        [[(x_min+x_max)/2, y_min - wall_thickness/2, (z_min+z_max)/2],
         [x_max-x_min, wall_thickness, z_max-z_min]],
        
        # Floor
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

# Main simulation
if __name__ == "__main__":
    # Initialize Swift environment
    env = swift.Swift()
    env.launch(realtime=True)
    
    # Create Panda robot
    panda = rtb.models.Panda()
    
    # Set to a reasonable starting configuration
    panda.q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.0, 0.0])
    
    # Add robot to environment
    env.add(panda)
    
    # Define workspace
    workspace = Workspace(
        x_limits=(0.0, 0.7),    # 0.5m depth
        y_limits=(-0.8, 0.8),   # 0.6m width  
        z_limits=(0.01, 0.8)    # 0.55m height
    )
    
    # Add workspace visualization
    walls = create_workspace_visual(env, workspace)
    
    # Create controller
    controller = NEOController(panda, workspace)
    
    # Set target pose (reachable and inside workspace)
    T_desired = SE3.Trans(0.45, 0.35, 0.08) * SE3.RPY(np.pi, 0, 0)
    
    # Add target visualization
    target = sg.Sphere(radius=0.025)
    target.T = T_desired
    target.color = [0, 1, 0, 0.8]
    env.add(target)
    
    # Simulation parameters
    dt = 0.05
    max_iterations = 400
    tolerance = 0.03
    
    # Step environment
    env.step()
    
    print(f"Starting simulation...")
    print(f"Initial position: {panda.fkine(panda.q).t}")
    print(f"Target position: {T_desired.t}")
    
    # Control loop
    positions = []
    errors = []
    
    try:
        for i in range(max_iterations):
            q = panda.q.copy()
            
            # Solve for velocities
            q_dot = controller.solve(q, T_desired)
            
            # Update robot
            panda.q = q + q_dot * dt
            
            # Log progress
            T_current = panda.fkine(panda.q)
            positions.append(T_current.t.copy())
            
            error = np.linalg.norm(T_current.t - T_desired.t)
            errors.append(error)
            
            if error < tolerance:
                print(f"SUCCESS: Reached target in {i} iterations!")
                break
            
            if i % 20 == 0:
                print(f"Iter {i:3d}: error={error:.4f}m, |q_dot|_max={np.max(np.abs(q_dot)):.3f}")
            
            # Step simulation
            env.step()
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("Simulation interrupted")
    
    # Results
    final_error = errors[-1] if errors else float('inf')
    final_pos = positions[-1] if positions else [0, 0, 0]
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Iterations: {len(errors)}")
    print(f"Final error: {final_error:.4f}m")
    print(f"Target: {T_desired.t}")
    print(f"Actual: {final_pos}")
    print(f"Success: {final_error < tolerance}")
    print(f"In workspace: {workspace.is_inside(final_pos)}")
    
    print(f"\nPress Enter to exit...")
    input()
    env.close()