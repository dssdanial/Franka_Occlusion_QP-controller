# Franka_Occlusion_QP-controller
QP-based control framework that unifies task-space tracking, workspace boundaries, and LOS constraints into a single optimization problem.



## Conceptual distinction (what walls vs obstacles represent)

### Walls / Workspace bounds
Represent global, static limits of where the robot is allowed to be.
- Usually known a priori, fixed, and considered hard safety boundaries (you really don’t want the robot to cross them).
- Geometrically convenient: planes/half-spaces, easy to compute distance to.

### Obstacles
- Represent local, possibly dynamic objects (boxes, humans, moving robots).
- May be uncertain, tracked from sensors, and often treated as soft constraints (you can allow small relaxation/slack if necessary to preserve feasibility).

Treating them differently allows reliable, safe behavior.





<img width="887" height="788" alt="image" src="https://github.com/user-attachments/assets/8920a31b-fe72-4a99-9ee5-3e69a43abf0b" />

## Result


https://github.com/user-attachments/assets/d0358c23-833a-4f18-9473-529f0c29b155

