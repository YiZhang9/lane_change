import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Simulation Parameters
dt = 0.05  # Time step (s)
t_max = 15  # Simulation time (s)
n_steps = int(t_max / dt)

# Vehicle Parameters
L_w = 2.5  # Vehicle wheelbase (m)
v_d = 30   # Desired speed (m/s)
a_C = 0.6  # Reaction time scaling factor for CAVs
b_C = 0.1  # Minor axis scaling for safe region
l = 4.0  # Lane width (m)

# Control limits
u_min = -3.3  # Minimum acceleration (m/s^2)
nu_max = 7.0  # Maximum acceleration (m/s^2)
phi_min = -ca.pi/4  # Minimum steering angle (rad)
phi_max = ca.pi/4  # Maximum steering angle (rad)

# Vehicle State Bounds
v_min = 15  # Minimum speed (m/s)
v_max = 35  # Maximum speed (m/s)
def vehicle_dynamics(x, u):
    """CAV Dynamics Model"""
    theta = x[2]
    v = x[3]
    u_acc = u[0]  # Acceleration
    u_steer = u[1]  # Steering angle
    
    dxdt = ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        v / L_w * u_steer,
        u_acc
    )
    return dxdt

def safety_constraint(x1, x2, a, b):
    """Elliptical Safety Constraint between vehicles"""
    return ((x2[0] - x1[0])**2 / (a * x1[3])**2) + ((x2[1] - x1[1])**2 / (b * x1[3])**2) - 1

def solve_qp(x_C, x_H, x_U):
    """Solve QP for CAV C"""
    opti = ca.Opti()
    u_C = opti.variable(2)  # Control variables (acceleration, steering)
    
    # Cost function (minimize deviation from desired speed and energy usage)
    cost = ca.sumsqr(u_C) + ca.sumsqr(x_C[3] - v_d)
    opti.minimize(cost)
    
    # Dynamics constraints
    x_C_next = x_C + dt * vehicle_dynamics(x_C, u_C)
    
    # Safety constraints (CBF-based)
    a, b = 0.6, 0.1  # Safe region parameters
    opti.subject_to(safety_constraint(x_C_next, x_H, a, b) >= 0)  # CAV-HDV constraint
    opti.subject_to(safety_constraint(x_C_next, x_U, a, b) >= 0)  # CAV-Obstacle constraint
    
    # Control bounds
    opti.subject_to(-3.3 <= u_C[0] <= 7)  # Acceleration limits
    opti.subject_to(-ca.pi/4 <= u_C[1] <= ca.pi/4)  # Steering limits
    
    # Solver settings
    opti.solver('ipopt')
    sol = opti.solve()
    return sol.value(u_C)

# Initial Conditions
x_C = np.array([20, 0, 0, 25])  # CAV initial state (x, y, theta, v)
x_H = np.array([10, 4, 0, 28])  # HDV initial state
x_U = np.array([60, 0, 0, 20])  # Static obstacle

dt_sim = []  # Store event-driven time steps
x_C_hist = [x_C.copy()]  # History of CAV states

t = 0
while t < t_max:
    u_C = solve_qp(x_C, x_H, x_U)  # Solve QP for control input
    x_C = x_C + dt * vehicle_dynamics(ca.DM(x_C), ca.DM(u_C)).full().flatten()  # Update state
    x_C_hist.append(x_C.copy())
    dt_sim.append(t)
    t += dt

# Plot Trajectory
x_C_hist = np.array(x_C_hist)
plt.plot(x_C_hist[:, 0], x_C_hist[:, 1], label='CAV Trajectory')
plt.scatter([x_H[0]], [x_H[1]], color='red', label='HDV Initial')
plt.scatter([x_U[0]], [x_U[1]], color='gray', label='Obstacle')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.title('CAV Safe Lane-Changing Trajectory')
plt.show()