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

# Safety Set and Related Parameters
u_min = -7.0  # Minimum acceleration (m/s^2)
nu_max = 3.3   # Maximum acceleration (m/s^2)
phi_min = -ca.pi/4  # Minimum steering angle (rad)
phi_max = ca.pi/4  # Maximum steering angle (rad)

# Vehicle State Bounds
v_min = 15  # Minimum speed (m/s)
v_max = 35  # Maximum speed (m/s)

# Safe Region Bounds
safety_set = {
    'x_bounds': (-100, 100),  # Longitudinal range (m)
    'y_bounds': (-l/2, 3*l/2),  # Lateral range (m)
    'v_bounds': (v_min, v_max),
    'u_bounds': [(nu_min, phi_min), (nu_max, phi_max)],
}

# Randomized HDV Parameters (σ1, σ2) and Disturbances (ε1, ε2, ε3, ε4)
def get_HDV_noise():
    sigma_1 = np.random.uniform(0.9, 1.1)
    sigma_2 = np.random.uniform(0.9, 1.1)
    epsilon_1 = np.random.uniform(-0.7, 0.7)
    epsilon_2 = np.random.uniform(-0.5, 0.5)
    epsilon_3 = np.random.uniform(-0.5, 0.5)
    epsilon_4 = np.random.uniform(-0.7, 0.7)
    return sigma_1, sigma_2, epsilon_1, epsilon_2, epsilon_3, epsilon_4

def HDV_dynamics(x_H, u_H):
    """ HDV dynamics with stochastic behavior """
    sigma_1, sigma_2, epsilon_1, epsilon_2, epsilon_3, epsilon_4 = get_HDV_noise()
    v_H = x_H[3]
    theta_H = x_H[2]
    u_acc_H = u_H[0]
    u_steer_H = u_H[1]
    
    dxdt = ca.vertcat(
        v_H * ca.cos(theta_H) * sigma_1 + epsilon_1,
        v_H * ca.sin(theta_H) * sigma_2 + epsilon_2,
        v_H / L_w * u_steer_H + epsilon_3,
        u_acc_H + epsilon_4
    )
    return dxdt

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
    opti.subject_to(safety_constraint(x_C_next, x_H, a_C, b_C) >= 0)  # CAV-HDV constraint
    opti.subject_to(safety_constraint(x_C_next, x_U, a_C, b_C) >= 0)  # CAV-Obstacle constraint
    
    # Control bounds
    opti.subject_to(nu_min <= u_C[0] <= nu_max)  # Acceleration limits
    opti.subject_to(phi_min <= u_C[1] <= phi_max)  # Steering limits
    
    # Speed limits
    opti.subject_to(v_min <= x_C_next[3] <= v_max)
    
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
    x_H = x_H + dt * HDV_dynamics(ca.DM(x_H), ca.DM(np.array([0, 0]))).full().flatten()  # Update HDV state
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
