import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics

# Global reference states
rg = np.array([0.0, 0, 0.0])
qg = np.array([1.0, 0, 0, 0])
vg = np.zeros(3)
omgg = np.zeros(3)
xg = np.hstack([rg, qg, vg, omgg])

# Get hover thrust from QuadrotorDynamics parameters
quad = QuadrotorDynamics()
uhover = quad.hover_thrust
Nx = quad.nx

def delta_x_quat(x_curr, x_ref=None):
    """Compute state error, either from hover or trajectory reference"""
    if x_ref is None:
        # Hover case
        pos_ref = rg
        vel_ref = vg
        omg_ref = omgg
        q_ref = qg
    else:
        # Trajectory following case
        pos_ref = x_ref[0:3]
        vel_ref = x_ref[6:9]  
        omg_ref = x_ref[9:12]
        q_ref = qg 
    
    # Current state
    q = x_curr[3:7]/np.linalg.norm(x_curr[3:7])  # Normalize quaternion
    
    # Compute attitude error in reduced form (3D)
    phi = QuadrotorDynamics.qtorp(QuadrotorDynamics.L(q_ref).T @ q)
    
    # Compute full error state
    delta_x = np.hstack([
        x_curr[0:3] - pos_ref,    # Position error
        phi,                       # Attitude error (reduced)
        x_curr[7:10] - vel_ref,   # Velocity error
        x_curr[10:13] - omg_ref   # Angular velocity error
    ])
    
    
    return delta_x

def tinympc_controller(x_curr, x_nom, u_nom, mpc, t=None, trajectory=None, quad=None):
    """MPC controller for trajectory following"""
    # Generate current reference for error computation
    x_ref = trajectory.generate_reference(t)
    
    # Compute error state
    delta_x = delta_x_quat(x_curr, x_ref)
    
    # Initialize MPC problem with zero reference since we're in error coordinates
    x_init = np.copy(mpc.x_prev)
    x_init[:,0] = delta_x
    u_init = np.copy(mpc.u_prev)
    
    
    x_out, u_out, status, k = mpc.solve_admm(x_init, u_init) 
    
    return u_nom[:,0] + u_out[:,0], k, status, x_out  # Return all 4 values


def shift_steps(x_nom, u_nom, x_curr, goals=None, dt=None):
    """Shift trajectory steps and update start state
        
    Args:
        x_nom (np.ndarray): Nominal state trajectory (nx x N)
        u_nom (np.ndarray): Nominal input trajectory (nu x N-1)
        x_curr (np.ndarray): Current state
        goals (np.ndarray, optional): Reference trajectory points (nx x N)
        dt (float, optional): Time step for integration
    """
    # 1. Shift existing trajectories
    x_nom[:, :-1] = x_nom[:, 1:]
    u_nom[:, :-1] = u_nom[:, 1:]
        
    # 2. Convert full state to linearized state if needed
    if x_curr.shape[0] == 13:  # Full state
        x_lin = x_curr[[0,1,2, 4,5,6, 7,8,9, 10,11,12]]  # Skip w (index 3)
    else:  # Already linearized
        x_lin = x_curr
        
    # 3. Update start state
    x_nom[:, 0] = x_lin
        
    # 4. Update final point using one of these methods:
    if goals is not None:
        # Method A: Use reference trajectory
        x_nom[:, -1] = goals[:, -1]
        u_nom[:, -1] = u_nom[:, -2]  # Copy last input
    else:
        # Method B: Copy previous point
        x_nom[:, -1] = x_nom[:, -2]
        u_nom[:, -1] = u_nom[:, -2]
            
        
    return x_nom, u_nom


# def generate_wind(t, seed=42):
#     """Generate time-varying wind disturbance according to the paper's model
#     w(t) = m * [cos(θ(t)), sin(θ(t)), η(t)]
#     where:
#     - m = 25.5 m/s^2 (wind acceleration magnitude)
#     - θ(t) ~ U(0, 2π) varies with time
#     - η(t) ~ U(-0.3, 0.3) varies with time
#     """
#     # Set random seed for this timestep
#     rng = np.random.RandomState(seed + int(t/0.1))  # Change seed every 0.1s
    
#     m = 25.5  # wind magnitude in m/s^2
#     theta = rng.uniform(0, 2*np.pi)  # random horizontal direction
#     eta = rng.uniform(-0.3, 0.3)     # random vertical component
    
#     wind = m * np.array([
#         np.cos(theta),  # x component
#         np.sin(theta),  # y component
#         eta            # z component
#     ])
    
#     return wind

def generate_wind(t):
    """Generate random wind disturbances"""
    # Generate random wind events (20% chance of wind at any time)
    if np.random.random() < 0.5:
        # Random wind direction and magnitude
        # wind_magnitude = np.random.uniform(5.5, 7.5)
        # wind_direction = np.random.uniform(0, 2*np.pi)

        wind_magnitude = 25.5

        wind_direction = np.pi/2

        wind = wind_magnitude * np.array([
            np.cos(wind_direction),  # x component
            np.sin(wind_direction),  # y component
            np.random.uniform(-0.3, 0.3)  # z component
        ])
    else:
        wind = np.zeros(3)

    return wind

def simulate_with_controller(x0, x_nom, u_nom, mpc, quad, trajectory, 
                           dt_sim=0.002,   
                           dt_mpc=0.02,    
                           NSIM=400,
                           use_wind=False):  # Add wind flag parameter
    """Simulate system with MPC controller
    
    Args:
        x0: Initial state
        x_nom: Initial nominal state trajectory
        u_nom: Initial nominal input trajectory
        mpc: TinyMPC controller instance
        quad: Quadrotor dynamics
        trajectory: Reference trajectory generator
        dt_sim: Simulation timestep (default: 0.002s)
        dt_mpc: MPC update interval (default: 0.02s)
        NSIM: Number of MPC steps (default: 400)
    """
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    iterations = []
    rho_history = [] if mpc.rho_adapter is not None else None
    current_time = 0.0
    
    # Add new metrics dictionaries
    metrics = {
        'trajectory_costs': [],
        'control_efforts': [],
        'solve_costs': [],      # Cost for each MPC solve
        'violations': [],       # Constraint violations for each solve
        'iterations': []        # Already tracking this
    }
    
    # Compute simulation steps per MPC update
    n_sim_steps = int(dt_mpc / dt_sim)

    for i in range(NSIM):
        # Generate goals for entire horizon
        goals = np.zeros((mpc.nx, mpc.N))
        for j in range(mpc.N):
            future_time = current_time + j*dt_mpc
            goals[:,j] = trajectory.generate_reference(future_time)
        
        # Update nominal trajectory using dt_mpc
        for j in range(mpc.N):
            x_nom[:,j] = trajectory.generate_reference(future_time)
            if j < mpc.N-1:
                #u_nom[:,j] = trajectory.compute_nominal_control(future_time, quad)
                u_nom[:,j] = quad.hover_thrust.reshape(-1)
        # Run MPC step
        u, k, status, x_traj = tinympc_controller(x_curr, x_nom, u_nom, mpc, 
                                         current_time, trajectory, quad)
        
        # Get current reference and compute error state (using delta_x_quat)
        current_ref = trajectory.generate_reference(current_time)
        state_error = delta_x_quat(x_curr, current_ref)  # This already handles quaternion properly
        
        # Compute costs
        state_cost = float(state_error.T @ mpc.cache['Q'] @ state_error)
        input_cost = float(u.T @ mpc.cache['R'] @ u)
        total_cost = state_cost + input_cost
        metrics['solve_costs'].append([state_cost, input_cost, total_cost])

        # Compute constraint violations for this solve
        u_violation = np.maximum(0, np.maximum(
            np.abs(u) - mpc.umax, 
            mpc.umin - np.abs(u)
        ))
        
        # Convert state to reduced form for constraint checking
        x_reduced = np.hstack([
            x_curr[0:3],           # Position
            quad.qtorp(x_curr[3:7]), # Convert quaternion to rpy
            x_curr[7:10],          # Velocity
            x_curr[10:13]          # Angular velocity
        ])
        x_violation = np.maximum(0, np.maximum(
            np.abs(x_reduced) - mpc.xmax, 
            mpc.xmin - np.abs(x_reduced)
        ))
        
        metrics['violations'].append([np.sum(u_violation), np.sum(x_violation)])
        metrics['iterations'].append(k)

        # Add detailed printing for solves

        if np.isnan(state_cost) or np.isnan(input_cost) or np.isnan(total_cost):

            #print time in simulation 
            print(f"Time in simulation: {current_time:.6f}")
            exit(0)
        
        print(f"\n=== Solve {i+1} Details ===")
        print(f"Iterations needed: {k}")
            
        print("\nAccuracy:")
        print(f"Position error: {np.linalg.norm(x_curr[0:3] - current_ref[0:3]):.6f}")
        print(f"Attitude error: {np.linalg.norm(quad.qtorp(x_curr[3:7])):.6f}")
            
        print("\nCosts:")
        print(f"State cost: {state_cost:.6f}")
        print(f"Input cost: {input_cost:.6f}")
        print(f"Total cost: {total_cost:.6f}")
            
        print("\nConstraint Violations:")
        print(f"Input constraints: {np.sum(u_violation):.6f}")
        print(f"State constraints: {np.sum(x_violation):.6f}")
            
        # if mpc.rho_adapter:
        print(f"\nRho value: {mpc.cache['rho']:.6f}")
            
        print("=" * 50)

        # Simulate with finer timestep
        for _ in range(n_sim_steps):
            # Only generate and apply wind if use_wind is True
            wind_vec = generate_wind(current_time) if use_wind else np.zeros(3)
            x_curr = quad.dynamics_rk4(x_curr, u, dt=dt_sim, wind_vec=wind_vec)
            current_time += dt_sim
        
        # Store basic results
        x_all.append(x_curr)
        u_all.append(u)
        iterations.append(k)
        
        # Store additional metrics
        current_ref = trajectory.generate_reference(current_time)
        pos_error = np.linalg.norm(x_curr[0:3] - current_ref[0:3])    # Position error
        att_error = np.linalg.norm(x_curr[3:7] - current_ref[3:7])    # Attitude error
        metrics['trajectory_costs'].append(pos_error + att_error)      # Tracking cost
        metrics['control_efforts'].append(np.sum(np.abs(u[1:])))  # Sum of absolute motor commands

        
        
        # Update rho if using adaptation and only every 10th step
        if mpc.rho_adapter is not None:
            rho_history.append(mpc.cache['rho'])
            
        # Shift nominal trajectories with goals
        x_nom, u_nom = shift_steps(x_nom, u_nom, x_curr, goals=goals)

    # Convert to numpy arrays for easier analysis
    x_all = np.array(x_all)
    u_all = np.array(u_all)
    
    # Return format matching your existing code
    if mpc.rho_adapter is not None:
        return x_all, u_all, iterations, rho_history, metrics
    else:
        return x_all, u_all, iterations, None, metrics  # Add None for rho_history