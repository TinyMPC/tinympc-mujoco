import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from utils.visualization import visualize_trajectory, plot_iterations, plot_rho_history, plot_costs_comparison, plot_violations_comparison, save_metrics, plot_all_metrics, plot_state_and_costs, plot_comparisons, plot_paper_trajectory_comparison
from utils.traj_simulation import simulate_with_controller
from scipy.spatial.transform import Rotation as spRot
from utils.reference_trajectories import Figure8Reference
from src.rho_adapter import RhoAdapter
import matplotlib.pyplot as plt
import argparse

def compute_tracking_error(x_all, trajectory, dt):
    """Compute L2 tracking error over the trajectory"""
    errors = []
    times = np.arange(len(x_all)) * dt
    
    for t, x in zip(times, x_all):
        x_ref = trajectory.generate_reference(t)
        pos_error = np.linalg.norm(x[0:3] - x_ref[0:3])
        errors.append(pos_error)
    
    return np.mean(errors), np.max(errors), errors

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt', action='store_true', 
                        help='Enable rho adaptation')
    parser.add_argument('--recache', action='store_true', 
                        help='Enable cache recomputation')
    parser.add_argument('--wind', action='store_true', 
                        help='Enable wind disturbance')
    parser.add_argument('--plot-comparison', action='store_true',
                        help='Plot comparison between adaptive and fixed runs')
    parser.add_argument('--plot-comparison-wind', action='store_true',
                        help='Plot comparison between wind and no-wind cases')
    parser.add_argument('--straight', action='store_true',
                        help='Use straight line trajectory')
    parser.add_argument('--curve', action='store_true',
                        help='Use curved trajectory')
    parser.add_argument('--heuristic', action='store_true',
                        help='Use heuristic rho adaptation')
    parser.add_argument('--plot-paper', action='store_true',
                        help='Generate paper-quality comparison plots')
    parser.add_argument('--wind-seed', type=int, default=42,
                        help='Random seed for wind generation')
    return parser.parse_args()

def main(use_rho_adaptation=False, use_recaching=False, use_wind=False, traj_type='full', use_heuristic=False, wind_seed=42):
    # Create quadrotor instance
    quad = QuadrotorDynamics()

    # Initialize hover state (for linearization)
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    ug = quad.hover_thrust

    # Get linearized system
    A, B = quad.get_linearized_dynamics(xg, ug)

    x0 = np.copy(xg)
    
    # Trajectory parameters
    amplitude = 0.5
    w = 2*np.pi/3.7

    #with wind lets go slow 
    #w = 2*np.pi/3.9
    trajectory = Figure8Reference(A=amplitude, w=w, segment_type=traj_type)
    
    # Get the initial reference point and set initial state
    x_ref_0 = trajectory.generate_reference(0.0)
    x0[0:3] = x_ref_0[0:3]     # Position
    x0[3:7] = qg               # Level quaternion
    x0[7:10] = x_ref_0[6:9]    # All velocities (not just x and z)
    x0[10:13] = np.zeros(3)    # Zero angular velocity

    # Cost matrices
    max_dev_x = np.array([
        0.01, 0.01, 0.01,    # position (tighter)
        0.5, 0.5, 0.05,      # attitude 
        0.5, 0.5, 0.5,       # velocity
        0.7, 0.7, 0.5        # angular velocity
    ])
    max_dev_u = np.array([0.1, 0.1, 0.1, 0.1])

    Q = np.diag(1.0 / (max_dev_x**2))  # Much higher position weights
    R = np.diag(1.0 / (max_dev_u**2))

    # Setup PC
    N = 15

    # Initialize rho (keep the last value from previous run)
    initial_rho = getattr(main, 'last_rho', 5.0)  
    
    if use_rho_adaptation:
        
        rho_adapter = RhoAdapter(
            rho_base=initial_rho, 
            rho_min=1.0, 
            rho_max=200.0,
            method="heuristic" if use_heuristic else "analytical",
            clip = True,
            mode = 'traj'
        )
    else:
        rho_adapter = None

    # Initialize MPC
    mpc = TinyMPC(
        A=A,
        B=B,
        Q=Q,
        R=R,
        Nsteps=N,
        rho=initial_rho,
        rho_adapter= rho_adapter,
        recache = use_recaching,
        mode = 'traj'
    )

   
    if use_rho_adaptation:
        mpc.rho_adapter.initialize_derivatives(mpc.cache)
       

    # Set bounds relative to hover thrust
    u_max = np.array([0.3] * quad.nu)  # Allow 30% above hover
    u_min = np.array([-0.3] * quad.nu)  # Allow 30% below hover
    x_max = [2.0] * 3 + [1.0] * 3 + [2.0] * 3 + [2.0] * 3  # [pos, att, vel, omega]
    x_min = [-x for x in x_max]
    mpc.set_bounds(u_max, u_min, x_max, x_min)

    # Create trajectory reference (keep your original parameters)
    x_nom = np.zeros((quad.nx, mpc.N))
    u_nom = np.zeros((quad.nu, mpc.N-1))

    u_nom[:] = ug.reshape(-1,1)  # hover thrust

    # Initialize nominal states from trajectory
    x_nom = np.zeros((quad.nx, mpc.N))
    for i in range(mpc.N):
        x_nom[:,i] = trajectory.generate_reference(i*quad.dt)


    # Run simulation
    try:
        print(f"Starting trajectory tracking simulation with:")
        print(f"- Rho adaptation: {'enabled' if use_rho_adaptation else 'disabled'}")
        print(f"- Cache recomputation: {'enabled' if use_recaching else 'disabled'}")

        simulation_result = simulate_with_controller(
            x0=x0,
            x_nom=x_nom,
            u_nom=u_nom,
            mpc=mpc,
            quad=quad,
            trajectory=trajectory,
            dt_sim=0.01,
            dt_mpc=0.02,
            NSIM=200,
            use_wind=use_wind,
            
        )

        # Unpack results based on whether we're using rho adaptation
        if use_rho_adaptation:
            x_all, u_all, iterations, rho_history, metrics = simulation_result
        else:
            x_all, u_all, iterations, _, metrics = simulation_result
            rho_history = None


        trajectory_costs = metrics['trajectory_costs']
        control_efforts = metrics['control_efforts']

        

        # Compute and display tracking error
        avg_error, max_error, errors = compute_tracking_error(x_all, trajectory, quad.dt)
        print("\nTracking Error Statistics:")
        print(f"Average L2 Error: {avg_error:.4f} meters")
        print(f"Maximum L2 Error: {max_error:.4f} meters")

        # Save data
        data_dir = Path('../data')
        
        # Create ALL necessary directories (including both old and new)
        for dir_name in ['iterations', 'rho_history', 'trajectory_costs', 'control_efforts', 
                        'costs', 'violations']:  # Added new directories while keeping old ones
            (data_dir / dir_name).mkdir(parents=True, exist_ok=True)

        # Determine suffix
        suffix = '_normal'
        if use_rho_adaptation:
            suffix = '_adaptive'
            if use_heuristic:
                suffix += '_heuristic'
        if use_wind:
            suffix += '_wind'
        if use_recaching:
            suffix += '_recache'
        suffix += f'_{traj_type}'

        # Save ALL metrics (both old and new)
        # Original metrics
        np.savetxt(data_dir / 'iterations' / f"traj{suffix}.txt", iterations)
        if use_rho_adaptation:
            np.savetxt(data_dir / 'rho_history' / f"traj{suffix}.txt", rho_history)
        np.savetxt(data_dir / 'trajectory_costs' / f"traj{suffix}.txt", metrics['trajectory_costs'])
        np.savetxt(data_dir / 'control_efforts' / f"traj{suffix}.txt", metrics['control_efforts'])

        # Additional metrics for comparison
        np.savetxt(data_dir / 'costs' / f"costs{suffix}.txt", metrics['solve_costs'])
        np.savetxt(data_dir / 'violations' / f"violations{suffix}.txt", metrics['violations'])

        visualize_trajectory(x_all, u_all, trajectory=trajectory, dt=quad.dt)


        # Update how we call plot_all_metrics
        # plot_all_metrics(suffix=suffix, use_rho_adaptation=use_rho_adaptation, dt=quad.dt)
        # plot_state_and_costs(suffix=suffix, use_rho_adaptation=use_rho_adaptation)

        print("\nSimulation completed successfully!")
        print(f"Average iterations per step: {np.mean(iterations):.2f}")
        print(f"Total iterations: {sum(iterations)}")
        print(f"Average trajectory cost: {np.mean(metrics['trajectory_costs']):.4f}")
        print(f"Average control effort: {np.mean(metrics['control_efforts']):.4f}")

        # Store final rho for next run
        if use_rho_adaptation and rho_history:
            main.last_rho = rho_history[-1]
            print(f"Saved rho {main.last_rho} for next run")

        # After simulation, save trajectory data
        data_dir = Path('../data')
        (data_dir / 'trajectories').mkdir(parents=True, exist_ok=True)
        
        # Save position data
        trajectory_data = np.array([x[0:3] for x in x_all])
        np.savetxt(data_dir / 'trajectories' / f'traj{suffix}.txt', trajectory_data)
        
        # Save reference trajectory (only needs to be done once)
        ref_path = data_dir / 'trajectories' / 'reference_trajectory.txt'
        if not ref_path.exists():
            t = np.arange(len(x_all)) * quad.dt
            ref_data = np.array([trajectory.generate_reference(ti)[0:3] for ti in t])
            np.savetxt(ref_path, ref_data)

       

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise



if __name__ == "__main__":
    args = parse_args()
    
    if args.plot_paper:
        plot_paper_trajectory_comparison()
    else:
        # Determine trajectory type
        if args.straight:
            traj_type = 'straight'
        elif args.curve:
            traj_type = 'curve'
        else:
            traj_type = 'full'  # default
        
        if args.plot_comparison or args.plot_comparison_wind:
            plot_comparisons(traj_type=traj_type, 
                            compare_type='wind' if args.plot_comparison_wind else 'normal')
        else:
            main(use_rho_adaptation=args.adapt,
                 use_recaching=args.recache,
                 use_wind=args.wind,
                 traj_type=traj_type,
                 use_heuristic=args.heuristic,
                 wind_seed=args.wind_seed)