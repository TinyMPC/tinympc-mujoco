import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from src.rho_adapter import RhoAdapter
from utils.visualization import visualize_trajectory, plot_iterations, plot_rho_history, plot_all_metrics, plot_state_and_costs, save_metrics, plot_comparisons, plot_hover_iterations_comparison, plot_paper_hover_iterations
from utils.hover_simulation import simulate_with_controller
from scipy.spatial.transform import Rotation as spRot
import matplotlib.pyplot as plt
import argparse

def compute_hover_error(x_all, xg):
    """Compute L2 tracking error over the hover trajectory"""
    errors = []
    for x in x_all:
        pos_error = np.linalg.norm(x[0:3] - xg[0:3])
        errors.append(pos_error)
    return np.mean(errors), np.max(errors), errors

def parse_args():
    parser = argparse.ArgumentParser(description='Hover control using TinyMPC')
    parser.add_argument('--adapt', action='store_true', 
                        help='Enable rho adaptation')
    parser.add_argument('--recache', action='store_true', 
                        help='Enable cache recomputation')
    parser.add_argument('--wind', action='store_true', 
                        help='Enable wind disturbance')
    parser.add_argument('--heuristic', action='store_true',
                        help='Use heuristic rho adaptation')
    parser.add_argument('--plot-comparison', action='store_true',
                        help='Plot comparison between adaptive and fixed runs')

    parser.add_argument('--plot-comparison-wind', action='store_true',
                        help='Plot comparison between wind and no-wind cases')
    
    parser.add_argument('--plot-iterations-comparison', action='store_true',
                        help='Plot comparison of iterations between methods')
    
    parser.add_argument('--plot-paper', action='store_true',
                        help='Generate paper-ready hover iterations plot')
    
    return parser.parse_args()

def main(use_rho_adaptation=False, use_recaching=False, use_wind=False, use_heuristic=False, rho_update_freq=1):
    """Main function for hover example"""
    print("\nStarting hover simulation with:")
    print(f"- Rho adaptation: {'enabled' if use_rho_adaptation else 'disabled'}")
    print(f"- Cache recomputation: {'enabled' if use_recaching else 'disabled'}")
    print(f"- Wind disturbance: {'enabled' if use_wind else 'disabled'}")
    
    # Create quadrotor instance
    quad = QuadrotorDynamics()

    # Initialize goal state
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    ug = quad.hover_thrust

    # Get linearized system
    A, B = quad.get_linearized_dynamics(xg, ug)

    # Initial state (offset from hover)
    x0 = np.copy(xg)
    x0[0:3] += np.array([0.2, 0.2, -0.2])
    x0[3:7] = quad.rptoq(np.array([1.0, 0.0, 0.0]))

    # Cost matrices
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)

    # print(Q)
    # print(R)



    # Setup MPC
    N = 10
    initial_rho = getattr(main, 'last_rho', 85.0)  # Default 85.0 if first run
    
    if use_rho_adaptation:
        print(f"Using warm-started rho: {initial_rho}")
        rho_adapter = RhoAdapter(
            rho_base=initial_rho, 
            rho_min=60.0, 
            rho_max=100.0,
            mode = 'hover'
        )
    else:
        rho_adapter = None
    
    # Initialize MPC with rho adapter
    mpc = TinyMPC(
        A=A,
        B=B,
        Q=Q,
        R=R,
        Nsteps=N,
        rho=initial_rho,
        rho_adapter=rho_adapter,
        recache=use_recaching,
        mode='hover'
    )


    # Set bounds
    u_max = [1.0-ug[0]] * quad.nu
    u_min = [-ug[0]] * quad.nu
    x_max = [1000.] * quad.nx
    x_min = [-1000.] * quad.nx
    mpc.set_bounds(u_max, u_min, x_max, x_min)

    # Set nominal trajectory
    R0 = spRot.from_quat(qg)
    eulerg = R0.as_euler('zxy')
    xg_euler = np.hstack((eulerg, xg[4:]))
    x_nom = np.tile(0*xg_euler, (N,1)).T
    u_nom = np.tile(ug, (N-1,1)).T

    # Run simulation
    simulation_result = simulate_with_controller(
        x0=x0,
        x_nom=x_nom,
        u_nom=u_nom,
        mpc=mpc,
        quad=quad,
        NSIM=100,
        use_wind=use_wind
    )
    
    # Unpack results
    

    if use_rho_adaptation:
        x_all, u_all, iterations, rho_history, metrics = simulation_result
    else:
        x_all, u_all, iterations, _, metrics = simulation_result
        rho_history = None
    
    # Now you can access the metrics separately
    trajectory_costs = metrics['trajectory_costs']
    control_efforts = metrics['control_efforts']

        

    # Compute and display tracking error
    avg_error, max_error, errors = compute_hover_error(x_all, xg)
    print("\nTracking Error Statistics:")
    print(f"Average L2 Error: {avg_error:.4f} meters")
    print(f"Maximum L2 Error: {max_error:.4f} meters")

    # Save data
    data_dir = Path('../data')
        
    # Create ALL necessary directories (including both old and new)
    for dir_name in ['iterations', 'rho_history', 'trajectory_costs', 'control_efforts', 
                    'costs', 'violations']:  # Added new directories while keeping old ones
        (data_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # Modify the suffix to include rho update frequency
    suffix = '_normal'
    if use_rho_adaptation:
        suffix = f'_adaptive_freq_{rho_update_freq}'
        if use_heuristic:
            suffix += '_heuristic'
    if use_wind:
        suffix += '_wind'
    if use_recaching:
        suffix += f'_recache_freq_{rho_update_freq}'
    suffix += f'_hover'

    # Create paper_plots directory
    paper_plots_dir = data_dir / 'paper_plots'
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    # Save iterations data to paper_plots directory
    np.savetxt(paper_plots_dir / f"iterations{suffix}.txt", iterations)
    if use_rho_adaptation:
        np.savetxt(paper_plots_dir / f"rho_history{suffix}.txt", rho_history)

    # Save metrics to files
    np.savetxt(data_dir / 'iterations' / f"traj{suffix}.txt", iterations)
    if use_rho_adaptation:
        np.savetxt(data_dir / 'rho_history' / f"traj{suffix}.txt", rho_history)
    np.savetxt(data_dir / 'trajectory_costs' / f"traj{suffix}.txt", metrics['trajectory_costs'])
    np.savetxt(data_dir / 'control_efforts' / f"traj{suffix}.txt", metrics['control_efforts'])
    np.savetxt(data_dir / 'costs' / f"costs{suffix}.txt", metrics['solve_costs'])
    np.savetxt(data_dir / 'violations' / f"violations{suffix}.txt", metrics['violations'])

    visualize_trajectory(x_all, u_all, dt=quad.dt)

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

    # if args.plot_iterations_comparison:
    #     plot_hover_iterations_comparison()

if __name__ == "__main__":
    args = parse_args()
    
    if args.plot_paper:
        from utils.visualization import plot_paper_figures
        plot_paper_figures()
    elif args.plot_comparison or args.plot_comparison_wind:
        plot_comparisons(traj_type='hover', 
                        compare_type='wind' if args.plot_comparison_wind else 'normal')
    else:
        main(use_rho_adaptation=args.adapt,
             use_recaching=args.recache,
             use_wind=args.wind,
             use_heuristic=args.heuristic,
             rho_update_freq=1)  # Add frequency parameter here

       
