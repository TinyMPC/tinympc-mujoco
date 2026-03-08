import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# Set paper-quality plotting parameters
mpl.rcParams.update({
    'font.size': 16,         # Increase base font size
    'axes.labelsize': 18,    # Larger axis labels
    'axes.titlesize': 20,    # Larger titles
    'legend.fontsize': 16,   # Bigger legend font
    'figure.figsize': (12, 8), # Larger figures
    'lines.linewidth': 2,    # Thicker lines
    'figure.dpi': 300,       # Higher DPI for crisp images
    'text.usetex': False     # Don't use LaTeX
})

def visualize_trajectory(x_all, u_all, trajectory=None, dt=0.02, save_path=None):
    """Visualize trajectory with paper-quality formatting in three horizontal subplots"""
    x_all = np.array(x_all)
    u_all = np.array(u_all)
    t = np.arange(len(x_all)) * dt
    t_full = np.arange(200) * dt  # Full trajectory time

    # Find divergence point (position error > 5m or NaN)
    divergence_idx = None
    if trajectory is not None:
        for i, x in enumerate(x_all):
            x_ref = trajectory.generate_reference(t[i])[0:3]
            pos_error = np.linalg.norm(x[0:3] - x_ref)
            if pos_error > 5.0 or np.any(np.isnan(x)):
                divergence_idx = i
                print(f"\nController diverged at t = {t[i]:.2f} seconds")
                print(f"Position error: {pos_error:.2f} meters")
                break
    
    # Trim data if divergence detected
    if divergence_idx is not None:
        x_all = x_all[:divergence_idx]
        u_all = u_all[:divergence_idx]
        t = t[:divergence_idx]

    # Create three horizontal subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # Position subplot
    ax1.plot(t, x_all[:, 0], 'r-', label='x', linewidth=1.5)
    ax1.plot(t, x_all[:, 1], 'g-', label='y', linewidth=1.5)
    ax1.plot(t, x_all[:, 2], 'b-', label='z', linewidth=1.5)
    if trajectory is not None:
        x_ref_full = np.array([trajectory.generate_reference(ti)[0:3] for ti in t_full])
        ax1.plot(t_full, x_ref_full[:, 0], 'r--', label='x ref', linewidth=1.5, alpha=0.7)
        ax1.plot(t_full, x_ref_full[:, 1], 'g--', label='y ref', linewidth=1.5, alpha=0.7)
        ax1.plot(t_full, x_ref_full[:, 2], 'b--', label='z ref', linewidth=1.5, alpha=0.7)
    ax1.set_title('Position', fontsize=14)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position (m)', fontsize=12)
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)

    # Attitude subplot
    ax2.plot(t, x_all[:, 3], 'r-', label='qw', linewidth=1.5)
    ax2.plot(t, x_all[:, 4], 'g-', label='qx', linewidth=1.5)
    ax2.plot(t, x_all[:, 5], 'b-', label='qy', linewidth=1.5)
    ax2.plot(t, x_all[:, 6], 'k-', label='qz', linewidth=1.5)
    ax2.set_title('Attitude', fontsize=14)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Quaternion', fontsize=12)
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)

    # Control subplot
    t_control = t[:len(u_all)]
    ax3.plot(t_control, u_all[:, 0], 'r-', label='u₁', linewidth=1.5)
    ax3.plot(t_control, u_all[:, 1], 'g-', label='u₂', linewidth=1.5)
    ax3.plot(t_control, u_all[:, 2], 'b-', label='u₃', linewidth=1.5)
    ax3.plot(t_control, u_all[:, 3], 'k-', label='u₄', linewidth=1.5)
    ax3.set_title('Control Inputs', fontsize=14)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Control Input', fontsize=12)
    ax3.legend(fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=10)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    # Create side view plot (X-Z plane) as a separate figure
    fig2 = plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()
    
    # Plot actual trajectory
    ax.plot(x_all[:, 0], x_all[:, 2], 'b-', label='Actual', linewidth=2)
    
    # Plot reference trajectory if available
    if trajectory is not None:
        t_full = np.linspace(0, 4, 1000)
        x_ref_full = np.array([trajectory.generate_reference(ti)[0:3] for ti in t_full])
        ax.plot(x_ref_full[:, 0], x_ref_full[:, 2], 'r--', label='Reference', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        path_parts = save_path.split('.')
        side_view_path = f"{path_parts[0]}_side_view.{path_parts[1]}"
        plt.savefig(side_view_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_iterations(iterations):
    """Plot ADMM iterations"""
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, label='Iterations')
    plt.ylabel('Iterations')
    plt.title('ADMM Iterations per Time Step')
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Total iterations: {sum(iterations)}")
    print(f"Average iterations: {np.mean(iterations):.2f}")

def plot_rho_history(rho_history):
    """Plot rho adaptation history (only for adaptive MPC)"""
    plt.figure(figsize=(10, 4))
    plt.plot(rho_history, label='Rho')
    plt.xlabel('Time Step')
    plt.ylabel('Rho Value')
    plt.title('Rho Adaptation History')
    plt.grid(True)
    plt.legend()
    plt.show()
    print(f"Final rho: {rho_history[-1]:.2f}")
    print(f"Rho range: [{min(rho_history):.2f}, {max(rho_history):.2f}]")

def plot_costs_comparison(adaptive_costs, fixed_costs):
    """Plot comparison of costs between adaptive and fixed rho"""
    plt.figure(figsize=(15, 5))
    
    # State Costs
    plt.subplot(131)
    plt.plot(adaptive_costs[:, 0], label='Adaptive', alpha=0.8)
    plt.plot(fixed_costs[:, 0], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.title('State Costs Comparison')
    plt.legend()
    plt.grid(True)
    
    # Input Costs
    plt.subplot(132)
    plt.plot(adaptive_costs[:, 1], label='Adaptive', alpha=0.8)
    plt.plot(fixed_costs[:, 1], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.title('Input Costs Comparison')
    plt.legend()
    plt.grid(True)
    
    # Statistics
    plt.subplot(133)
    stats_text = (
        f'Average State Cost:\n'
        f'Adaptive: {np.mean(adaptive_costs[:, 0]):.3f}\n'
        f'Fixed: {np.mean(fixed_costs[:, 0]):.3f}\n\n'
        f'Average Input Cost:\n'
        f'Adaptive: {np.mean(adaptive_costs[:, 1]):.3f}\n'
        f'Fixed: {np.mean(fixed_costs[:, 1]):.3f}'
    )
    plt.text(0.1, 0.1, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.title('Cost Statistics')
    
    plt.tight_layout()
    plt.show()

def plot_violations_comparison(adaptive_violations, fixed_violations):
    """Plot comparison of constraint violations between adaptive and fixed rho"""
    plt.figure(figsize=(15, 5))
    
    # Input Violations
    plt.subplot(131)
    plt.plot(adaptive_violations[:, 0], label='Adaptive', alpha=0.8)
    plt.plot(fixed_violations[:, 0], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.ylabel('Input Constraint Violation')
    plt.title('Input Violations Comparison')
    plt.legend()
    plt.grid(True)
    
    # State Violations
    plt.subplot(132)
    plt.plot(adaptive_violations[:, 1], label='Adaptive', alpha=0.8)
    plt.plot(fixed_violations[:, 1], label='Fixed', alpha=0.8)
    plt.xlabel('MPC Step')
    plt.ylabel('State Constraint Violation')
    plt.title('State Violations Comparison')
    plt.legend()
    plt.grid(True)
    
    # Statistics
    plt.subplot(133)
    stats_text = (
        f'Average Input Violation:\n'
        f'Adaptive: {np.mean(adaptive_violations[:, 0]):.3f}\n'
        f'Fixed: {np.mean(fixed_violations[:, 0]):.3f}\n\n'
        f'Average State Violation:\n'
        f'Adaptive: {np.mean(adaptive_violations[:, 1]):.3f}\n'
        f'Fixed: {np.mean(fixed_violations[:, 1]):.3f}'
    )
    plt.text(0.1, 0.1, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.title('Violation Statistics')
    
    plt.tight_layout()
    plt.show()

def save_metrics(metrics, suffix, data_dir='data'):
    """Save metrics to files"""
    data_dir = Path(data_dir)
    for subdir in ['costs', 'violations', 'iterations']:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    np.savetxt(data_dir / 'costs' / f'costs_{suffix}.txt', metrics['solve_costs'])
    np.savetxt(data_dir / 'violations' / f'violations_{suffix}.txt', metrics['violations'])
    np.savetxt(data_dir / 'iterations' / f'iterations_{suffix}.txt', metrics['iterations'])

def plot_state_and_costs(suffix, use_rho_adaptation=False, data_dir='../data'):
    """Plot state and cost metrics from saved files"""
    data_dir = Path(data_dir)
    
    # Load data from files
    costs = np.loadtxt(data_dir / 'costs' / f'costs{suffix}.txt')
    violations = np.loadtxt(data_dir / 'violations' / f'violations{suffix}.txt')
    
    plt.figure(figsize=(15, 5))
    
    # Plot costs
    plt.subplot(121)
    plt.plot(costs[:, 0], label='State Cost')
    plt.plot(costs[:, 1], label='Input Cost')
    plt.plot(costs[:, 2], label='Total Cost')
    plt.xlabel('MPC Step')
    plt.ylabel('Cost')
    plt.title(f"{'Adaptive' if use_rho_adaptation else 'Fixed'} Rho Costs")
    plt.legend()
    plt.grid(True)
    
    # Plot state violations
    plt.subplot(122)
    plt.plot(violations[:, 1], label='State Violation')
    plt.xlabel('MPC Step')
    plt.ylabel('State Constraint Violation')
    plt.title('State Violations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_all_metrics(suffix, use_rho_adaptation=False, data_dir='../data', dt=0.02):
    """Plot all metrics from saved files"""
    data_dir = Path(data_dir)
    
    # Load all data from files
    iterations = np.loadtxt(data_dir / 'iterations' / f'traj{suffix}.txt')
    violations = np.loadtxt(data_dir / 'violations' / f'violations{suffix}.txt')
    trajectory_costs = np.loadtxt(data_dir / 'trajectory_costs' / f'traj{suffix}.txt')
    control_efforts = np.loadtxt(data_dir / 'control_efforts' / f'traj{suffix}.txt')
    
    if use_rho_adaptation:
        rho_history = np.loadtxt(data_dir / 'rho_history' / f'traj{suffix}.txt')
    else:
        rho_history = None
    
    plt.figure(figsize=(20, 10))
    
    # Plot tracking error/trajectory costs
    plt.subplot(231)
    plt.plot(np.arange(len(trajectory_costs))*dt, trajectory_costs)
    plt.xlabel('Time [s]')
    plt.ylabel('Trajectory Cost')
    plt.title('Tracking Cost over Time')
    plt.grid(True)
    
    # Plot violations
    plt.subplot(232)
    plt.plot(violations[:, 0], label='Input Violation')
    plt.plot(violations[:, 1], label='State Violation')
    plt.xlabel('MPC Step')
    plt.ylabel('Constraint Violation')
    plt.title('Constraint Violations')
    plt.legend()
    plt.grid(True)
    
    # Plot iterations
    plt.subplot(233)
    plt.plot(iterations, label='Iterations')
    plt.xlabel('MPC Step')
    plt.ylabel('Iterations')
    plt.title('ADMM Iterations')
    plt.grid(True)
    
    # Plot rho history if adaptive
    plt.subplot(236)
    if use_rho_adaptation and rho_history is not None:
        plt.plot(rho_history, label='Rho')
        plt.xlabel('Time Step')
        plt.ylabel('Rho Value')
        plt.title('Rho Adaptation History')
    else:
        plt.text(0.5, 0.5, 'Statistics:\n' + 
                f"Avg iterations: {np.mean(iterations):.2f}\n" +
                f"Total iterations: {sum(iterations)}\n" +
                f"Avg traj cost: {np.mean(trajectory_costs):.4f}\n" +
                f"Avg control effort: {np.mean(control_efforts):.4f}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.axis('off')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_comparisons(data_dir='../data', traj_type='full', compare_type='normal'):
    """Load saved data and create comparison plots"""
    try:
        data_dir = Path(data_dir)
        print("\nLoading metrics for comparison...")
        
        # Build suffixes exactly as used in hover.py/traj.py
        if compare_type == 'wind':
            adaptive_suffix = f'_adaptive_wind_{traj_type}'
            fixed_suffix = f'_normal_wind_{traj_type}'
            title_suffix = 'with Wind'
        else:
            adaptive_suffix = f'_adaptive_{traj_type}'
            fixed_suffix = f'_normal_{traj_type}'
            title_suffix = ''
            
        # Load data using exact same paths as hover.py/traj.py
        adaptive_costs = np.loadtxt(data_dir / 'costs' / f"costs{adaptive_suffix}.txt")
        adaptive_violations = np.loadtxt(data_dir / 'violations' / f"violations{adaptive_suffix}.txt")
        adaptive_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f"traj{adaptive_suffix}.txt")
        adaptive_control = np.loadtxt(data_dir / 'control_efforts' / f"traj{adaptive_suffix}.txt")
        adaptive_iterations = np.loadtxt(data_dir / 'iterations' / f"traj{adaptive_suffix}.txt")
        
        # Load fixed data
        fixed_costs = np.loadtxt(data_dir / 'costs' / f"costs{fixed_suffix}.txt")
        fixed_violations = np.loadtxt(data_dir / 'violations' / f"violations{fixed_suffix}.txt")
        fixed_traj_costs = np.loadtxt(data_dir / 'trajectory_costs' / f"traj{fixed_suffix}.txt")
        fixed_control = np.loadtxt(data_dir / 'control_efforts' / f"traj{fixed_suffix}.txt")
        fixed_iterations = np.loadtxt(data_dir / 'iterations' / f"traj{fixed_suffix}.txt")
        
        # Create figure with extra space on right for stats
        plt.figure(figsize=(22, 10))  # Made figure wider to accommodate stats
        
        # State Costs
        plt.subplot(231)
        plt.plot(fixed_costs[:, 0], 'b-o', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_costs[:, 0], 'r-^', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        #plt.title(f'State Costs Comparison ({title_suffix})')
        plt.title('State Costs Comparison')
        plt.legend()
        plt.grid(True)
        
        # Input Costs
        plt.subplot(232)
        plt.plot(fixed_costs[:, 1], 'b-s', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_costs[:, 1], 'r-d', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('Input Costs Comparison')
        plt.legend()
        plt.grid(True)
        
        # Input Violations
        plt.subplot(233)
        plt.plot(fixed_violations[:, 0], 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_violations[:, 0], 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('Input Violations')
        plt.legend()
        plt.grid(True)

        # State Violations (with adjusted scale)
        plt.subplot(234)
        plt.plot(fixed_violations[:, 1], 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_violations[:, 1], 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('State Violations (×10⁻³)')
        plt.legend()
        plt.grid(True)

        #Iterations
        plt.subplot(235)
        plt.plot(fixed_iterations, 'b-*', label='Fixed', alpha=0.5, linewidth=2, markersize=4, markevery=20)
        plt.plot(adaptive_iterations, 'r-p', label='Adaptive', alpha=0.8, linewidth=2.5, markersize=6, markevery=20)
        plt.xlabel('MPC Step')
        
        plt.title('Iterations')
        plt.legend()
        plt.grid(True)
        

        
        # Add statistics box to the right of plots
        stats_text = (
            f'Average Metrics:\n\n'
            f'State Cost:\n'
            f'  Fixed: {np.mean(fixed_costs[:, 0]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_costs[:, 0]):.3f}\n\n'
            f'Input Cost:\n'
            f'  Fixed: {np.mean(fixed_costs[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_costs[:, 1]):.3f}\n\n'
            f'Total Cost:\n'
            f'  Fixed: {np.mean(fixed_costs[:, 0] + fixed_costs[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_costs[:, 0] + adaptive_costs[:, 1]):.3f}\n\n'
            f'Input Violation:\n'
            f'  Fixed: {np.mean(fixed_violations[:, 0]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_violations[:, 0]):.3f}\n\n'
            f'State Violation:\n'
            f'  Fixed: {np.mean(fixed_violations[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_violations[:, 1]):.3f}\n\n'
            f'Total Violation:\n'
            f'  Fixed: {np.mean(fixed_violations[:, 0] + fixed_violations[:, 1]):.3f}\n'
            f'  Adaptive: {np.mean(adaptive_violations[:, 0] + adaptive_violations[:, 1]):.3f}\n\n'
            f'Total Iterations:\n'
            f'  Fixed: {np.sum(fixed_iterations)}\n'
            f'  Adaptive: {np.sum(adaptive_iterations)}\n\n'

        )
        # Place stats box to the right of plots
        plt.figtext(0.92, 0.5, stats_text, 
                   fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                   verticalalignment='center')
        
        plt.tight_layout()
        # Adjust layout to make room for stats box
        plt.subplots_adjust(right=0.88)
        plt.show()
        
    except FileNotFoundError as e:
        print("\nError: Make sure you've run both cases before comparison")
        print(f"Missing file: {e}")

def plot_hover_iterations_comparison():
    """Compare iterations between different hover control approaches"""
    data_dir = Path('../data/iterations')
    
    # Load the data
    normal = np.loadtxt(data_dir / 'traj_normal_hover.txt')
    adaptive = np.loadtxt(data_dir / 'traj_adaptive_hover.txt')
    recompute = np.loadtxt(data_dir / 'traj_adaptive_recache_hover.txt')
    
    # Create plot
    plt.figure(figsize=(10, 6), dpi=100)
    t = np.arange(len(normal)) * 0.02  # assuming dt = 0.02
    
    plt.plot(t, normal, 'b-', label='Fixed ρ', linewidth=2)
    plt.plot(t, adaptive, 'r-', label='Adaptive ρ', linewidth=2)
    plt.plot(t, recompute, 'g-', label='Adaptive ρ (recompute)', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Iterations', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.title('ADMM Iterations Comparison for Hover Control', fontsize=14)
    
    plt.tight_layout()
    plt.show()

def plot_paper_hover_iterations():
    """Generate hover iterations plot for paper with specific formatting"""
    data_dir = Path('../data/iterations')
    
    # Load the data
    fixed = np.loadtxt(data_dir / 'traj_normal_hover.txt')
    adaptive = np.loadtxt(data_dir / 'traj_adaptive_hover.txt')
    heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_hover.txt')
    recompute = np.loadtxt(data_dir / 'traj_adaptive_recache_hover.txt')

    
    # Calculate cumulative iterations
    fixed_cum = np.cumsum(fixed)
    adaptive_cum = np.cumsum(adaptive)
    heuristic_cum = np.cumsum(heuristic)
    recompute_cum = np.cumsum(recompute)

    
    # Create plot with paper-style formatting
    plt.figure(figsize=(8, 4), dpi=300)  # High DPI for paper quality
    t = np.arange(len(fixed)) * 0.02  # assuming dt = 0.02
    
    # Plot lines with paper-specified colors and styles
    plt.plot(t, fixed_cum, 'r-', label='Fixed ρ', linewidth=2)
    plt.plot(t, adaptive_cum, 'b-', label='Adaptive ρ', linewidth=2)
    plt.plot(t, heuristic_cum, 'g-', label='Heuristic ρ', linewidth=2)
    plt.plot(t, recompute_cum, color='gray', linestyle='--', label='Recomputation', linewidth=2)
    
    # Formatting
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Cumulative Iterations', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    
    # Calculate reductions from fixed (baseline)
    fixed_to_adaptive = ((fixed_cum[-1] - adaptive_cum[-1]) / fixed_cum[-1]) * 100
    fixed_to_recompute = ((fixed_cum[-1] - recompute_cum[-1]) / fixed_cum[-1]) * 100
    fixed_to_heuristic = ((fixed_cum[-1] - heuristic_cum[-1]) / fixed_cum[-1]) * 100

    print("\nIteration Reduction Analysis:")
    print(f"• Adaptive ρ reduces iterations by {fixed_to_adaptive:.1f}% from baseline")
    print(f"• Heuristic ρ reduces iterations by {fixed_to_heuristic:.1f}% from baseline")
    print(f"• Recomputation reduces iterations by {fixed_to_recompute:.1f}% from baseline")
    
    # Also print raw numbers for verification
    print(f"\nRaw iteration counts:")
    print(f"Fixed ρ: {fixed_cum[-1]:.0f}")
    print(f"Adaptive ρ: {adaptive_cum[-1]:.0f}")
    print(f"Recompute: {recompute_cum[-1]:.0f}")
    print(f"Heuristic ρ: {heuristic_cum[-1]:.0f}")

    plt.tight_layout()

    plt.show()


def plot_paper_rho_trends():
    """Generate rho adaptation trends plot for paper"""
    data_dir = Path('../data/rho_history')
    
    # Load the rho histories
    analytical = np.loadtxt(data_dir / 'traj_adaptive_hover.txt')  # OSQP/analytical method
    heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_hover.txt')  # Heuristic method


    # Prepend initial rho (85.0) to both arrays
    analytical = np.concatenate(([85.0], analytical))
    heuristic = np.concatenate(([85.0], heuristic))
    
    # Create plot with paper-style formatting
    plt.figure(figsize=(8, 4), dpi=300)
    t = np.arange(len(analytical)) * 0.02  # assuming dt = 0.02
    
    # Plot lines
    plt.plot(t, analytical, 'b-', label='OSQP', linewidth=2)
    plt.plot(t, heuristic, 'g-', label='Residual Heuristic', linewidth=2)
    
    # Formatting
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel(r'$\rho$', fontsize=12)  # Using LaTeX formatting
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    
    plt.tight_layout()

    plt.show()

def plot_paper_rho_comparison():
    """Generate comparison plot of different fixed rho values vs adaptive"""
    data_dir = Path('../data/iterations')
    
    fig = plt.figure(figsize=(10, 6), dpi=100)

    # Load data
    rho_50 = np.loadtxt(data_dir / 'traj_normal_0.1_hover.txt')
    rho_75 = np.loadtxt(data_dir / 'traj_normal_1_hover.txt')
    rho_100 = np.loadtxt(data_dir / 'traj_normal_100_hover.txt')
    
    rho_50_adapt = np.loadtxt(data_dir / 'traj_adaptive_50_hover.txt')
    rho_75_adapt = np.loadtxt(data_dir / 'traj_adaptive_75_hover.txt')
    rho_100_adapt = np.loadtxt(data_dir / 'traj_adaptive_100_hover.txt')

    rho_50_heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_50_hover.txt')
    rho_75_heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_75_hover.txt')
    rho_100_heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_100_hover.txt')

    t = np.arange(len(rho_50)) * 0.02  # assuming dt = 0.02

    # Calculate and print iteration statistics
    def calculate_reduction(fixed, adaptive):
        total_fixed = np.sum(fixed)
        total_adaptive = np.sum(adaptive)
        reduction = (total_fixed - total_adaptive) / total_fixed * 100
        return total_fixed, total_adaptive, reduction

    print("\nIteration Reduction Statistics:")
    
    print("\nρ = 50:")
    fixed_50, adapt_50, red_50 = calculate_reduction(rho_50, rho_50_adapt)
    fixed_50, heur_50, red_50_h = calculate_reduction(rho_50, rho_50_heuristic)
    print(f"Fixed: {fixed_50} iterations")
    print(f"Adaptive: {adapt_50} iterations ({red_50:.1f}% reduction)")
    print(f"Heuristic: {heur_50} iterations ({red_50_h:.1f}% reduction)")

    print("\nρ = 75:")
    fixed_75, adapt_75, red_75 = calculate_reduction(rho_75, rho_75_adapt)
    fixed_75, heur_75, red_75_h = calculate_reduction(rho_75, rho_75_heuristic)
    print(f"Fixed: {fixed_75} iterations")
    print(f"Adaptive: {adapt_75} iterations ({red_75:.1f}% reduction)")
    print(f"Heuristic: {heur_75} iterations ({red_75_h:.1f}% reduction)")

    print("\nρ = 100:")
    fixed_100, adapt_100, red_100 = calculate_reduction(rho_100, rho_100_adapt)
    fixed_100, heur_100, red_100_h = calculate_reduction(rho_100, rho_100_heuristic)
    print(f"Fixed: {fixed_100} iterations")
    print(f"Adaptive: {adapt_100} iterations ({red_100:.1f}% reduction)")
    print(f"Heuristic: {heur_100} iterations ({red_100_h:.1f}% reduction)")

    # Plot the data with explicit axis control for all subplots
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t, rho_50, 'r-', label=r'$\rho = 50$ fixed', linewidth=2)
    ax1.plot(t, rho_50_adapt, 'b-', label=r'$\rho = 50$ adaptive', linewidth=2)
    ax1.plot(t, rho_50_heuristic, 'g-', label=r'$\rho = 50$ heuristic', linewidth=2)
    ax1.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    ax1.grid(False)

    ax2 = plt.subplot(3,1,2)
    ax2.plot(t, rho_75, 'r-', label=r'$\rho = 75$ fixed', linewidth=2)
    ax2.plot(t, rho_75_adapt, 'b-', label=r'$\rho = 75$ adaptive', linewidth=2)
    ax2.plot(t, rho_75_heuristic, 'g-', label=r'$\rho = 75$ heuristic', linewidth=2)
    ax2.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    ax2.grid(False)

    ax3 = plt.subplot(3,1,3)
    ax3.plot(t, rho_100, 'r-', label=r'$\rho = 100$ fixed', linewidth=2)
    ax3.plot(t, rho_100_adapt, 'b-', label=r'$\rho = 100$ adaptive', linewidth=2)
    ax3.plot(t, rho_100_heuristic, 'g-', label=r'$\rho = 100$ heuristic', linewidth=2)
    ax3.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    ax3.grid(False)
    
    plt.tight_layout()

    plt.show()


def plot_paper_trajectory_comparison():
    """Three clean figure-8 plots, flipped vertically for better space usage."""
    data_dir = Path('../data')
    
    try:
        # Load data
        x_normal = np.loadtxt(data_dir / 'trajectories' / 'traj_normal_wind_full.txt')
        x_adaptive = np.loadtxt(data_dir / 'trajectories' / 'traj_adaptive_wind_full.txt')
        x_heuristic = np.loadtxt(data_dir / 'trajectories' / 'traj_adaptive_heuristic_wind_full.txt')
        reference = np.loadtxt(data_dir / 'trajectories' / 'reference_trajectory.txt')
        
        # Find divergence points
        def find_divergence(traj_data, ref_data):
            for i, x in enumerate(traj_data):
                if np.linalg.norm(x - ref_data[i]) > 5.0 or np.any(np.isnan(x)):
                    return i
            return len(traj_data)
        
        div_idx_normal = find_divergence(x_normal, reference)
        div_idx_adaptive = find_divergence(x_adaptive, reference)
        div_idx_heuristic = find_divergence(x_heuristic, reference)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))
        plt.subplots_adjust(wspace=0.1)  # Reduce spacing between plots
        
        # Common plotting parameters
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.2)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        # Plot trajectories (swapping x and z for vertical orientation)
        # Fixed ρ
        ax1.plot(reference[:, 2], reference[:, 0], 'k--', alpha=0.7, linewidth=1.5)
        ax1.plot(x_normal[:div_idx_normal, 2], x_normal[:div_idx_normal, 0], 'r-', linewidth=2)
        ax1.plot(x_normal[div_idx_normal-1, 2], x_normal[div_idx_normal-1, 0], 'r*', markersize=10)
        
        # OSQP-Adaptive
        ax2.plot(reference[:, 2], reference[:, 0], 'k--', alpha=0.7, linewidth=1.5)
        ax2.plot(x_adaptive[:, 2], x_adaptive[:, 0], 'b-', linewidth=2)
        
        # Heuristic
        ax3.plot(reference[:, 2], reference[:, 0], 'k--', alpha=0.7, linewidth=1.5)
        ax3.plot(x_heuristic[:div_idx_heuristic, 2], x_heuristic[:div_idx_heuristic, 0], 'g-', linewidth=2)
        ax3.plot(x_heuristic[div_idx_heuristic-1, 2], x_heuristic[div_idx_heuristic-1, 0], 'g*', markersize=10)
        
        # Ensure same limits
        xlims = [ax.get_xlim() for ax in [ax1, ax2, ax3]]
        ylims = [ax.get_ylim() for ax in [ax1, ax2, ax3]]
        x_min, x_max = min(l[0] for l in xlims), max(l[1] for l in xlims)
        y_min, y_max = min(l[0] for l in ylims), max(l[1] for l in ylims)
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")


def plot_paper_rho_wind_comparison():
    """Generate plot showing rho adaptation during wind disturbance"""
    data_dir = Path('../data')
    
    try:
        # Load rho history data
        rho_fixed = np.ones(200) * 1.0  # Full trajectory length
        rho_adaptive = np.loadtxt(data_dir / 'rho_history' / 'traj_adaptive_wind_full.txt')
        rho_heuristic = np.loadtxt(data_dir / 'rho_history' / 'traj_adaptive_heuristic_wind_full.txt')
        
        # Create time array for full trajectory
        dt = 0.02
        t = np.arange(len(rho_adaptive)) * dt
        
        # Create the plot
        fig = plt.figure(figsize=(12, 5))
        
        # Plot rho values
        plt.plot(t, rho_fixed, 'r:', label='Fixed ρ', linewidth=2, alpha=0.8)
        plt.plot(t, rho_adaptive, 'b-', label='Adaptive ρ', linewidth=1.5)
        plt.plot(t, rho_heuristic, 'g-.', label='Heuristic ρ', linewidth=1.5)
        
        # Add divergence line
        plt.axvline(x=2.60, color='gray', linestyle='--', alpha=0.5, label='Divergence')
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('ρ Value', fontsize=12)
        plt.legend(fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=10)
        
        # Set y-axis limits with some padding
        max_rho = max(np.max(rho_adaptive), np.max(rho_heuristic))
        plt.ylim([0, max_rho * 1.1])
        
        plt.tight_layout()
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run all wind simulations first:")
        print("1. python traj.py --wind")
        print("2. python traj.py --wind --adapt")
        print("3. python traj.py --wind --adapt --heuristic")

def plot_combined_paper_figures():
    """Combine rho_trend, hover_iterations, and rho_comparison plots horizontally"""
    
    # Create one wide figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # First subplot - Rho Trends
    data_dir = Path('../data/rho_history')
    analytical = np.loadtxt(data_dir / 'traj_adaptive_hover.txt')
    heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_hover.txt')
    analytical = np.concatenate(([85.0], analytical))
    heuristic = np.concatenate(([85.0], heuristic))
    
    t = np.arange(len(analytical)) * 0.02
    ax1.plot(t, analytical, 'b-', label='OSQP', linewidth=2)
    ax1.plot(t, heuristic, 'g-', label='Residual Heuristic', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel(r'$\rho$', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    ax1.set_title('Penalty Parameter Adaptation', fontsize=14)
    
    # Second subplot - Hover Iterations
    data_dir = Path('../data/iterations')
    rho_50 = np.loadtxt(data_dir / 'traj_normal_0.1_hover.txt')
    rho_75 = np.loadtxt(data_dir / 'traj_normal_1_hover.txt')
    rho_100 = np.loadtxt(data_dir / 'traj_normal_100_hover.txt')
    t = np.arange(len(rho_50)) * 0.02
    
    ax2.plot(t, rho_50, 'r-', label=r'$\rho = 50$', linewidth=2)
    ax2.plot(t, rho_75, 'b-', label=r'$\rho = 75$', linewidth=2)
    ax2.plot(t, rho_100, 'g-', label=r'$\rho = 100$', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Iterations', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    ax2.set_title('Fixed Parameter Performance', fontsize=14)
    
    # Third subplot - Rho Comparison
    rho_50_adapt = np.loadtxt(data_dir / 'traj_adaptive_50_hover.txt')
    rho_75_adapt = np.loadtxt(data_dir / 'traj_adaptive_75_hover.txt')
    rho_100_adapt = np.loadtxt(data_dir / 'traj_adaptive_100_hover.txt')

    # Load data
    rho_50 = np.loadtxt(data_dir / 'traj_normal_0.1_hover.txt')
    rho_75 = np.loadtxt(data_dir / 'traj_normal_1_hover.txt')
    rho_100 = np.loadtxt(data_dir / 'traj_normal_100_hover.txt')
    
    rho_50_adapt = np.loadtxt(data_dir / 'traj_adaptive_50_hover.txt')
    rho_75_adapt = np.loadtxt(data_dir / 'traj_adaptive_75_hover.txt')
    rho_100_adapt = np.loadtxt(data_dir / 'traj_adaptive_100_hover.txt')

    rho_50_heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_50_hover.txt')
    rho_75_heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_75_hover.txt')
    rho_100_heuristic = np.loadtxt(data_dir / 'traj_adaptive_heuristic_100_hover.txt')
    
    ax3.plot(t, rho_50, 'r:', label='Fixed', linewidth=2)
    ax3.plot(t, rho_50_adapt, 'b-', label='Adaptive', linewidth=2)
    ax3.plot(t, rho_50_heuristic, 'g-', label='Heuristic', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Iterations', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12, framealpha=1.0, edgecolor='black')
    ax3.set_title('Parameter Adaptation Comparison', fontsize=14)
    
    plt.tight_layout()
    
    # # Save the combined figure
    # plt.savefig('../images/combined_analysis.png', 
    #             dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()




def plot_paper_figures():
    """Generate all figures for the paper"""
    print("\nGenerating paper figures...")
    
    # # 1. Hover Iterations Plot
    # print("\n1. Generating hover iterations plot...")
    # try:
    #     plot_paper_hover_iterations()
    #     print("✓ Hover iterations plot saved")
    # except FileNotFoundError as e:
    #     print(f"✗ Error: {e}")
    #     print("  Make sure you've run: python hover.py --adapt and python hover.py")
    
    # # 2. Rho Trends Plot
    # print("\n2. Generating rho trends plot...")
    # try:
    #     plot_paper_rho_trends()
    #     print("✓ Rho trends plot saved")
    # except FileNotFoundError as e:
    #     print(f"✗ Error: {e}")
    #     print("  Make sure you've run: python hover.py --adapt and python hover.py --adapt --heuristic")


    # # 3. Rho Comparison Plot
    # print("\n3. Generating rho comparison plot...")
    # try:
    #     plot_paper_rho_comparison()
    #     print("✓ Rho comparison plot saved")
    # except FileNotFoundError as e:
    #     print(f"✗ Error: {e}")
    #     print("  Make sure you've run: python hover.py --adapt and python hover.py --adapt --heuristic")


    try:
        plot_combined_paper_figures()
        print("✓ Combined paper figures saved")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Make sure you've run: python hover.py --adapt and python hover.py --adapt --heuristic")
    
    # 4. Trajectory Comparison Plot
    print("\n4. Generating trajectory comparison plot...")
    try:
        plot_paper_trajectory_comparison()
        print("✓ Trajectory comparison plot saved")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Make sure you've run: python traj.py with all necessary configurations")

    # 5. Rho Wind Comparison Plot
    print("\n5. Generating rho wind comparison plot...")
    try:
        plot_paper_rho_wind_comparison()
        print("✓ Rho wind comparison plot saved")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Make sure you've run: python traj.py --wind --adapt --heuristic")

    print("\nPaper figures generation complete!")



