import matplotlib.pyplot as plt
import numpy as np
# plot convergence of the optimization methods
def plot_convergence(gaps,iteration, title):
    plt.plot(range(iteration), np.full(iteration, gaps), label="Gaps")
    plt.xlabel('Iterations')
    plt.ylabel('Relative Gap')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_convergence_compariosn(wolfe_gaps,cvxopt_gaps, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(wolfe_gaps)), wolfe_gaps, label="Wolfe")
    cvx_iteration = len(wolfe_gaps)   
    # plot cvxopt gaps
    plt.plot(range(len(cvxopt_gaps)), cvxopt_gaps, label="CVXOPT Gaps", marker='x')
    
    if cvx_iteration is not None:
        plt.axvline(x=cvx_iteration, color='r', linestyle='--', label=f"CVXOPT Convergence at Iteration {cvx_iteration}")

    plt.xlabel('Iterations')
    plt.ylabel('Gap (log scale)')
    plt.yscale('log')   
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_execution_time_comparison(wolfe_time,cvxopt_time,matrix_sizes,title):
    plt.figure()
    plt.plot(matrix_sizes, cvxopt_time, label='CVXOPT Solver Time', marker='o')
    plt.plot(matrix_sizes, wolfe_time, label='Adaptive Wolfe Time', marker='x')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# visualize the sensitivity analysis results
def plot_tolerance_sensitivity(tolerances, execution_times, gaps,title):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(tolerances, execution_times, label="Execution Time", color='orange', marker='s')
    ax1.set_xlabel("Tolerance (log scale)")
    ax1.set_ylabel("Execution Time (s)", color='orange')
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

     #visualize final gap 
    ax2 = ax1.twinx()
    ax2.plot(tolerances, gaps, label="Final Relative Gap", color='blue', marker='o')
    ax2.set_ylabel("Final Relative Gap (log scale)", color='blue')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='blue')    
    plt.title(title)
    fig.tight_layout()
    plt.show()
    
    

def plot_partition_sensitivity(results, title):
    partition_nums = [res['num_partitions'] for res in results]
    execution_times = [res['execution_time'] for res in results]
    plt.figure(figsize=(10, 6))
    plt.plot(partition_nums, execution_times, label='adaptive wolfe time', marker='o')
    plt.xlabel('number of partitions')
    plt.ylabel('execution time (s)')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()