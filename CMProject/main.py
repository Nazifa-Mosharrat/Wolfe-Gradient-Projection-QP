import os
import numpy as np
from utils import Initialization
from analysis import compare_with_solver
from analysis import test_sensitivity_to_partition,test_sensitivity_to_tolerance
from visualization import plot_partition_sensitivity, plot_tolerance_sensitivity,plot_convergence_compariosn


# entry point to process all matrices in the folder
def main(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            path = os.path.join(folder_path, file_name)
            Q, q, _ = Initialization(path)
            results = compare_with_solver(Q, q, max_iters=500, tol=1e-9,file_name=file_name)
            
            cvx_solver_time=results["cvx_solver_time"]
            cvx_f_opt=results["cvx_f_opt"]
            wolfe_time=results["wolfe_time"]
            wolfe_opt=results["wolfe_f_opt"]
            wolfe_gap=results["wolfe_gap"]
            cvx_gap=results["cvx_gap"]
            print(f"File Name: {file_name} size : {Q.shape}")
            print(f"CVXOPT Solver Time: {cvx_solver_time:.4f} s, f* = {cvx_f_opt:.4e}")
            print(f"wolfe Time: {wolfe_time} s, f* = {wolfe_opt}")   
            print(f"relative gap wolfe without format: {float(wolfe_gap[-1])}")
            print(f"relative gap cvxopt without format: {float(cvx_gap[-1])}")
            #visualize convergence comparison
            plot_convergence_compariosn(wolfe_gap,cvx_gap,"Convergence comparison for file "+file_name);

            # test sensitivity to partition
            partition_values = [1, 2, 4, 8, 16, 32, max(1, int(np.sqrt(Q.shape[0])))]  # partition values to test
            results = test_sensitivity_to_partition(Q, q, max_iters=500, partition_values=partition_values, tol=1e-9)
            plot_partition_sensitivity(results, "Partition sensitivity comparison for file "+file_name)

            # test sensitivity to tolerance
            tolerance_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-13]
            tolerance_results = test_sensitivity_to_tolerance(Q, q, max_iters=500, tolerance_values=tolerance_values)
            execution_times = [res['execution_time'] for res in tolerance_results]
            gaps = [res['opt_value'] for res in tolerance_results]
            plot_tolerance_sensitivity(tolerance_values, execution_times, gaps,"Tolerance sensitivity comparison for file "+file_name)


