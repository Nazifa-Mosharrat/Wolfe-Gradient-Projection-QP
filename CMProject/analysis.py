import numpy as np
from solver import WolfeProjectionMethod, CVXOPT_Solver
from utils import partition_indices

def compare_with_solver(Q, q, max_iters, tol, file_name):
     
    x_ref, f_opt, cvx_solver_time,cvx_gap = CVXOPT_Solver(Q, q, max_iters, tol)
    print(f"cvx_gap={cvx_gap}")
    
    Isets = partition_indices(Q.shape[0],None)
    exact_solution, exact_gap, exact_time, exact_iters, exact_opt = WolfeProjectionMethod(Q, q,Isets, max_iters, tol)
     
    print(f"CVXOPT Solver Time: {cvx_solver_time:.4f} s, f* = {f_opt:.4e}")
    print(f"wolfe Time: {exact_time:.4f} s, f* = {exact_opt:.4e}")   
    print(f"relative gap wolfe without format: {float(exact_gap[-1])}")
    print(f"relative gap cvxopt without format: {float(cvx_gap[-1])}")
     
   

    return {
        'file': file_name,
        'matrix_size': Q.shape[0],
        'cvx_solver': 'CVXOPT',
        'cvx_solver_time': cvx_solver_time,
        'cvx_f_opt': f_opt,
        'wolfe_time': exact_time,
        'wolfe_f_opt': exact_opt,         
        'wolfe_iterations': exact_iters,
        'wolfe_gap': exact_gap,
        'cvx_gap': cvx_gap,
    }

# test the sensitivity of the wolfe projection method to tolerance values
def test_sensitivity_to_tolerance(Q, q, max_iters, tolerance_values):
    results = []
    for tol in tolerance_values:
        Isets = partition_indices(Q.shape[0])
        exact_solution, exact_gap, exact_time, exact_iters, exact_opt = WolfeProjectionMethod(Q, q, Isets, max_iters, tol)
        results.append({
            "tolerance": tol,
            "solution": exact_solution,
            "execution_time":exact_time,
            "opt_value": exact_opt,
            "exact_gap": exact_gap
        })
    return results

# analyze sensitivity of wolfe projection method to the number of partitions
def test_sensitivity_to_partition(Q, q, max_iters, partition_values, tol):
    results = []
    for num_partitions in partition_values:
        Isets = partition_indices(Q.shape[0], num_partitions)  # create partitions
        exact_solution, exact_gap, exact_time, exact_iters, exact_opt = WolfeProjectionMethod(
            Q, q, Isets, max_iter=max_iters, tol=tol
        )
        results.append({
            "num_partitions": num_partitions,
            "solution": exact_solution,
            "opt_value": exact_opt,
            "execution_time": exact_time,
            "num_iterations": exact_iters,
            "gap_history": exact_gap
        })
    return results

  

