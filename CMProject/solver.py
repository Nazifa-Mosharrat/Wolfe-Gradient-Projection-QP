import numpy as np
import cvxpy as cp
from utils import projection, objective
import time
# implement the wolfe projection method for quadratic optimization
def WolfeProjectionMethod(Q, q, Isets, max_iter=500, tol=1e-6):
    start_time = time.time()
    Lf = np.linalg.eigvalsh(Q).max()
    n = Q.shape[0]
    x = np.zeros(n)
    gap_history = []      
    for I in Isets:
        x[I] = 1 / len(I)  # initialize x in a distributed  over subsets
    for k in range(max_iter):
        grad_f = 2 * Q @ x + q  # compute gradient
        x_old = x.copy()
        d = -grad_f  # descent direction
        alpha = -d.T @ Q @ x + q.T @ d  # compute step size
        alpha /= d.T @ Q @ d
        x_new = x + alpha * d  # take the step
        for I in Isets:
            x_new[I] = projection(x_new[I])  # project onto simplex
        
        x = x_new
        gap = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-6)
        gap_history.append(gap)
        #convergence check
        if np.linalg.norm(grad_f) < (alpha*tol)/2 or np.linalg.norm(x - x_old) < tol / Lf:
            print(f"converged at iteration {k+1} with relative gap: {gap:.6e}")             
            break
    f_opt = objective(Q, q, x)
    end_time = time.time()
    execution_time = end_time - start_time 
     
    return x, gap_history,execution_time, k+1,f_opt,

# solve the quadratic optimization problem using cvxpy's cvxopt solver
def CVXOPT_Solver(Q, q, max_iters, tol):
    start_time = time.time()
    n = Q.shape[0]
    x = cp.Variable(n)
    objective = cp.quad_form(x, cp.psd_wrap(Q)) + q @ x
    constraints = [x >= 0, cp.sum(x) == 1]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CVXOPT)
    relative_gaps = []
    prev_value = None

    start_time = time.time()
    for iteration in range(1, max_iters + 1):
        try:
            problem.solve(solver=cp.CVXOPT, abstol=1e-4, reltol=1e-4)
        except Exception as e:
            print("solver encountered an error:", e)
            break

        current_value = problem.value
        if prev_value is not None:
            relative_gap = abs(current_value - prev_value) / (1 + abs(prev_value) + abs(current_value))
            relative_gaps.append(relative_gap)
            if relative_gap < tol:
                print(f"converged at iteration {iteration} with relative gap: {relative_gap:.6e}")
                break
        prev_value = current_value
    end_time = time.time()
    execution_time =end_time - start_time
    return x.value, problem.value,execution_time, relative_gaps
