from bench_algo import *
from algo_config import *
import numpy as np

class AcceleratedGradientCvx(Optimizer):
    """
    Implements Nesterov's Accelerated Gradient (NAG) for convex functions with variable momentum.

    One common update rule is:
        y_k = x_k + beta_k * (x_k - x_{k-1})
        x_{k+1} = y_k - alpha * grad_f(y_k)

    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Constructor for the NesterovAcceleratedGradientFixedMomentum optimizer.
        """
        if params is None:
            params = {}

        self.stats = {}

        super().__init__(params.get(ALG_UNIVERSAL_PARAM_NAME, "Accelerated Gradient Descent with Strong Convexity"), params)

    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize function f using Nesterov's Accelerated Gradient (fixed momentum).

        Parameters
        ----------
        x : np.ndarray
            Initial point
        f : callable
            Objective function
        grad_f : callable
            Gradient of the objective function

        Returns
        -------
        Dict[str, Any]
            A dictionary containing optimization statistics
        """

        # Extract parameters
        tol = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-6)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        L_est = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, np.inf)

        # Step size alpha = 1 / L, but if L = inf, pick something small by default
        if L_est == np.inf:
            raise ValueError("Lipschitz constant L must be provided for Accelerated Gradient with Strong Convexity")
        
        alpha = 1 / L_est
        lbd = 0.0
            
        # Counters
        n_func_evals = 0
        n_grad_evals = 0
        n_iter = 0

        # Statistics
        fvals = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)

        # Keep track of the previous x for momentum
        x_prev = np.copy(x)

        for i in range(max_iter):
            # Compute the "lookahead" point
            if i == 0:
                # At the first iteration, we have no previous step
                y = x
                lbd = 1.0
            else:
                lbdtmp = (1 + np.sqrt(1 + 4 * lbd**2)) / 2.0
                beta = (lbd - 1) / lbdtmp
                lbd = lbdtmp
                y = x + beta * (x - x_prev)

            gx = grad_f(x)
            gy = grad_f(y)
            n_grad_evals += 1

            grad_norm = np.linalg.norm(gx, ord=np.inf)
            n_iter += 1

            # Save info for stats
            fvals[i] = f(x)  
            gnorms[i] = grad_norm

            # Check stopping condition
            if grad_norm < tol:
                break

            # NAG update
            x_next = y - alpha * gy

            # Prepare for next iteration
            x_prev = np.copy(x)
            x = x_next

        # If we ended early, fill trailing stats
        if n_iter < max_iter:
            fvals[n_iter:] = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]

        # Collect stats
        stats = {
            ALG_STATS_ITERATIONS: n_iter,
            ALG_STATS_OPTIMAL_VALUE: f(x),
            ALG_STATS_OPTIMAL_SOL: x,
            ALG_STATS_RUNNING_TIME: 0,  # placeholder if timing is tracked
            ALG_STATS_FUNCVALS: fvals,
            ALG_STATS_GNORMS: gnorms,
            ALG_STATS_FEVALS: n_func_evals,
            ALG_STATS_GEVALS: n_grad_evals
        }

        self.stats = stats
        return stats

    def get_optimizer_stats(self) -> Dict[str, Any]:
        """
        Get the optimizer statistics
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer
        """
        return self.stats
    
    
if __name__ == "__main__":
    
    params = accelerated_gradient_descent_cvx_params
    params[ALG_UNIVERSAL_PARAM_L_EST] = 4.0
    
    # Initialize the optimizer
    agdcvx = AcceleratedGradientCvx(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = agdcvx.optimize(x_init, f, grad_f)
    
    # Print results
    print("Optimizer Name:", params[ALG_UNIVERSAL_PARAM_NAME])
    print("Optimizer Stats:")
    print(f"Iterations:         {stats[ALG_STATS_ITERATIONS]}")
    print(f"Optimal Value:      {stats[ALG_STATS_OPTIMAL_VALUE]:.6f}")
    print(f"Optimal Solution:   {stats[ALG_STATS_OPTIMAL_SOL]}")
    print(f"Function Evaluations: {stats[ALG_STATS_FEVALS]}")
    print(f"Gradient Evaluations: {stats[ALG_STATS_GEVALS]}")
    print(f"Final Gradient Norm: {np.linalg.norm(grad_f(stats[ALG_STATS_OPTIMAL_SOL])):.6f}")
    print("\nFunction Values (first 5):", stats[ALG_STATS_FUNCVALS][0:5])
    print("Gradient Norms (first 5):", stats[ALG_STATS_GNORMS][0:5])
    print("\nTest completed!")
    
    