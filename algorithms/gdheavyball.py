from bench_algo import *
from algo_config import *
import numpy as np

class GradientDescentHeavyBall(Optimizer):
    """
    Implements the Heavy-Ball method (Gradient Descent with Momentum).
    The update rule is:
        v_{k+1} = beta * v_k + alpha * grad_f(x_k)
        x_{k+1} = x_k - v_{k+1}
    where
    - alpha is typically 1 / L (step size)
    - beta is the momentum parameter in (0,1)
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Constructor for the Heavy-Ball (momentum) optimizer.
        """
        if params is None:
            params = {}
        
        self.stats = {}

        super().__init__(params[ALG_UNIVERSAL_PARAM_NAME], params)

    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize the function f using Heavy-Ball momentum.

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
        tol = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-06)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        L_est = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, inf)
        beta = self.params.get(ALG_HEAVY_BALL_MOMENTUM, 0.9)

        # Step size alpha = 1 / L, but if L = inf, pick something small by default
        if L_est == inf:
            alpha = 1e-3  
        else:
            alpha = 1.0 / L_est

        # Counters
        n_func_evals = 0
        n_grad_evals = 0
        n_iter = 0

        # Statistics
        fvals = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)

        # Initialize momentum "velocity"
        v = np.zeros_like(x)

        for i in range(max_iter):
            
            gx = grad_f(x)
            n_grad_evals += 1

            grad_norm = np.linalg.norm(gx, ord=np.inf)
            n_iter += 1

            # Save info for stats
            fvals[i] = f(x)  # function value
            gnorms[i] = grad_norm

            # Check stopping condition
            if grad_norm < tol:
                break

            # Heavy-Ball momentum update
            v = beta * v + alpha * gx
            x = x - v

        # If we ended early, fill in trailing stats
        if n_iter < max_iter:
            fvals[n_iter:] = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]

        # Collect stats
        stats = {
            ALG_STATS_ITERATIONS: n_iter,
            ALG_STATS_OPTIMAL_VALUE: f(x),
            ALG_STATS_OPTIMAL_SOL: x,
            ALG_STATS_RUNNING_TIME: 0,
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
    
    params = gradient_descent_params
    params[ALG_UNIVERSAL_PARAM_L_EST] = 4.0
    params[ALG_HEAVY_BALL_MOMENTUM] = 0.15
    
    # Initialize the optimizer
    hb = GradientDescentHeavyBall(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = hb.optimize(x_init, f, grad_f)
    
    # Print results
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