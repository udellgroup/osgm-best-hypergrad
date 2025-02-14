from bench_algo import *
from algo_config import *
import numpy as np

class GradientDescent(Optimizer):
    
    def __init__(self, params: dict = None):
        """
        Constructor for the gradient descent optimizer.
        """
        if params is None:
            params = {}
            
        self.stats = {}
        
        super().__init__(params[ALG_UNIVERSAL_PARAM_NAME], params)
        
    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize the function f using the gradient grad_f
        
        Parameters
        ----------
        x : np.ndarray
            Initial point
        f : callable
            Function to optimize
        grad_f : callable
            Gradient of the function
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer
        """
        
        tol = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-06)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        L_est = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, inf)
        
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
        
        for i in range(max_iter):
            
            gx = grad_f(x)
            n_grad_evals += 1
            fvals[i] = f(x)
            
            grad_norm = np.linalg.norm(gx, ord=np.inf)
            n_iter += 1
            
            if grad_norm < tol:
                break
            
            x = x - alpha * gx
            gnorms[i] = grad_norm
        
        if n_iter < max_iter:
            fvals[n_iter:] = fvals[n_iter - 1]
            gnorms[n_iter:] = gnorms[n_iter - 1]
            
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
    
    # Initialize the optimizer
    gd = GradientDescent(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = gd.optimize(x_init, f, grad_f)
    
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