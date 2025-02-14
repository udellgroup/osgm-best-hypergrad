from bench_algo import *
from algo_config import *
import numpy as np
from scipy.optimize import minimize


class SciPyLBFGS(Optimizer):
    """
    A wrapper class that uses SciPy's 'BFGS' method for optimization.
    This class uses a callback function to extract the function value
    (and optionally gradient norm) at each iteration.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Constructor for the SciPyBFGS optimizer.
        """
        if params is None:
            params = {}
        self.stats = {}
        super().__init__(params.get(ALG_UNIVERSAL_PARAM_NAME, "SciPy-LBFGS"), params)

    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize the function f using SciPy's 'LBFGS' method.

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
        mem_size = self.params.get(ALG_LBFGS_MEMORY_SIZE, 1)

        # Arrays (lists) to store function values and gradient norms
        fvals = [f(x)]
        gnorms = [np.linalg.norm(grad_f(x), ord=np.inf)]

        # Callback to collect stats at each iteration
        def callback(xk):
            # Record function value
            fxk = f(xk)
            fvals.append(fxk)
            # (Optionally) record gradient norm
            gxk = grad_f(xk)
            gnorms.append(np.linalg.norm(gxk, ord=np.inf))

        # Run SciPy's minimize with BFGS
        res = minimize(
            fun=f,
            x0=x,
            method="L-BFGS-B",
            jac=grad_f,
            callback=callback,
            options={
                "maxcor": mem_size,  # memory size
                "ftol": 1e-25,  
                "gtol": tol,       # gradient tolerance
                "maxiter": max_iter - 1,
                "disp": False
            }
        )

        # Final solution
        x_opt = res.x
        f_opt = res.fun
        n_iter = res.nit  # number of iterations according to SciPy
        # If the user wants the actual # of function or gradient calls from SciPy,
        # those are recorded in self._n_func_evals and self._n_grad_evals.

        # Convert lists to numpy arrays
        # If no iterations occurred, fvals might be empty, so handle carefully:
        if len(fvals) == 0:
            # This means BFGS might have converged immediately
            fvals = np.array([f_opt])
            gnorms = np.array([np.linalg.norm(grad_f(x_opt), ord=np.inf)])
        else:
            fvals = np.array(fvals)
            gnorms = np.array(gnorms)
            
        # Fill in fvals to an array of length maxiter
        fvals = np.pad(fvals, (0, max_iter - len(fvals)), mode='edge')
        gnorms = np.pad(gnorms, (0, max_iter - len(gnorms)), mode='edge')
        
        n_fev = res.nfev
        n_jev = res.njev
        
        if not res.success or np.min(gnorms) > 1.1 * tol:
            n_iter = max_iter
            n_fev = max_iter
            n_jev = max_iter
            
        # Construct stats dictionary
        stats = {
            ALG_STATS_ITERATIONS: n_iter,
            ALG_STATS_OPTIMAL_VALUE: f_opt,
            ALG_STATS_OPTIMAL_SOL: x_opt,
            ALG_STATS_RUNNING_TIME: 0,  # placeholder if timing is desired
            ALG_STATS_FUNCVALS: fvals,
            ALG_STATS_GNORMS: gnorms,
            ALG_STATS_FEVALS: n_fev,  # function evaluations from SciPy
            ALG_STATS_GEVALS: n_jev,  # gradient evaluations from SciPy
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
    
    params = lbfgs_params
    
    lbfgs_params[ALG_LBFGS_MEMORY_SIZE] = 1
    
    # Initialize the optimizer
    lbfgs = SciPyLBFGS(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = lbfgs.optimize(x_init, f, grad_f)
    
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