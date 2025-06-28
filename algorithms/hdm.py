from bench_algo import *
from algo_config import *
import numpy as np

"""
Notes on algorithm tuning

There are several important factors that affect algorithm performance.

1. Scalar or diagonal candidate set for stepsize and momentum
2. AdaGrad learning rate for {P_k} and {\beta_k}
3. Dynamic curvature estimation
    Number of curvature estimates to collect
    How to estimate the curvature
    How to use the curvature estimation: mean or geometric mean
4. Momentum initialization strategy: 0 or 0.95 for different candidate sets
    Range of momentum clipping for different candidate sets
5. Whether to reset momentum to 0 when a null step is taken

"""

class HyperGradientDescent(Optimizer):
    """
    Implement hypergradient descent with momentum
    Optimized version with reduced function and gradient calls
    
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Constructor for the hypergradient descent optimizer.
        """
        
        if params is None:
            params = {}
        
        self.stats = {}
        super().__init__(params.get(ALG_UNIVERSAL_PARAM_NAME, "AdaGrad"), params)
        
    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> Dict[str, Any]:
        """
        Optimize the function f using hypergradient descent with momentum
        
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
        is_log = self.params.get(ALG_HDM_LOGGING, False)
        is_convex = not self.params.get(ALG_HDM_NONCONVEX, True)
        
        # Note 1. Scalar or diagonal candidate set for stepsize and momentum
        version = self.params.get(ALG_HDM_LR_VERSION, ALG_HDM_VERSION_DIAG)
        beta_version = self.params.get(ALG_HDM_BETA_VERSION, ALG_HDM_VERSION_SCALAR)
        
        # Note 2. AdaGrad learning rate for {P_k} (Dynamic) and {\beta_k} (Prespecified)
        lr = self.params.get(ALG_HDM_LEARNING_RATE, -1)
        beta_lr = self.params.get(ALG_HDM_BETA_LEARNING_RATE, 1.0)
        
        # Note 3. Dynamic curvature estimation (Turned on)
        L_est = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, np.inf)
        L_guesses = []
        
        is_guessed = 0
        n_monotone_step = 0
        
        if L_est != np.inf and lr == -1: 
            lr = 10.0 / L_est if is_convex else 2.0 / L_est
        
        if lr == -1:
            lr = 0.1
        
        lr_orig = lr
        
        G = None
        
        # Scaling matrix learning buffer
        if version == ALG_HDM_VERSION_DIAG:
            G = np.zeros_like(x)
            P = np.zeros_like(x)
        elif version == ALG_HDM_VERSION_MATRIX:
            G = np.zeros((x.shape[0], x.shape[0]))
            P = np.zeros((x.shape[0], x.shape[0]))
        elif version == ALG_HDM_VERSION_SCALAR:
            G = 0.0
            P = 0.0
        else:
            raise ValueError("Unknown version of hypergradient descent")
        
        # Momentum learning buffer
        # Note 4. Momentum initialization strategy (0.95)
        if beta_version == ALG_HDM_VERSION_DIAG:
            beta = np.zeros_like(x) + 0.95
            Gm = np.zeros_like(x) + 1e-04
        elif beta_version == ALG_HDM_VERSION_SCALAR:
            beta = 0.95
            Gm = 0.0
        else:
            raise ValueError("Unknown version of hypergradient descent")
        
        # Counters
        n_func_evals = 0
        n_grad_evals = 0
        n_iter = 0

        # Statistics
        fvals = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)
        
        x_old = x
        
        gx = grad_f(x)
        n_grad_evals += 1
        fx = f(x)
        gtmp = np.zeros_like(gx)
        ftmp = 0.0
        # omega = 0.0
        
        adagrad_eps = 1e-12
        
        for i in range(max_iter):
        
            grad_norm = np.linalg.norm(gx)
            grad_norm_inf = np.linalg.norm(gx, ord=np.inf)
            n_iter += 1
            
            # Save info for stats
            fvals[i] = fx  # function value
            gnorms[i] = grad_norm_inf
            
            if is_log:
                print("%d: f=%f, |g|_inf=%f, beta=%f, lr=%f" % (i, fx, grad_norm_inf, np.mean(beta), lr))
            
            # Check stopping condition
            if grad_norm_inf < tol:
                break
            
            # Update the primal iterate
            if version == ALG_HDM_VERSION_MATRIX:
                xtmp = x - P.dot(gx) + beta * (x - x_old)
            else:
                xtmp = x - P * gx + beta * (x - x_old)
                
            ftmp = f(xtmp)
            n_func_evals += 1
            gtmp = grad_f(xtmp)
            n_grad_evals += 1
            
            diff_norm_sqr = np.linalg.norm(x - x_old) ** 2
            
            if is_guessed:
                gnorm_eps = 0.25 * diff_norm_sqr * L_est ** 2
            else:   
                gnorm_eps = 1e-12
            
            # P_old = P
            if version == ALG_HDM_VERSION_DIAG:
                gr = - gtmp * gx / (grad_norm ** 2 + gnorm_eps)
                # gr += omega * L_est * (P * gx * gx - beta * gx * (x - x_old)) / (grad_norm ** 2 + gnorm_eps)
                G += gr ** 2
                P -= lr * gr / (np.sqrt(G) + adagrad_eps)
            elif version == ALG_HDM_VERSION_MATRIX:
                gr = - np.outer(gtmp, gx) / (grad_norm ** 2 + gnorm_eps)
                G += gr ** 2
                P -= lr * gr / (np.sqrt(G) + adagrad_eps)
            elif version == ALG_HDM_VERSION_SCALAR:
                gr = - np.dot(gtmp, gx) / (grad_norm ** 2 + gnorm_eps)
                G += gr ** 2
                P -= lr * gr / (np.sqrt(G) + adagrad_eps)
            else:
                raise ValueError("Unknown version of hypergradient descent")
            
            if is_convex:
                P = np.clip(P, 0.0 / L_est, 1e+10 / L_est)

            if beta_version == ALG_HDM_VERSION_DIAG:
                gm = gtmp * (x - x_old) / (grad_norm ** 2 + gnorm_eps)
                Gm += gm ** 2
                beta -= beta_lr * gm / (np.sqrt(Gm) + adagrad_eps)
                beta = np.clip(beta, 0.0, 0.9995)
            else:
                gm = (np.dot(gtmp, x - x_old)) / (grad_norm ** 2 + gnorm_eps)
                # gm += omega * L_est * (beta * diff_norm_sqr - np.dot(x - x_old, P_old * gx)) / (grad_norm ** 2 + gnorm_eps)
                Gm += gm ** 2
                beta -= beta_lr * gm / (np.sqrt(Gm) + adagrad_eps)
                beta = min(max(beta, 0.0), 0.9995)
            
            if n_iter % 50 == 49 and not is_guessed:
                if is_convex:
                    lr = np.maximum(lr * 0.8, lr_orig * 1e-03)
                else:
                    lr = np.maximum(lr * 0.05, lr_orig * 1e-02)
            
            # Note 5. Whether to reset momentum to 0 when a null step is taken (No)
            x_old = x
            if ftmp < fx:
                # Note 3-2. How to estimate the curvature (Using the ratio)
                diff_norm = np.linalg.norm(xtmp - x)
                if diff_norm > 1e-03:
                    curve_est = np.linalg.norm(gtmp - gx) / diff_norm
                    L_guesses.append(curve_est)

                # curve_est = 0.5 * (grad_norm ** 2) / (fx - ftmp)
                x = xtmp
                fx = ftmp
                gx = gtmp
                n_monotone_step += 1
            else:
                n_monotone_step -= 2
                n_monotone_step = max(0, n_monotone_step)
                    
            # Estimate the curvature
            # Note 3-1. Number of curvature estimates to collect (100)
            if n_monotone_step > max(min(max_iter / 4, 200), 20) and not is_guessed:
                # Note 3-3. How to use the curvature estimation (arithmetic mean)
                L_guess = np.mean(L_guesses)
                L_est = L_guess
                # Average with initial estimate
                # L_guess = (L_guess ** 0.8) * (L_est ** 0.2)
                lr = 1.0 / L_guess
                lr_orig = lr
                G = G * 0.0
                is_guessed = 1
                
        # If we ended early, fill trailing stats
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
    
    params = hdm_params
    params[ALG_HDM_LR_VERSION] = ALG_HDM_VERSION_DIAG
    params[ALG_HDM_BETA_VERSION] = ALG_HDM_VERSION_SCALAR
    params[ALG_HDM_LEARNING_RATE] = 0.25
    params[ALG_HDM_BETA_LEARNING_RATE] = 1.0
    params[ALG_UNIVERSAL_PARAM_L_EST] = 4.0
    
    # Initialize the optimizer
    hdm = HyperGradientDescent(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = hdm.optimize(x_init, f, grad_f)
    
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