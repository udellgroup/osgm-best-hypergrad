from bench_algo import *
from algo_config import *
import numpy as np
from scipy.stats import gmean, hmean
from typing import Tuple

class OSGMOptimizer(Optimizer):
    """
    Gradient Methods with Online Scaling
    Implement OSGM with momentum and two kinds of landscape actions

    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Constructor for the OSGM optimizer.
        """
        
        if params is None:
            params = {}
        
        self.stats = {}
        self.n_function_eval = 0
        self.n_grad_eval = 0
        self.linesearch_ratio = 2.0
        
        self.func = None
        self.grad = None
        
        super().__init__(params.get(ALG_UNIVERSAL_PARAM_NAME, "OSGM"), params)
        
    def _func_eval(self, x: np.ndarray) -> float:
        
        if self.func is None:
            raise NotImplementedError("No function value oracle")    
        
        self.n_function_eval += 1
        
        return self.func(x)
    
    def _grad_eval(self, x: np.ndarray) -> np.ndarray:
        
        if self.grad is None:
            raise NotImplementedError("No gradient oracle")
        
        self.n_grad_eval += 1
        
        return self.grad(x)
        
    def _osgm_potential(self, z: Tuple[np.ndarray], omega: float, fval: float = None) -> float:
        """
        Evaluate the Polyak potential function for OSGM:
        
        \phi(x, x^-) = f(x) + 0.5 * \omega \|x - x^-\|^2

        """
        
        if fval is None:
            fval = self._func_eval(z[0]) 
            
        return fval + 0.5 * omega * np.linalg.norm(z[0] - z[1]) ** 2, fval
    
    def _osgm_get_alg_params(self, L_est: float) -> Tuple[float, float, float, float]:
        """
        Get omega and tau for OSGM
        """
        
        lr_P = 1.0 / L_est
        lr_beta = min(1.0, L_est)
        omega = 0.0 * L_est
        tau = 0.5 * L_est ** 2
        
        return omega, tau, lr_P, lr_beta
    
    def _osgm_fval_linesearch(self, fval: float, x: np.ndarray, g: np.ndarray, L_est: float, P=None) -> np.ndarray:
        """
        Implement Armijo line-search for function value in OSGM
        
        """
        
        if P is None:
            P = np.ones_like(x)
            eta = self.linesearch_ratio / L_est
        else:
            # P = P
            eta = 1.0
            
        gnorm_sqr = g.dot(P * g)
        is_linesearch_successful = False
        
        while eta > 1e-06:
            
            x_new = x - eta * P * g
            fval_new = self._func_eval(x_new)

            if fval_new <= fval - 0.5 * eta * gnorm_sqr:
                is_linesearch_successful = True
                break
            
            eta = eta * 0.8
            
        self.linesearch_ratio = min(eta * L_est * 2.0, 100)
        
        if not is_linesearch_successful:
            x_new = x
            
        return x_new

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
            
            
        Notes
        -----
        
        The implementation follows the theory developed in: 
        
        Gao, W., Chu, Y. C., Ye, Y., & Udell, M. (2025). Gradient Methods with Online Scaling Part I. Theoretical Foundations
        Chu, Y. C., Gao, W., Ye, Y., & Udell, M. (2025). Gradient Methods with Online Scaling Part II. Practical Aspects
        
        """
        
        # Register callback
        self.func = f
        self.grad = grad_f
        
        # Extract parameters
        tol = self.params.get(ALG_UNIVERSAL_PARAM_TOL, 1e-06)
        max_iter = self.params.get(ALG_UNIVERSAL_PARAM_MAXITER, 1000)
        is_log = self.params.get(ALG_OSGM_LOGGING, False)
        is_convex = not self.params.get(ALG_OSGM_NONCONVEX, True)
        
        use_lookahead = False
        is_adagrad = True
        is_L_est_dynamic = True
        
        version = self.params.get(ALG_OSGM_LR_VERSION, ALG_OSGM_VERSION_DIAG)
        
        L_guesses = []
        n_guesses = 0
        
        n_monotone_step = 0
        
        # Estimate Lipschitz constant by sampling in a unit ball
        # L_est = self.params.get(ALG_UNIVERSAL_PARAM_L_EST, 1.0) / 100.0
        L_est = 0.0
        g = self._grad_eval(x)
        
        for _ in range(3):
            delta = np.random.randn(*x.shape)
            delta /= np.linalg.norm(delta)
            delta *= 1e-04
            g_diff = self._grad_eval(x + delta) - g
            L_est = max(L_est, np.linalg.norm(g_diff) / 1e-04)
            
        if np.isinf(L_est) or np.isnan(L_est):
            L_est = 1e-05
        
        # Stepsize learning
        if version == ALG_OSGM_VERSION_DIAG:
            G = np.zeros_like(x)
            P = np.zeros_like(x) + 0.00 / L_est
        elif version == ALG_OSGM_VERSION_SCALAR:
            G = 0.0
            P = 0.0 + 0.00 / L_est
        else:
            raise NotImplementedError(f"Version {version} is not implemented for OSGM")
        
        # Momentum learning
        beta = 0.95
        Gm = 0.0
        
        x_prev = x.copy()
        omega, tau, lr_P, lr_beta = self._osgm_get_alg_params(L_est)

        pot_val, fval = self._osgm_potential((x, x_prev), omega, fval=None)

        # Statistics
        fvals = np.zeros(max_iter)
        gnorms = np.zeros(max_iter)
        is_descent_step = False
        
        # Counters
        n_iter = 0
        
        for i in range(max_iter):
            
            grad_norm = np.linalg.norm(g)
            grad_norm_inf = np.linalg.norm(g, ord=np.inf)
            n_iter += 1
            
            # Save info for stats
            fvals[i] = fval
            gnorms[i] = grad_norm_inf

            # Check stopping criterion
            if grad_norm_inf < tol:
                break
            
            x_mmtm = x - x_prev
            
            # Update the primal iterate
            xtmp = x - P * g + beta * x_mmtm
            gtmp = self._grad_eval(xtmp)
            
            grad_tmp_norm = np.linalg.norm(gtmp)
            
            # Lookahead
            is_lookahead = False
            if use_lookahead and grad_tmp_norm > 2.0 * grad_norm:
                xlookahead = self._osgm_fval_linesearch(self._func_eval(xtmp), xtmp, gtmp, L_est, P)
                is_lookahead = True
            else:
                xlookahead = xtmp
            
            # Null step
            new_pot_val, fvaltmp = self._osgm_potential((xlookahead, x), omega, fval=None)
            is_descent_step = False
            x_prev = x
            
            if fvaltmp <= fval:
                pot_val = new_pot_val
                fval = fvaltmp
                is_descent_step = True
            
            # Hypergradient update
            diff_norm_sqr = np.linalg.norm(x_mmtm) ** 2
            
            if is_L_est_dynamic:
                gnorm_eps = 1e-12
            else:
                gnorm_eps = 0.5 * tau * diff_norm_sqr
                
            gpot = gtmp - omega * (P * g - beta * x_mmtm)
            
            if version == ALG_OSGM_VERSION_DIAG:
                gr = - gpot * g / (grad_norm ** 2 + gnorm_eps)
            else:
                gr = - np.dot(gpot, g) / (grad_norm ** 2 + gnorm_eps)
                
            gm = np.dot(gpot, x_mmtm) / (grad_norm ** 2 + gnorm_eps)
            G = G + gr * gr
            Gm = Gm + gm * gm
            
            if is_adagrad:
                P -= lr_P * gr / (np.sqrt(G) + 1e-12)
                beta -= lr_beta * gm / (np.sqrt(Gm) + 1e-12)
            else:
                P -= lr_P * gr
                beta -= lr_beta * gm
            
            P = np.clip(P, 0.0 / L_est, 1e+05 / L_est) 
            beta = max(0.0, min(beta, 0.9995))
            
            # Dynamic smoothness constant estimate
            if is_descent_step:
                pot_val = new_pot_val
                diff_norm = np.linalg.norm(xtmp - x)
                
                if diff_norm > 1e-03:
                    curv_est = np.linalg.norm(gtmp - g) / diff_norm
                    L_guesses.append(curv_est)
                    
                x = xlookahead
                n_monotone_step += 1
                
                if is_lookahead:
                    g = self._grad_eval(x)
                else:
                    g = gtmp
            else:
                n_monotone_step -= 2
                n_monotone_step = max(n_monotone_step, 0)
                    
            if is_log:
                print("%d: f=%f, |g|_inf=%f, lr=%f, lr_beta=%f  Pmax:%f beta: %f L_est=%f Mmtm=%f [%d]" % 
                      (i, fval, grad_norm_inf, lr_P, lr_beta, np.linalg.norm(P, ord=np.inf), beta, L_est, gm, is_descent_step))

            if n_monotone_step > max(min(max_iter / 4, 200), 20) and is_L_est_dynamic:
                n_guesses += 1
                L_est = np.mean(L_guesses)  
                L_guesses = []
                omega, tau, lr_P, lr_beta = self._osgm_get_alg_params(L_est)
                is_L_est_dynamic = False
                use_lookahead = True
                G = G * 0.0
                
            if n_iter % 50 == 49 and is_L_est_dynamic:
                lr_P = max(0.8 * lr_P, 1e-03 / L_est)
                
            if self.n_grad_eval > max_iter - 1:
                break

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
            ALG_STATS_FEVALS: self.n_function_eval,
            ALG_STATS_GEVALS: self.n_grad_eval
        }
        
        self.stats = stats
        return stats
    
    def get_optimizer_stats(self):
        """
        Get the optimizer statistics
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer
        """
        return self.stats
    
    
if __name__ == "__main__":
    
    params = osgm_params
    params[ALG_OSGM_LR_VERSION] = ALG_OSGM_VERSION_DIAG
    
    osgm = OSGMOptimizer(params)
    
    # Initial guess
    x_init = np.array([0.0, 0.0])
    
    # Run optimization
    stats = osgm.optimize(x_init, f, grad_f)
    
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