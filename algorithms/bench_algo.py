from typing import List, Dict, Any
import numpy as np

# Abstract benchmarking class for algorithms
ALG_STATS_ITERATIONS = "Iterations"
ALG_STATS_OPTIMAL_VALUE = "Optimal value"
ALG_STATS_OPTIMAL_SOL = "Optimal point"
ALG_STATS_RUNNING_TIME = "Time"
ALG_STATS_FUNCVALS = "Function values"
ALG_STATS_GNORMS = "Gradient norms"
ALG_STATS_FEVALS = "Function evaluations"
ALG_STATS_GEVALS = "Gradient evaluations"

class Optimizer(object):
    
    def __init__(self, algo: str = "Algorithm", params: Dict[str, Any] = None):
        
        self.algo = algo
        self.params = params
        
    def optimize(self, x: np.ndarray, f: callable, grad_f: callable) -> List[float]:
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
        List[float]
            Optimal point
        """
        
        raise NotImplementedError("Method optimize is not implemented")
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """
        Get the optimizer statistics
        
        Returns
        -------
        Dict[str, Any]
            Statistics of the optimizer
        """
        
        raise NotImplementedError("Method get_optimizer_stats is not implemented")
    
# Toy example
def f(x: np.ndarray) -> float:
        return np.sum(x[0]**2 + 2.0 * x[1] ** 2) - np.array([2, 4]) @ x
    
def grad_f(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * x[0] - 2.0, 4.0 * x[1] - 4.0])
    
    