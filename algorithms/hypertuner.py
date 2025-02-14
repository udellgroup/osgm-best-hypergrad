import itertools
import numpy as np
from bench_algo import *
from typing import List, Dict, Any, Callable, Union


class HyperGrid(object):
    """
    This class takes:
      - An optimizer class,
      - A dictionary of parameter *ranges* (each entry is a list of possible values),
      - Objective function and gradient,
      - Initial point,
    and runs the optimizer for every combination of parameters.
    It returns the statistics for the parameter combination that yields the
    fewest number of iterations (ALG_STATS_ITERATIONS).
    """
    
    def __init__(self,
                 param_base: Dict[str, Any],
                 optimizer_cls: Any,
                 param_grid: Dict[str, List[Any]]):
        """
        Parameters
        ----------
        param_base : Dict[str, Any]
            The base parameter dictionary for the optimizer.
        optimizer_cls : class
            The optimizer class (e.g., GradientDescent)
        param_grid : Dict[str, List[Any]]
            A dict whose keys are parameter names and values are lists
            of possible values to try.
        """
        self.param_base = param_base
        self.optimizer_cls = optimizer_cls
        self.param_grid = param_grid
    
    def search(self,
               x0: np.ndarray,
               f: Callable[[np.ndarray], float],
               grad_f: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """
        Runs a Cartesian product over param_grid, returning the
        best stats (fewest iterations) and the associated parameters.
        
        Parameters
        ----------
        x0 : np.ndarray
            The initial guess for the optimizer.
        f : callable
            Objective function.
        grad_f : callable
            Gradient function.
        
        Returns
        -------
        best_result : Dict
            A dictionary containing:
                - "best_params": the best parameter dictionary
                - "best_stats": the stats dictionary from the optimizer
        """
        
        if self.param_grid is None:
            return {
                "best_params": self.param_base,
                "best_stats": self.optimizer_cls(self.param_base).optimize(x0.copy(), f, grad_f)
            }
        
        all_param_keys = list(self.param_grid.keys())
        # Build lists of lists for itertools.product
        all_param_values = [self.param_grid[key] for key in all_param_keys]
        
        best_stats = None
        best_params = None
        
        for combination in itertools.product(*all_param_values):
            
            # Build a param dict out of the current combination
            current_params = self.param_base.copy()
            for k, v in zip(all_param_keys, combination):
                current_params[k] = v
            
            # Instantiate the optimizer with current params
            optimizer = self.optimizer_cls(current_params)
            
            # Run optimization
            stats = optimizer.optimize(x0.copy(), f, grad_f)
            
            # Compare with best so far (fewest iterations)
            if (best_stats is None) or (stats[ALG_STATS_ITERATIONS] < best_stats[ALG_STATS_ITERATIONS]):
                best_stats = stats
                best_params = current_params
        
        return {
            "best_params": best_params,
            "best_stats": best_stats
        }
        