# Benchmark different algorithms on LIBSVM datasets
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

import numpy as np

from algorithms.bench_algo import *
from algorithms.adagrad import AdaGrad
from algorithms.adam import Adam
from algorithms.agdcvx import AcceleratedGradientCvx
from algorithms.agdscvx import AcceleratedGradientStrongCvx
from algorithms.bfgs import SciPyBFGS
from algorithms.lbfgs import SciPyLBFGS
from algorithms.gd import GradientDescent
from algorithms.gdheavyball import GradientDescentHeavyBall
from algorithms.hdm import HyperGradientDescent
from algorithms.osgm import OSGMOptimizer

from algorithms.algo_config import *

from algorithms.hypertuner import HyperGrid

# Load the dataset
from def_problems import read_logistic_problem_from_libsvm

from utils import plot_descent_curves

import argparse
parser = argparse.ArgumentParser(description='Benchmark algorithms on different problems and datasets')
parser.add_argument('--dataset', type=str, default='./problems/a1a', help='Path to the dataset')
parser.add_argument('--plot_curves', type=int, default=1, help='Whether to plot figures')
parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of gradient evaluations')
parser.add_argument('--tol', type=float, default=1e-04, help='Tolerance of gradient norm')

if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.dataset
    plot_curves = args.plot_curves
    max_iter = args.max_iter
    tol = args.tol
    
    np.random.seed(20250701)
    reg = 0.0
    L_est, n, fval, grad = read_logistic_problem_from_libsvm(file_path=dataset, reg=reg)
    
    # Get optimal value benchmark
    bench_bfgs_params = lbfgs_params.copy()
    bench_bfgs_params[ALG_UNIVERSAL_PARAM_TOL] = 1e-08
    bench_bfgs_params[ALG_UNIVERSAL_PARAM_MAXITER] = 2000
    bench_optimizer = SciPyLBFGS(bench_bfgs_params)
    
    # L-BFGS with different memory sizes
    lbfgs_params_m1 = lbfgs_params.copy()
    lbfgs_params_m1[ALG_LBFGS_MEMORY_SIZE] = 1
    lbfgs_params_m2 = lbfgs_params.copy()
    lbfgs_params_m2[ALG_LBFGS_MEMORY_SIZE] = 2
    lbfgs_params_m3 = lbfgs_params.copy()
    lbfgs_params_m3[ALG_LBFGS_MEMORY_SIZE] = 3
    lbfgs_params_m4 = lbfgs_params.copy()
    lbfgs_params_m4[ALG_LBFGS_MEMORY_SIZE] = 4
    lbfgs_params_m5 = lbfgs_params.copy()
    lbfgs_params_m5[ALG_LBFGS_MEMORY_SIZE] = 5
    lbfgs_params_m10 = lbfgs_params.copy()
    lbfgs_params_m10[ALG_LBFGS_MEMORY_SIZE] = 10
    
    # Construct OSGM parameters
    osgm_params[ALG_OSGM_NONCONVEX] = True
    osgm_params[ALG_OSGM_LR_VERSION] = ALG_OSGM_VERSION_DIAG
    osgm_param_base = osgm_params.copy()
    
    # Construct HDM parameters
    hdm_params[ALG_HDM_NONCONVEX] = False
    hdm_params[ALG_HDM_LR_VERSION] = ALG_HDM_VERSION_DIAG
    hdm_params[ALG_HDM_BETA_VERSION] = ALG_HDM_VERSION_SCALAR
    hdm_param_base = hdm_params.copy()
    
    hdm_search_grid = {
        ALG_HDM_LEARNING_RATE: [0.1 / L_est, 1.0 / L_est, 10.0 / L_est, 100.0 / L_est],
        ALG_HDM_BETA_LEARNING_RATE: [1.0, 3.0, 5.0, 10.0, 100.0]
    }
    
    # Construct AdaGrad parameters
    adagrad_param_base = adagrad_params.copy()
    adagrad_search_grid = {
        ALG_ADAGRAD_LEARNING_RATE: [1e-03, 1e-02, 1e-01, 1.0, 10.0, 1 / L_est],
    }
    
    # Construct Adam parameters
    adam_param_base = adam_params.copy()
    adam_search_grid = {
        ALG_ADAM_LEARNING_RATE: [1e-03, 1e-02, 1e-01, 1.0, 10.0, 1 / L_est],
    }
    
    # Construct heavy ball parameters
    heavy_ball_base_params = gradient_descent_heavy_ball_params.copy()
    heavy_ball_search_grid = {
        ALG_HEAVY_BALL_MOMENTUM: [0.1, 0.5, 0.9, 0.99]
    }
    
    alg_list = {
        "GD": [GradientDescent, gradient_descent_params, None],
        "GD-HB": [GradientDescentHeavyBall, heavy_ball_base_params, heavy_ball_search_grid],
        "AGD-CVX": [AcceleratedGradientCvx, accelerated_gradient_descent_cvx_params, None],
        "AGD-SCVX": [AcceleratedGradientStrongCvx, accelerated_gradient_descent_cvx_params, None],
        "Adam": [Adam, adam_param_base, adam_search_grid],
        "AdaGrad": [AdaGrad, adagrad_param_base, adagrad_search_grid],
        "BFGS": [SciPyBFGS, bfgs_params, None],
        "L-BFGS-M1": [SciPyLBFGS, lbfgs_params_m1, None],
        "L-BFGS-M3": [SciPyLBFGS, lbfgs_params_m3, None],
        "L-BFGS-M5": [SciPyLBFGS, lbfgs_params_m5, None],
        "L-BFGS-M10": [SciPyLBFGS, lbfgs_params_m10, None],
        "OSGM": [OSGMOptimizer, osgm_param_base, None],
    }
    
    mu_est = reg if reg > 0 else 1e-04
    
    # Set algorithm parameters and initialize
    for algo in alg_list.keys():
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_L_EST] = L_est
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_TOL] = tol
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_MU_EST] = mu_est
        alg_list[algo][1][ALG_UNIVERSAL_PARAM_MAXITER] = max_iter
        algo_class = alg_list[algo][0] # type: Optimizer
        algo_param = alg_list[algo][1]
        algo_grid = alg_list[algo][2]
        alg_list[algo].append(HyperGrid(
                                param_base=algo_param, 
                                optimizer_cls=algo_class, 
                                param_grid=algo_grid))
    
    x0 = np.random.randn(n)
    x0 = x0 / np.linalg.norm(x0)
    
    stats_bench = bench_optimizer.optimize(x=x0, f=fval, grad_f=grad)
    opt_val = stats_bench[ALG_STATS_OPTIMAL_VALUE]

    # Print solver statistics
    print("================================================")
    print("%20s [S]  %10s  %10s" % ("Solver", "nFvalCall", "nGradCall"))
    # Run the algorithms
    for algo in alg_list.keys():
        optimizer = alg_list[algo][3]
        info = optimizer.search(x0=x0, f=fval, grad_f=grad)
        stats = info["best_stats"]
        opt_param = info["best_params"]
        alg_list[algo].append(stats)
        print("%20s [%d]  %10d  %10d" % (algo, stats[ALG_STATS_GEVALS] < max_iter - 1, 
                                         stats[ALG_STATS_FEVALS], stats[ALG_STATS_GEVALS]))
    print("================================================")
    
    data_name = dataset.split('/')[-1]
    data_name = data_name.split('.')[0]
    
    opt_val = min(np.min([alg_list[algo][4][ALG_STATS_OPTIMAL_VALUE] for algo in alg_list.keys()]), opt_val)
    opt_val = opt_val - 1e-12
            
    # Plot the descent curves
    if plot_curves:
        # Function value gap
        alg_descent_curves = {algo: alg_list[algo][4][ALG_STATS_FUNCVALS] - opt_val for algo in alg_list.keys()}
        plot_descent_curves(alg_descent_curves, 
                            use_log_scale=True, 
                            legend_loc="upper right",
                            ylabel="Function value gap",
                            title=f"{data_name}",
                            fname=os.path.join(".", "figures", f"{data_name}_objval_logistic.pdf"))
        
        # Gradient norm
        alg_descent_curves = {algo: alg_list[algo][4][ALG_STATS_GNORMS] for algo in alg_list.keys()}
        plot_descent_curves(alg_descent_curves, 
                            use_log_scale=True, 
                            legend_loc="upper right",
                            ylabel="Gradient Norm",
                            title=f"{data_name}",
                            fname=os.path.join(".", "figures", f"{data_name}_gnorm_logistic.pdf"))
        