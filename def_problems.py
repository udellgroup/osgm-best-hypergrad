import numpy as np
import scipy.sparse as sp
from scipy.special import expit
try:
    import pycutest
except:
    pass
from typing import Tuple
from sklearn.datasets import load_svmlight_file

# Define gradient and losses for different problems
# Linear regression utilities
def get_linear_regression_problem(A, b) -> Tuple[int, callable, callable]:
    
    f = lambda w: 0.5 * np.linalg.norm(A.dot(w) - b) ** 2
    grad_f = lambda w: A.T.dot(A.dot(w) - b)
    
    return A.shape[1], f, grad_f

# Read from LIBSVM dataset
def read_smoothed_svm_problem_from_libsvm(file_path: str = None, reg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    
    X, y = load_svmlight_file(file_path)
    return get_smoothed_svm_problem(X=X, y=y, reg=reg)

# Smoothed SVM utilities
def get_smoothed_svm_problem(X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> Tuple[float, int, callable, callable]:
    
    ones_column = np.ones((X.shape[0], 1))
    X_extended = sp.hstack([X, -ones_column])
    X_extended = X_extended.multiply(y.reshape(-1, 1))
    
    f = lambda w: smoothed_svm_loss(w, X_extended) + 0.5 * reg * np.linalg.norm(w) ** 2
    grad_f = lambda w: smoothed_svm_loss_grad(w, X_extended) + reg * w
    
    # Get the largest singular value of the matrix
    L_est = sp.linalg.svds(X_extended, k=1, return_singular_vectors=False)[0] ** 2 + reg
    # L_est = sp.linalg.norm(X_extended, 'fro') ** 2
    
    return L_est, X_extended.shape[1], f, grad_f

# Define the smoothed SVM loss
def smoothed_svm_loss(w: np.ndarray, X_scaled: np.ndarray) -> float:

    logits = X_scaled.dot(w) 
    hinge_loss = np.maximum(0, 1 - logits)
    return 0.5 * np.sum(hinge_loss ** 2)

# Gradient of smoothed SVM
def smoothed_svm_loss_grad(w: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:

    logits = X_scaled.dot(w)
    hinge_loss = np.maximum(0, 1 - logits)
    grad = - X_scaled.T.dot(hinge_loss)
    return grad

def read_logistic_problem_from_libsvm(file_path: str = None, reg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        
    X, y = load_svmlight_file(file_path)
    return get_logistic_problem(X=X, y=y, reg=reg)

# Logistic regression utilities
# Extract a logistic regression problem
def get_logistic_problem(X: np.ndarray, y: np.ndarray, reg: float = 0.0) -> Tuple[int, callable, callable]:

    ones_column = np.ones((X.shape[0], 1))
    X_extended = sp.hstack([X, ones_column])
    f = lambda w: logistic_loss(w, X_extended, y) + 0.5 * reg * np.linalg.norm(w) ** 2
    grad_f = lambda w: logistic_loss_grad(w, X_extended, y) + reg * w
    
    L_est = sp.linalg.svds(X_extended, k=1, return_singular_vectors=False)[0] ** 2 + reg
    # L_est = sp.linalg.norm(X_extended, 'fro') ** 2
    
    return L_est, X_extended.shape[1], f, grad_f

# Define the logistic loss
def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    
    logits = -y * (X @ w)
    # logaddexp(0, t) = log(1 + exp(t)) computed in log-space:
    return np.logaddexp(0.0, logits).sum()

# Gradient of logistic regression
def logistic_loss_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    
    logits = - y * (X.dot(w))
    # sigma = 1 / (1 + exp(logits))
    # derivative wrt w is: 1/N * X^T [ -y * sigma ]
    sigma  = expit(logits)
    grad = X.T.dot(-y * sigma)
    return grad

# CuTEST utilities
def get_cutest_problem(prob_i: int) -> Tuple[float, int, callable, callable, np.ndarray, str]:
    
    probs = sorted(pycutest.find_problems(constraints='unconstrained'))
    prob_name = probs[prob_i]
    properties = pycutest.problem_properties(prob_name)
    
    if not properties['n'] == 'variable' and properties['n'] > 1000:
        raise ValueError(f"Problem size {properties['n']} too large for this example")
    
    prob = pycutest.import_problem(prob_name, quiet=True)
    n = prob.n
    
    if n > 1000:
        raise ValueError(f"Problem size {n} too large for this example")
    
    f = prob.obj
    grad_f = prob.grad
    x0 = prob.x0
    
    H_init = prob.sphess(x0)
    L_est = sp.linalg.svds(H_init, k=1, return_singular_vectors=False)[0]
    
    # Estimate the gradient Lipschitz constant
    # ptb = np.random.randn(n)
    # g1 = grad_f(x0)
    # g2 = grad_f(x0 + ptb)
    # L_est = np.linalg.norm(g1 - g2) / np.linalg.norm(ptb)
    
    return L_est, n, f, grad_f, x0, prob.name


if __name__ == "__main__":
    
    fname = "problems/a1a"
    ns, fs, gs = read_smoothed_svm_problem_from_libsvm(fname, reg=0.1338)
    nl, fl, gl = read_logistic_problem_from_libsvm(fname, reg=0.1)
    
    A = np.random.randn(10, 5)
    b = np.random.randn(10)
    nq, fq, gq = get_linear_regression_problem(A, b)

    w = np.ones(nq)
    
    print(fq(w))
    print(gq(w))
    
    pass