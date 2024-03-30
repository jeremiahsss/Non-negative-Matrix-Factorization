import numpy as np
from numpy.linalg import solve


def nmf_frobenius(X, K, max_iter=100, tol=1e-4):
    """
    NMF using Frobenius norm as the reconstruction error.
    
    Parameters:
    - X: Input data matrix.
    - K: Number of components.
    - max_iter: Maximum number of iterations.
    - epsilon: convergence condition.
    
    Returns:
    - D, R: Factorized matrices.
    """
    np.random.seed(0)
    m, n = X.shape
    
    # Initialize D and R with non-negative random values
    D = np.abs(np.random.randn(m, K))
    R = np.abs(np.random.randn(K, n))
    
    # Previous error
    prev_error = float('inf')
    
    for _ in range(max_iter):
        # Update R given fixed D
        R = solve(D.T @ D + np.eye(K) * 1e-5, D.T @ X) 
        R[R < 0] = 0  # Ensure non-negativity
        
        # Update D given fixed R
        D = solve(R @ R.T + np.eye(K) * 1e-5, R @ X.T).T
        D[D < 0] = 0  # Ensure non-negativity
        
        # Compute the reconstruction error
        error = np.linalg.norm(X - D @ R, 'fro')
        
        # Check convergence
        if np.abs(prev_error - error) < tol:
            break
        
        prev_error = error
    
    return D, R
