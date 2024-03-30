import numpy as np
def l1_norm_nmf(X, K, max_iter=100, tol=1e-5):
    """
    NMF using L1 norm as the reconstruction error.
    
    Parameters:
    - X: Input data matrix.
    - K: Number of components.
    - max_iter: Maximum number of iterations.
    - tol: convergence condition.
    
    Returns:
    - D, R: Factorized matrices.
    """
    # Set random seed to make the result reproducible
    np.random.seed(0)
    
    m, n = X.shape
    # Initialize D and R with non-negative random values
    D = np.abs(np.random.rand(m, K))
    R = np.abs(np.random.rand(K, n))
    # Previous error
    pre_error = np.inf
    eps = 1e-7
    for _ in range(max_iter):
        
        # Update D and R
        D = D * np.dot(X, R.T) / (np.dot(D,R).dot(R.T) + eps)
        R = R * np.dot(D.T, X) / (np.dot(D.T, D).dot(R) + eps)

        # Compute L1 loss
        error = np.sum(np.abs(X - np.dot(D, R)))

        # Check for convergence
        if np.abs(pre_error - error) < tol:
            break
        pre_error = error
        
    return D, R
