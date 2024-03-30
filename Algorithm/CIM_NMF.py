import numpy as np
np.random.seed(0)

def cim_weight(X, D, R):
    """
    Args:
        X: Input matrix (M x N)
        D, R: Factorized matrix 
    Returns:
        w: Correntropy Induced Metric weight
    """
    # Compute square error
    square_error = np.square(X - (D @ R))

    # Compute two sigma square 
    two_sigma_square = 2 * np.mean(square_error)

    # Compute CIM weight
    W = np.exp(- square_error / two_sigma_square)

    return W

def cim_loss(X, D, R):
    """
    Args:
        X: Input matrix (M x N)
        D, R: Factorized matrix 
    Returns:
        loss: Correntropy Induced Metric loss 
    """
    # Compute square error
    square_error = np.square(X - (D @ R))

    # Compute sigma
    sigma = np.sqrt(np.mean(square_error))

    # Compute two sigma square
    two_sigma_square = 2 * np.mean(square_error)

    # Compute the gaussian kernel
    gauss_kernel = 1. / np.sqrt(2 * np.pi * sigma) * np.exp(- square_error / two_sigma_square)

    # Compute the CIM loss
    loss = np.sum(1. - gauss_kernel)
    
    return loss


def cim_nmf(X, K, max_iter=100, tol = 1e-4):
    """
    Args:
        X: Input matrix (M x N)
        K: Number of components
        max_iter: Maximum number of iteration
        tol: convergence threshold
    
    Returns
        D, R: Factorized matrices
    """

    np.random.seed(0)
    # Identify the shape of the input matrix
    M, N = X.shape

    # Initialize matrices D and R
    D = np.abs(np.random.randn(M, K))
    R = np.abs(np.random.randn(K, N))
    
    # Initiate machine epsilon constant
    eps = np.finfo(np.float32).eps
    pre_loss = np.inf

    for _ in range(max_iter):
        # Compute the update weight for Huber loss
        W = cim_weight(X, D, R)
        
        # Compute numerator of D
        D_numerator = (W * X) @ R.T
        # Compute the denominator of D
        D_denominator = (W * (D @ R)) @ R.T
        # Replace zero elements with machine epsilon to avoid zero division
        D_denominator[D_denominator == 0] = eps
        # Update D
        D = D * D_numerator / D_denominator

        # Compute numerator of R
        R_numerator = D.T @ (W * X)
        # Compute the denominator of R
        R_denominator = D.T @ (W * (D @ R))
        # Replace zero elements with machine epsilon to avoid zero division
        R_denominator[R_denominator == 0] = eps
        # Update R
        R = R * R_numerator / R_denominator

        # Calculate the Huber loss
        loss = cim_loss(X, D, R)

        # Check for convergence
        if np.abs(pre_loss - loss) < tol:
            break
        
        pre_loss = loss
        
    return D, R













    