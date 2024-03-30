import numpy as np


def L1_loss(X, D, R):
    """
    Args: 
        X: Input matrix
        D, R: Factorized matrix
    Returns:
        abs_error: Absolute reconstruction error
    """
    # Absolute error (L1 loss)
    abs_error = np.abs(X - D @ R)

    # Replace zero elements with machine epsilon to avoid zero division in subsequent calculation
    abs_error[abs_error == 0] = np.finfo(np.float32).eps

    return abs_error


def huber_weight(X, D, R):
    """
    Args:
        X: Input matrix
        D, R: Factorized matrix

    Returns:
        w = Huber loss update weight
    """
    # Absolute error (L1 loss)
    abs_error = L1_loss(X, D, R)

    # Set delta as median of absolute error
    delta = np.median(abs_error)

    # The update weight for Huber loss 
    w = np.where(abs_error < delta, 1, delta / abs_error)

    return w


def huber_loss(X, D, R):
    """
    Huber loss explanation: 
    Huber loss function uses a squared term if the absolute element-wise error 
    falls below delta and a delta-scaled L1 term otherwise

    Args:
        X: Input matrix
        D, R: Factorized matrices

    Returns:
        huber loss
    """
    # Absolute error (L1 loss)
    abs_error = L1_loss(X, D, R)

    # Squared error
    squared_error = np.square(abs_error)

    # Set delta as median of absolute error
    delta = np.median(abs_error)

    return np.sum(np.where(abs_error < delta,
                            squared_error,
                            2 * delta * abs_error - (delta ** 2)))


def huber_nmf(X, K, max_iter=100, tol = 1e-4):
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

    # pre_loss
    pre_loss = np.inf
    for _ in range(max_iter):
        # Compute the update weight for Huber loss
        w = huber_weight(X, D, R)
        
        # Compute numerator of D
        D_numerator = (w * X) @ R.T
        # Compute the denominator of D
        D_denominator = (w * (D @ R)) @ R.T
        # Replace zero elements with machine epsilon to avoid zero division
        D_denominator[D_denominator == 0] = eps
        # Update D
        D = D * D_numerator / D_denominator

        # Compute numerator of R
        R_numerator = D.T @ (w * X)
        # Compute the denominator of R
        R_denominator = D.T @ (w * (D @ R))
        # Replace zero elements with machine epsilon to avoid zero division
        R_denominator[R_denominator == 0] = eps
        # Update R
        R = R * R_numerator / R_denominator

        # Calculate the Huber loss
        loss = huber_loss(X, D, R)

        # Check for convergence
        if np.abs(pre_loss - loss) < tol:
            break
        
        pre_loss = loss


    return D, R