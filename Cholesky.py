import numpy as np

def cholesky_decomposition(A):
    """
    Computes the Cholesky decomposition of a symmetric, positive-definite matrix A.

    Args:
        A (np.ndarray): The input matrix (n x n).

    Returns:
        np.ndarray: The lower triangular matrix L such that A = L @ L.T,
                    or None if the matrix is not positive definite.
    """
    n = A.shape[0]
    L = np.zeros((n, n))

    # Basic check for symmetry (optional, but good practice)
    if not np.allclose(A, A.T):
        print("Error: Input matrix must be symmetric for Cholesky Decomposition.")
        # Attempt to make it symmetric by averaging (A + A.T) / 2
        # A = (A + A.T) / 2 # Can be used if slight asymmetry is due to floating point

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                # Diagonal element calculation: L[i, i] = sqrt(A[i, i] - sum(L[i, k]**2 for k=0 to i-1))
                sum_sq = np.sum(L[i, :j]**2)
                
                term = A[i, i] - sum_sq
                
                if term <= 0:
                    print(f"Matrix is not positive definite. Cannot compute Cholesky decomposition at L[{i}, {i}].")
                    return None
                
                L[i, i] = np.sqrt(term)
            else:
                # Off-diagonal element calculation: L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] for k=0 to j-1)) / L[j, j]
                sum_prod = np.sum(L[i, :j] * L[j, :j])
                
                # Check for division by zero (shouldn't happen if L[j,j] > 0)
                if np.isclose(L[j, j], 0.0):
                    # This implies the matrix is not positive definite
                    print(f"Division by zero at L[{j}, {j}]. Matrix is not positive definite.")
                    return None
                    
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]

    return L

# --- Example Usage ---
if __name__ == '__main__':
    # Define a symmetric positive definite matrix
    # A = [[4, 12, -16],
    #      [12, 37, -43],
    #      [-16, -43, 98]]
    # This matrix is used in many Cholesky examples.
    A_input = np.array([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ], dtype=float)

    print("--- Input Matrix A ---")
    print(A_input)

    L_result = cholesky_decomposition(A_input)

    if L_result is not None:
        print("\n--- Resulting Lower Triangular Matrix L ---")
        print(L_result)
        
        # Calculate and print the transpose (L^T)
        L_transpose = L_result.T
        print("\n--- Resulting Upper Triangular Matrix L^T (L Transpose) ---")
        print(L_transpose)

        # Verification: L @ L.T should be close to A
        A_reconstructed = np.dot(L_result, L_result.T)
        print("\n--- Verification (L @ L^T) ---")
        print(A_reconstructed)
        
        if np.allclose(A_input, A_reconstructed):
            print("\nVerification successful: A is close to L @ L^T")
        else:
            print("\nVerification failed.")