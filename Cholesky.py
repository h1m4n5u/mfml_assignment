import math

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):

            # Summation for L[i][j]
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:  # Diagonal element
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][i] = math.sqrt(val)
            else:  # Off-diagonal element
                L[i][j] = (A[i][j] - s) / L[j][j]

    return L


# --- Input ---
n = int(input("Enter matrix size n: "))
A = [list(map(float, input().split())) for _ in range(n)]

# --- Cholesky ---
L = cholesky(A)

print("\nL matrix:")
for row in L:
    print(row)
