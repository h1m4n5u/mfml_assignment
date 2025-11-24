def gaussian_elimination(A, b):
    n = len(A)

    # Build augmented matrix Ab = [A | b]
    Ab = [A[i] + [b[i]] for i in range(n)]

    # Forward elimination
    for i in range(n):

        # Pivot handling: If pivot is zero → swap
        if Ab[i][i] == 0:
            for k in range(i+1, n):
                if Ab[k][i] != 0:
                    Ab[i], Ab[k] = Ab[k], Ab[i]
                    break
            else:
                raise ValueError("Matrix is singular — no unique solution.")

        # Eliminate rows below
        for k in range(i+1, n):
            factor = Ab[k][i] / Ab[i][i]
            for j in range(i, n+1):
                Ab[k][j] -= factor * Ab[i][j]

    # Back substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = sum(Ab[i][j] * x[j] for j in range(i+1, n))
        x[i] = (Ab[i][n] - s) / Ab[i][i]

    return x


# --- INPUT HANDLING ---
n = int(input("Enter number of equations: "))

A = []
print("\nEnter the coefficient matrix row by row:")
for i in range(n):
    row = list(map(float, input(f"Enter row {i+1}: ").split()))
    if len(row) != n:
        raise ValueError("Each row must contain exactly n values.")
    A.append(row)

b = []
print("\nEnter the RHS values:")
for i in range(n):
    rhs = float(input(f"Enter RHS b{i+1}: "))
    b.append(rhs)

# Solve
solution = gaussian_elimination(A, b)

print("\nSolution:")
for i, val in enumerate(solution):
    print(f"x{i+1} = {val}")
