import math

def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))

def norm(v):
    return math.sqrt(dot(v, v))

def modified_gram_schmidt(A):
    m = len(A)
    n = len(A[0])
    
    # Convert rows to columns
    cols = [[A[i][j] for i in range(m)] for j in range(n)]
    
    Q = [[0.0]*n for _ in range(m)]
    R = [[0.0]*n for _ in range(n)]

    for j in range(n):
        v = cols[j][:]

        for i in range(j):
            qi = [Q[row][i] for row in range(m)]
            R[i][j] = dot(qi, v)
            v = [v[k] - R[i][j] * qi[k] for k in range(m)]

        R[j][j] = norm(v)
        qj = [v[k] / R[j][j] for k in range(m)]

        for row in range(m):
            Q[row][j] = qj[row]

    return Q, R


# ---------- USER INPUT ----------
m = int(input("Enter number of rows: "))
n = int(input("Enter number of columns: "))

A = []
for i in range(m):
    row = list(map(float, input(f"Enter row {i+1}: ").split()))
    A.append(row)

# ---------- Compute ----------
Q, R = modified_gram_schmidt(A)

print("\nQ matrix:")
for r in Q:
    print(r)

print("\nR matrix:")
for r in R:
    print(r)