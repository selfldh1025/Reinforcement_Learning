import numpy as np

d = [[-1, 0], [0, -1], [0, 1], [1, 0]]  # up, left, right, down
N = 5

M = np.zeros((N, N))  # Grid
A = np.zeros((N * N, N * N))  # A matrix
B = np.zeros((N * N, 1))  # B matrix

# Equiprobable random policy
for y in range(N):
    for x in range(N):
        i = y * 5 + x
        A[i][i] = 1
        if i == 1:
            A[i][21] = A[i][21] - 0.9
            B[i] = 10
        elif i == 3:
            A[i][13] = A[i][13] - 0.9
            B[i] = 5
        else:
            for k in range(4):
                ny = y + d[k][0]
                nx = x + d[k][1]
                if ny == -1:
                    ny = y
                    R = -1.0
                elif nx == -1:
                    nx = x
                    R = -1.0
                elif ny == N:
                    ny = N - 1
                    R = -1.0
                elif nx == N:
                    nx = N - 1
                    R = -1.0
                else:
                    R = 0.0
                ni = ny * 5 + nx
                A[i][ni] = A[i][ni] - 0.25 * 0.9
                B[i] = B[i] + 0.25 * R

# IA = np.lineal.inv(A)  # Inverse Matrix 구하는 함수
# V = np.dot(IA, B) # V = A-1B
V = np.linalg.solve(A, B)
# print(np.round(V, 1))

for i in range(25):
    if i % 5 == 0:
        print('\n')
    print(np.round(V[i], 1), end='\t')

print('\n')

M = np.zeros((N, N))  # Grid
A = np.zeros((N * N, N * N))  # A matrix
B = np.zeros((N * N, 1))  # B matrix

# optimal policy
for y in range(N):
    for x in range(N):
        i = y * 5 + x
        A[i][i] = 1
        if i == 0:
            A[i][1] = A[i][1] - 0.9
        elif i == 1:
            A[i][21] = A[i][21] - 0.9
            B[i] = 10
        elif i == 3:
            A[i][13] = A[i][13] - 0.9
            B[i] = 5
        elif i == 2 or i == 4 or i == 8 or i == 9:
            A[i][i - 1] = A[i][i - 1] - 0.9
        elif i == 5 or i == 10 or i == 15 or i == 20:
            A[i][i - 5] = A[i][i - 5] - 0.5 * 0.9
            A[i][i + 1] = A[i][i + 1] - 0.5 * 0.9
        elif i == 6 or i == 11 or i == 16 or i == 21:
            A[i][i - 5] = A[i][i - 5] - 0.9
        else:
            A[i][i - 5] = A[i][i - 5] - 0.5 * 0.9
            A[i][i - 1] = A[i][i - 1] - 0.5 * 0.9

IA = np.linalg.inv(A)  # Inverse Matrix 구하는 함수
V = np.dot(IA, B)  # V = A-1B

for i in range(25):
    if i % 5 == 0:
        print('\n')
    print(np.round(V[i], 1), end='\t')
