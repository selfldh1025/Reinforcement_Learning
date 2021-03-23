import numpy as np


d = [[-1, 0], [0, 1], [1, 0], [0, -1]]


def build_grids(n):
    a = np.zeros((2, n, n))
    return a


def opt_policy(v, n, y, x, gamma):
    q = np.zeros(4)
    if y == 0 and x == 1:  # A(10)
        return 10.0 + gamma * v[0][4][1]
    elif y == 0 and x == 3:  # B(5)
        return 5.0 + gamma * v[0][2][3]
    else:
        for i in range(4):
            ny = y + d[i][0]
            nx = x + d[i][1]
            if ny == -1:
                ny = y
                r = -1.0
            elif nx == -1:
                nx = x
                r = -1.0
            elif ny == n:
                ny = n - 1
                r = -1.0
            elif nx == n:
                nx = n - 1
                r = -1.0
            else:
                r = 0.0
            q[i] = r + gamma * v[0][ny][nx]
        return max(q)


def eq_random_policy(v, n, y, x, gamma):
    if y == 0 and x == 1:  # A(10)
        return 10.0 + gamma * v[0][4][1]
    elif y == 0 and x == 3:  # B(5)
        return 5.0 + gamma * v[0][2][3]
    else:
        sum_v = 0.0
        for i in range(4):
            ny = y + d[i][0]
            nx = x + d[i][1]
            if ny == -1:
                ny = y
                r = -1.0
            elif nx == -1:
                nx = x
                r = -1.0
            elif ny == n:
                ny = n - 1
                r = -1.0
            elif nx == n:
                nx = n - 1
                r = -1.0
            else:
                r = 0.0
            sum_v = sum_v + 0.25 * (r + gamma * v[0][ny][nx])
        return sum_v


def bellman_eq(n):
    # Equiprobable-random policy
    time_step = 0
    v = build_grids(n)
    gamma = 0.9

    while True:
        if time_step > 1000:
            break
        for i in range(n):
            for j in range(n):
                v[1][i][j] = eq_random_policy(v, n, i, j, gamma)

        for i in range(n):
            for j in range(n):
                v[0][i][j] = v[1][i][j]
                v[1][i][j] = 0

        time_step += 1

    for i in range(n):
        for j in range(n):
            print(round(v[0][i][j], 1), end=' ')
        print('')

    print('\n')
    # optimal policy
    time_step = 0
    v = build_grids(n)
    gamma = 0.9

    while True:
        if time_step > 1000:
            break
        for i in range(n):
            for j in range(n):
                v[1][i][j] = opt_policy(v, n, i, j, gamma)

        for i in range(n):
            for j in range(n):
                v[0][i][j] = v[1][i][j]
                v[1][i][j] = 0

        time_step += 1

    for i in range(n):
        for j in range(n):
            print(round(v[0][i][j], 1), end=' ')
        print('')


# simulation
bellman_eq(5)
