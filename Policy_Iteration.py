import numpy as np


WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]


def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


def step(state, action):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward


def policy_evaluation(c_policy, in_place=True, discount=0.99):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                action = ACTIONS[int(c_policy[i][j])]
                (ni, nj), reward = step([i, j], action)
                value = (reward + discount * state_values[ni, nj])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


def policy_improvement(value):
    new_policy = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            max_v = float("-inf")
            for k in range(4):
                (ni, nj), reward = step([i, j], ACTIONS[k])
                if max_v < value[ni][nj]:
                    max_v = value[ni][nj]
                    new_policy[i][j] = k
    return new_policy


# policy 출력부
def print_policy(_policy):
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if _policy[i][j] == 0:
                print("◀", end=" ")
            elif _policy[i][j] == 1:
                print("▲", end=" ")
            elif _policy[i][j] == 2:
                print("▶", end=" ")
            elif _policy[i][j] == 3:
                print("▼", end=" ")
        print()
    return


# simulation
policy = np.zeros((WORLD_SIZE, WORLD_SIZE))
opt_policy = np.ones((WORLD_SIZE, WORLD_SIZE))
t = 0

while True:
    new_values, sync_iteration = policy_evaluation(opt_policy, in_place=True)
    if np.array_equal(policy, opt_policy):
        break
    else:
        policy = opt_policy
        opt_policy = policy_improvement(new_values)
    t += 1
    print(f'{t} times')
    print("Number of iteration ", sync_iteration)
    print("State value")
    print(np.round(new_values, 1), "\n")
    print("Policy")
    print_policy(opt_policy)
    print()
