import numpy as np
import random
import matplotlib.pyplot as plt


# initial setting
# MeanData = list() #10-armed testbed
random.seed(4)
MeanData = np.random.randn(1000, 10) + np.random.randn(10)
# 앞의 식이 10개의 데이터이고, 뒤의 식이 평균 값을 결정하게 된다


# Violin plot
fig, ax = plt.subplots()
violin = ax.violinplot(MeanData, showmeans=True)
ax.set_ylim(-6.0, 6.0)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax.set_xlabel('Action')
ax.set_ylabel('Reward\ndistribution')
plt.axhline(y=0, linestyle='--')
plt.show()


AveReward_1 = []  # epsilon = 0.0 average reward
AveReward_2 = []  # epsilon = 0.1 average reward
Q = np.zeros(10) + 0.0  # Q value


n = 0  # step
Ave_R = 0


def select_bandit(epsilon):
    a = np.random.rand()  # 0과 1 사이의 랜덤 변수를 발생
    max_index = np.argmax(Q)  # Q value 중 가장 큰 것의 index 를 추출
    if a > epsilon:  # Exploitation
        return max_index
    else:  # Exploration
        eg_index = np.random.randint(0, 10)
        return eg_index


# simulation
for y in range(4):
    Q = np.zeros(10) + 2.0
    n = 0
    Ave_R = 0
    AveReward = []
    # greedy(epsilon=0)
    while n <= 100:
        # select bandit by Q
        i = select_bandit(0.0)
        R = np.random.choice(MeanData[:, i], 1)
        Ave_R = (Ave_R * n + R) / (n + 1)
        Q[i] = Q[i] + (R - Q[i]) / (n + 1)
        AveReward.append(Ave_R)
        n += 1
    AveReward_1.append(AveReward)
    Q = np.zeros(10) + 2.0
    n = 0
    Ave_R = 0
    AveReward = []
    # greedy(epsilon=0.1)
    while n <= 100:
        # select bandit by Q
        i = select_bandit(0.1)
        R = np.random.choice(MeanData[:, i], 1)
        Ave_R = (Ave_R * n + R) / (n + 1)
        Q[i] = Q[i] + (R - Q[i]) / (n + 1)
        AveReward.append(Ave_R)
        n += 1
    AveReward_2.append(AveReward)


fig = plt.figure()
ax3 = plt.subplot(2, 2, 1)
plt.plot(AveReward_1[0], label='epsilon = 0.0')
plt.plot(AveReward_2[0], label='epsilon = 0.1')
ax3.set_xlabel('steps')
ax3.set_ylabel('average reward')
plt.legend()


ax4 = plt.subplot(2, 2, 2)
plt.plot(AveReward_1[1], label='epsilon = 0.0')
plt.plot(AveReward_2[1], label='epsilon = 0.1')
ax4.set_xlabel('steps')
ax4.set_ylabel('average reward')
plt.legend()


ax5 = plt.subplot(2, 2, 3)
plt.plot(AveReward_1[2], label='epsilon = 0.0')
plt.plot(AveReward_2[2], label='epsilon = 0.1')
ax5.set_xlabel('steps')
ax5.set_ylabel('average reward')
plt.legend()


ax6 = plt.subplot(2, 2, 4)
plt.plot(AveReward_1[3], label='epsilon = 0.0')
plt.plot(AveReward_2[3], label='epsilon = 0.1')
ax6.set_xlabel('steps')
ax6.set_ylabel('average reward')
plt.legend()
fig.tight_layout()
plt.show()
