import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m = np.amax(vector)     # 해당 쪽의 최대 점수
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(   # 게임 생성
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs = {'map_name' : '4x4', 'is_slippery' : False}
)
env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n]) # 어디로 갈지, 그 곳으로 가면 몇 점일지
num_episode = 2000   # 몇번 실행 할지

rList = []  # 성공회수 기록

for i in range(num_episode):
    state = env.reset()     # 게임 초기화
    rAll = 0    # 게임 진행시 얻게 되는 점수
    done = False    # 게임오버 여부

    while not done:
        action = rargmax(Q[state, :])   # 가장 큰 점수를 따라가지만 모두 동점인 경우 랜덤 행동
        # new_state : 다음 행선지, reward : 이동 할 시의 점수, done : 게임오버 여부, _(info) : 정보
        new_state, reward, done, _ = env.step(action)
        # 지금의 결과 값은 이동 할 시에 돌아오는 reward와 다음 행선지에서 결과값이 합쳐진다.
        # 처음에는 모두 0 이기 때문에 목적지에 도착 하기 전까진 모든 방향은 랜덤
        # 목적지에 도착하게 되면 그 곳의 Q값은 1을 가르키기 때문에 그 때부터 본격적으로 경로 탐색
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate : " + str(sum(rList) / num_episode))
print("Final Q-Table values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()