from TicTacToe_env import TicTacToeEnv
import random
import numpy as np, tensorflow as tf
import pickle, time
import torch

EPISODES = 50_000
learning_rate = 1e-3
discount = 0.999
EPSILON_DECAY = 0.999
epsilon = 1
scores = []
unique_value_list = []
start_q_table = None

l1 = 64
l2 = 150
l3 = 100
l4 = 4
model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)
loss_fn = torch.nn.MSELoss()


if start_q_table is None:
    q_table = np.random.uniform(low=0, high=1, size=([255168] + [9]))
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

def generate_states_list():
    all_states = []
    for a in range(0,3):
        for b in range(0,3):
            for c in range(0,3):
                for d in range(0,3):
                    for e in range(0,3):
                        for f in range(0,3):
                            for g in range(0,3):
                                for h in range(0,3):
                                    for i in range(0,3):
                                        all_states.append(np.array([[a,b,c], [d,e,f], [g,h,i]]))
    for i in range(0, len(all_states)):
        unique_value_list.append(get_unique_value(all_states[i]))
    unique_value_list.sort()
def get_unique_value(state):
    unique_value = 0
    for i in range(0, 3):
        for j in range(0, 3):
            unique_value += 10**(i*3+j)*(state[i,j])
    return unique_value
def get_state_index(state):
    uval = int(get_unique_value(state))
    return unique_value_list.index(uval)
def update(q_table, current_state, action, reward, new_state):
        new_state_index = get_state_index(new_state)
        current_state_index = get_state_index(current_state)
        max_future_q = np.max(q_table[new_state_index])
        current_q = q_table[current_state_index][action]    
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        q_table[current_state_index][action] = new_q

def initial_population():
    epsilon = 1
    env = TicTacToeEnv()
    generate_states_list()
    for i in range(EPISODES):
        env.reset()
        initial_state = env.reset()
        done = False
        score = 0

        while not done:

            if np.random.random() > epsilon:
                action = np.argmax(q_table[get_state_index(initial_state)])
            else:
                action = random.randint(0,8)

            new_state, reward, done, _ = env.step(action)

            if not done:
                update(q_table, initial_state, action, reward, new_state)
            elif done:
                q_table[get_state_index(initial_state)][action] += learning_rate*reward

            initial_state = new_state
            score += reward

        epsilon = epsilon * EPSILON_DECAY
        scores.append(score)
    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)

def play_few_games(n):
    env = TicTacToeEnv()
    generate_states_list()
    env.reset()
    initial_state = env.reset()
    done = False
    for i in range(n):
        while not done:
            action = np.argmax(q_table[get_state_index(initial_state)])
            new_state, reward, done, _ = env.step(action)
            print(new_state, reward)





#play_few_games()
#initial_population()
print(scores)
#print(np.argmax(q_table[0]))