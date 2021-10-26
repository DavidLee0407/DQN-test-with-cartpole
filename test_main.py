from test_env import TestEnv
import random
import numpy as np, tensorflow as tf
import pickle, time
import torch
import matplotlib.pyplot as plt
from collections import deque
from exp_rep import Experience_Replay

scores = []
mylist = []
learning_rate = 1e-3
discount = 0.95
EPSILON_DECAY = 0.99
epsilon = 1
loss_fn = torch.nn.HuberLoss()
epochs = 2_000
losses = []

model = None
if model is None:
    model = torch.nn.Sequential(
    torch.nn.Linear(3, 64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, 256),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(128, 3))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_state(state):
    return torch.from_numpy(state).float()

def train_model():
    epsilon = 1
    env = TestEnv()
    score = 0
    memory = Experience_Replay()

    for i in range(0, epochs):
        env.reset()
        initial_state_ = env.reset() 
        initial_state = initial_state_.squeeze()
        done = False
        score = 0

        while not done:

            qval = model(get_state(initial_state)).squeeze()
            if np.random.random() > epsilon:
                qval_ = qval.data.numpy()
                action = np.argmax(qval_)
            else:
                action = random.randint(0,2)
            
            new_state, reward, done = env.step(action)
            done_mask = 1.0 if done else 0.0
            memory.put((initial_state, action, reward, new_state, done_mask))
            initial_state = new_state
            score += reward
            if done:break

            if len(memory.buffer) > memory.min_len:
                s_T, a_T, r_T, s2_T, d_T = memory.sample()
                Q1 = model(s_T)
                with torch.no_grad():
                    Q2 = model(s2_T)
                    max_Q2 = Q2.max(1)[0].unsqueeze(1)

                Y = (r_T + discount * ((1 - d_T) * max_Q2)).float()
                X = Q1.gather(1, a_T)

                loss = loss_fn(X, Y.detach())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        scores.append(score)
        if epsilon > 0.1:
            epsilon = epsilon * EPSILON_DECAY
        else:
            epsilon = 0.1

def play_few_games(n):
    env = TestEnv()
    env.reset()
    initial_state = env.reset()
    done = False
    for i in range(n):
        while not done:
            qval = model(get_state(initial_state))
            qval_ = qval.data.numpy()
            action = np.argmax(qval_)
            new_state, reward, done = env.step(action)
            print(new_state, reward)

def env_test():
    env = TestEnv()
    for i in range(1):
        env.reset()
        done = False
        while not done:
            action = random.randint(0,2)
            new_state, reward, done = env.step(action)
            print(new_state, reward)
    #env_test()

def show_graph():
    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Loss",fontsize=22)
    plt.show()
    #torch.save(model, 'model_Test.pt')
    plt.plot(scores)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Scores",fontsize=22)
    plt.show()
    with torch.no_grad():
        print(model(torch.Tensor([0,1,1])))
        print(model(torch.Tensor([1,0,1])))
        print(model(torch.Tensor([1,1,0])))
        print(model(torch.Tensor([0,0,0])))

train_model()
show_graph()

