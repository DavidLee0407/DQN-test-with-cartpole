from test_env import TestEnv
import random
import numpy as np, tensorflow as tf
import pickle, time
import torch
import matplotlib.pyplot as plt
from collections import deque

mylist = []
learning_rate = 1e-2
discount = 0.99
EPSILON_DECAY = 0.999
epsilon = 1
loss_fn = torch.nn.MSELoss()
epochs = 15_000
losses = []

#torch.load(r'model_Test.pt')
model = None
if model is None:
    model = torch.nn.Sequential(
    torch.nn.Linear(3, 150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 3))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_state(state):
    return torch.from_numpy(state).float()

def initial_population():
    epsilon = 1
    env = TestEnv()

    for i in range(0, epochs):
        env.reset()
        initial_state_ = env.reset() + np.random.rand(1,3)/10.0
        initial_state = initial_state_.squeeze()
        done = False

        while not done:

            qval = model(get_state(initial_state)).squeeze()
            if np.random.random() > epsilon:
                qval_ = qval.data.numpy()
                action = np.argmax(qval_)
            else:
                action = random.randint(0,2)
            
            new_state_, reward, done = env.step(action)
            new_state = new_state_ + np.random.rand(1,3)/10.0
            #if reward==1: mylist.append(i)

            if not done:
                with torch.no_grad():
                    newQ = model(get_state(new_state))
                maxQ = torch.max(newQ)
                Y = (reward + (discount * maxQ))
            else:
                Y = reward  # learning rate?

            Y = torch.Tensor([Y]).detach()
            #Y.requires_grad = True
            x = qval[action]
            X = torch.tensor([x])
            X.requires_grad = True
            loss = loss_fn(X, Y)

            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            initial_state = new_state

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

initial_population()
plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)
plt.show()
torch.save(model, 'model_Test.pt')
#play_few_games(1)
#print(len(mylist))
with torch.no_grad():
    print(model(torch.Tensor([1,1,0])))
    print(model(torch.Tensor([1,0,1])))
    print(model(torch.Tensor([0,1,1])))
play_few_games(2)
#env_test()
