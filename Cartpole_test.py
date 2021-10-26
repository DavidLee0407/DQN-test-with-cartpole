import torch, gym, numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

batch_size = 32
max_len = 50_000
scores = []
learning_rate, discount, epochs = 1e-3, 0.95, 10_000
EPSILON_DECAY = 0.999
loss_fn = torch.nn.MSELoss()
model = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2),
    torch.nn.Softmax())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def change_to_tensor(np):
    return torch.from_numpy(np).float()

def test_env(n):
    score_list = []
    env = gym.make('CartPole-v0')
    for i in range(n):
        done = False
        score = 0
        env.reset()
        init_state = env.reset()
        while not done:
            env.render()
            qval = model(change_to_tensor(init_state))
            qval_ = qval.data.numpy()
            action = np.argmax(qval_)
            new_observation, reward, done, _ = env.step(action)
            init_state = new_observation
            score += reward
        score_list.append(score)
        env.close()
    return score_list

def train_model():
    env = gym.make('CartPole-v0')
    epsilon = 0.5

    for i in range(0, epochs):
        env.reset()
        initial_state = env.reset()
        done = False
        score = 0
        qval = model(change_to_tensor(initial_state))

        while not done:
            if np.random.random() > epsilon:
                qval_ = qval.data.numpy()
                action = np.argmax(qval_)
            else:
                action = env.action_space.sample()
            
            new_state, reward, done, _ = env.step(action)

            if not done:
                with torch.no_grad():
                    newQ = model(change_to_tensor(new_state))
                maxQ = torch.max(newQ)
                Y = (reward + (discount * maxQ))
            else:
                Y = reward  

            score += reward
            Y = torch.Tensor([Y]).detach()
            X = torch.Tensor([qval[action]])
            X.requires_grad = True
            loss = loss_fn(X, Y)

            optimizer.zero_grad()
            optimizer.step()

            initial_state = new_state

        scores.append(score)
        if epsilon > 0.05:
            epsilon = epsilon * EPSILON_DECAY
        else:
            epsilon = 0

def draw_graph():
    plt.figure(figsize=(10,7))
    plt.plot(scores)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Rewards",fontsize=22)
    plt.show()
    #torch.save(model, 'model_Test.pt')
    
train_model()
draw_graph()
#print(test_env(3))
